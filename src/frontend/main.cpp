#include "emulator.hpp"
#include "gpu/vulkan/vulkan_renderer.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_filesystem.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
const char* current_stage = "startup";
bool sdl_initialized = false;

struct Game
{
    std::string title;
    fs::path rpx_path;
};

std::string path_utf8(const fs::path& path)
{
    const auto encoded = path.u8string();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

fs::path path_from_utf8(std::string_view text)
{
    std::u8string encoded;
    encoded.reserve(text.size());
    for (const unsigned char byte : text)
    {
        encoded.push_back(static_cast<char8_t>(byte));
    }
    return fs::path(encoded);
}

std::string lower_ascii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::vector<fs::path> load_folders(const fs::path& config_file)
{
    std::vector<fs::path> folders;
    std::ifstream input(config_file);
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty())
        {
            folders.push_back(path_from_utf8(line));
        }
    }
    return folders;
}

void save_folders(const fs::path& config_file, const std::vector<fs::path>& folders)
{
    std::ofstream output(config_file, std::ios::trunc);
    for (const fs::path& folder : folders)
    {
        output << path_utf8(folder) << '\n';
    }
}

std::vector<Game> discover_games(const std::vector<fs::path>& roots)
{
    // A normal extracted title has its main executable under code/. Treat all
    // RPXs in that directory as candidates for one title, preferring a file
    // named after the title directory or main.rpx. Other loose RPXs are kept
    // individually so developer folders remain useful too.
    std::vector<fs::path> candidates;
    std::error_code error;
    for (const fs::path& root : roots)
    {
        if (!fs::is_directory(root, error))
        {
            error.clear();
            continue;
        }
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, error);
        const fs::recursive_directory_iterator end;
        while (it != end)
        {
            if (error)
            {
                error.clear();
                it.increment(error);
                continue;
            }
            if (it->is_regular_file(error) && lower_ascii(path_utf8(it->path().extension())) == ".rpx")
            {
                candidates.push_back(it->path());
            }
            error.clear();
            it.increment(error);
        }
    }

    std::vector<Game> games;
    std::vector<bool> consumed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        if (consumed[i])
        {
            continue;
        }
        const fs::path& candidate = candidates[i];
        if (lower_ascii(path_utf8(candidate.parent_path().filename())) == "code")
        {
            const fs::path title_dir = candidate.parent_path().parent_path();
            std::vector<std::size_t> group;
            for (std::size_t j = i; j < candidates.size(); ++j)
            {
                if (lower_ascii(path_utf8(candidates[j].parent_path().filename())) == "code" &&
                    candidates[j].parent_path().parent_path() == title_dir)
                {
                    consumed[j] = true;
                    group.push_back(j);
                }
            }
            if (!group.empty())
            {
                const std::string expected = lower_ascii(path_utf8(title_dir.filename()));
                const auto rank = [&](std::size_t index)
                {
                    const std::string stem = lower_ascii(path_utf8(candidates[index].stem()));
                    return stem == expected ? 0 : (stem == "main" ? 1 : 2);
                };
                const auto best = *std::min_element(group.begin(), group.end(), [&](auto a, auto b)
                {
                    if (rank(a) != rank(b)) return rank(a) < rank(b);
                    return candidates[a].filename() < candidates[b].filename();
                });
                const std::string title_directory = path_utf8(title_dir.filename());
                const std::string label = title_directory.empty()
                    ? path_utf8(candidates[best].stem())
                    : title_directory;
                games.push_back({label, candidates[best]});
            }
        }
        else
        {
            consumed[i] = true;
            games.push_back({path_utf8(candidate.stem()), candidate});
        }
    }

    std::sort(games.begin(), games.end(), [](const Game& a, const Game& b)
    {
        return lower_ascii(a.title) < lower_ascii(b.title);
    });
    return games;
}

struct LibraryScan
{
    std::mutex mutex;
    bool running{};
    bool result_ready{};
    std::vector<Game> games;
    std::string error;
};

void start_library_scan(
    const std::vector<fs::path>& roots,
    LibraryScan& scan,
    std::jthread& thread)
{
    if (thread.joinable())
    {
        thread.join();
    }
    {
        std::lock_guard lock(scan.mutex);
        scan.running = true;
        scan.result_ready = false;
        scan.error.clear();
    }

    thread = std::jthread([roots, &scan]
    {
        std::vector<Game> discovered;
        std::string error;
        try
        {
            discovered = discover_games(roots);
        }
        catch (const std::exception& exception)
        {
            error = exception.what();
        }
        catch (...)
        {
            error = "Unknown error while scanning game folders.";
        }

        std::lock_guard lock(scan.mutex);
        scan.games = std::move(discovered);
        scan.error = std::move(error);
        scan.running = false;
        scan.result_ready = true;
    });
}

void append_fallback_profile(
    std::ostringstream& text,
    const affogato::cpu::espresso::RecompilerStatistics& statistics)
{
    std::vector<std::pair<std::uint16_t, std::uint64_t>> fallbacks(
        statistics.fallbacks_by_opcode.begin(), statistics.fallbacks_by_opcode.end());
    std::ranges::sort(fallbacks, [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });
    if (fallbacks.empty()) return;

    text << "\nTop interpreter fallback opcodes:";
    const std::size_t count = std::min<std::size_t>(8U, fallbacks.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto [opcode, executions] = fallbacks[i];
        text << "\n  "
             << affogato::cpu::espresso::opcode_mnemonic(
                    static_cast<affogato::cpu::espresso::Opcode>(opcode))
             << ": " << executions;
    }
}

std::string format_result(const affogato::EmulatorRunResult& result)
{
    using affogato::cpu::espresso::StopReason;
    std::ostringstream text;
    text << "Loaded entry point: 0x" << std::hex << std::uppercase << result.image.entry_point
         << "\nSections loaded: " << std::dec << result.image.loaded_sections
         << "\nExecuted instructions: " << result.execution.steps
         << "\nStop CIA: 0x" << std::hex << std::uppercase << result.execution.cia
         << "\nGPR3: " << std::dec << result.gpr3;
    if (result.mode == affogato::ExecutionMode::recompiler)
    {
        text << "\nExecution mode: experimental x86-64 recompiler"
             << "\nNative instructions: " << result.recompiler_statistics.native_instructions
             << "\nInterpreter fallbacks: " << result.recompiler_statistics.interpreter_fallbacks
             << "\nCompiled blocks: " << result.recompiler_statistics.compiled_blocks
             << "\nBlock-cache hits: " << result.recompiler_statistics.cache_hits
             << "\nFast lookup hits: " << result.recompiler_statistics.fast_lookup_hits
             << "\nJIT compilation time: "
             << result.recompiler_statistics.compilation_time_ns / 1'000'000U << " ms"
             << "\nNative block execution time: "
             << result.recompiler_statistics.native_execution_time_ns / 1'000'000U
             << " ms";
        append_fallback_profile(text, result.recompiler_statistics);
    }
    switch (result.execution.reason)
    {
    case StopReason::instruction_limit:
        text << "\nStopped: instruction limit";
        break;
    case StopReason::unsupported_instruction:
        text << "\nStopped: unsupported PowerPC instruction 0x" << std::hex
             << std::uppercase << result.execution.instruction_word;
        break;
    case StopReason::unimplemented_hle_call:
        text << "\nStopped: unimplemented HLE call " << result.execution.hle_call;
        break;
    case StopReason::hle_error:
        text << "\nStopped: HLE error";
        if (!result.execution.detail.empty()) text << "\nDetail: " << result.execution.detail;
        break;
    case StopReason::memory_fault:
        text << "\nStopped: memory fault";
        if (!result.execution.detail.empty()) text << "\nDetail: " << result.execution.detail;
        break;
    }
    const std::string history =
        affogato::cpu::espresso::format_instruction_history(result.execution);
    if (!history.empty()) text << '\n' << history;
    if (result.gpr3 == 42)
    {
        text << "\nReturned value observed in r3: 42";
    }
    bool has_gx2_calls = false;
    for (const auto& statistic : result.hle_call_statistics)
    {
        if (statistic.name.starts_with("gx2::") || statistic.name.starts_with("coreinit::GX2"))
        {
            if (!has_gx2_calls)
            {
                text << "\nGX2 HLE calls:";
                has_gx2_calls = true;
            }
            text << "\n  " << statistic.name << ": " << statistic.calls;
        }
    }
    return text.str();
}

std::string format_progress(
    const affogato::EmulatorRunResult& result,
    std::size_t total_steps,
    std::chrono::milliseconds file_read_time,
    std::chrono::milliseconds rpx_load_time,
    std::chrono::milliseconds guest_run_time)
{
    std::ostringstream text;
    text << "RPX execution in progress"
         << "\nLoaded entry point: 0x" << std::hex << std::uppercase
         << result.image.entry_point
         << "\nSections loaded: " << std::dec << result.image.loaded_sections
         << "\nExecuted instructions: " << total_steps
         << "\nCurrent CIA: 0x" << std::hex << std::uppercase
         << result.execution.cia
         << "\nGPR3: " << std::dec << result.gpr3
         << "\nFile read: " << file_read_time.count() << " ms"
         << "\nRPX load: " << rpx_load_time.count() << " ms"
         << "\nGuest run so far: " << guest_run_time.count() << " ms";
    if (result.mode == affogato::ExecutionMode::recompiler)
    {
        text << "\nNative instructions: " << result.recompiler_statistics.native_instructions
             << "\nInterpreter fallbacks: " << result.recompiler_statistics.interpreter_fallbacks
             << "\nCompiled blocks: " << result.recompiler_statistics.compiled_blocks
             << "\nBlock-cache hits: " << result.recompiler_statistics.cache_hits
             << "\nFast lookup hits: " << result.recompiler_statistics.fast_lookup_hits
             << "\nJIT compilation time: "
             << result.recompiler_statistics.compilation_time_ns / 1'000'000U << " ms"
             << "\nNative block execution time: "
             << result.recompiler_statistics.native_execution_time_ns / 1'000'000U
             << " ms";
        append_fallback_profile(text, result.recompiler_statistics);
    }
    std::uint64_t gx2_call_total = 0U;
    for (const auto& statistic : result.hle_call_statistics)
    {
        if (statistic.name.starts_with("gx2::") || statistic.name.starts_with("coreinit::GX2"))
        {
            gx2_call_total += statistic.calls;
        }
    }
    text << "\nGX2 HLE calls observed: " << gx2_call_total;
    const std::string history =
        affogato::cpu::espresso::format_instruction_history(result.execution);
    if (!history.empty()) text << "\nRecent instruction trace:\n" << history;
    return text.str();
}

int run_rpx_diagnostic(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: AffogatoFrontend.exe --run-rpx <file.rpx> "
                     "[--recompiler] [--limit <instructions>] "
                     "[--vpad-buttons <mask>] "
                     "[--watch-write <guest-address>] "
                     "[--trace-hle <library::symbol>] "
                     "[--dump-memory <address> <bytes>]... "
                     "[--dump-code <address> <words>]...\n";
        return 2;
    }

    affogato::ExecutionMode mode = affogato::ExecutionMode::interpreter;
    std::size_t instruction_limit = 5'000'000'000ULL;
    std::optional<std::uint32_t> virtual_vpad_buttons;
    std::optional<std::uint32_t> watched_guest_write_address;
    std::optional<std::string> hle_trace_filter;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> guest_memory_dumps;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> guest_code_dumps;
    for (int index = 3; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--recompiler")
        {
            mode = affogato::ExecutionMode::recompiler;
        }
        else if (argument == "--limit" && index + 1 < argc)
        {
            const unsigned long long parsed = std::stoull(argv[++index]);
            if (parsed == 0ULL || parsed > std::numeric_limits<std::size_t>::max())
            {
                throw std::invalid_argument("--limit must be in [1, SIZE_MAX]");
            }
            instruction_limit = static_cast<std::size_t>(parsed);
        }
        else if (argument == "--vpad-buttons" && index + 1 < argc)
        {
            const std::string_view mask_text(argv[++index]);
            std::size_t parsed_length = 0U;
            const unsigned long long parsed =
                std::stoull(std::string(mask_text), &parsed_length, 0);
            if (parsed_length != mask_text.size() ||
                parsed > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument(
                    "--vpad-buttons requires a 32-bit integer mask");
            }
            virtual_vpad_buttons = static_cast<std::uint32_t>(parsed);
        }
        else if (argument == "--watch-write" && index + 1 < argc)
        {
            const std::string_view address_text(argv[++index]);
            std::size_t parsed_length = 0U;
            const unsigned long long parsed =
                std::stoull(std::string(address_text), &parsed_length, 0);
            if (parsed_length != address_text.size() ||
                parsed > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument(
                    "--watch-write requires a 32-bit guest address");
            }
            watched_guest_write_address = static_cast<std::uint32_t>(parsed);
        }
        else if (argument == "--trace-hle" && index + 1 < argc)
        {
            hle_trace_filter = argv[++index];
            if (hle_trace_filter->find("::") == std::string::npos)
            {
                throw std::invalid_argument(
                    "--trace-hle requires an exact library::symbol name");
            }
        }
        else if (argument == "--dump-memory" && index + 2 < argc)
        {
            const std::string_view address_text(argv[++index]);
            const std::string_view size_text(argv[++index]);
            std::size_t parsed_address_length = 0U;
            std::size_t parsed_size_length = 0U;
            const unsigned long long address =
                std::stoull(std::string(address_text), &parsed_address_length, 0);
            const unsigned long long size =
                std::stoull(std::string(size_text), &parsed_size_length, 0);
            if (parsed_address_length != address_text.size() ||
                address > std::numeric_limits<std::uint32_t>::max() ||
                parsed_size_length != size_text.size() || size < 4U || size > 256U ||
                (address & 3U) != 0U || (size & 3U) != 0U)
            {
                throw std::invalid_argument(
                    "--dump-memory requires an aligned address and 4..256-byte "
                    "multiple-of-four size");
            }
            guest_memory_dumps.emplace_back(
                static_cast<std::uint32_t>(address),
                static_cast<std::uint32_t>(size));
        }
        else if (argument == "--dump-code" && index + 2 < argc)
        {
            const std::string_view address_text(argv[++index]);
            const std::string_view word_count_text(argv[++index]);
            std::size_t parsed_address_length = 0U;
            std::size_t parsed_count_length = 0U;
            const unsigned long long address =
                std::stoull(std::string(address_text), &parsed_address_length, 0);
            const unsigned long long word_count =
                std::stoull(std::string(word_count_text), &parsed_count_length, 0);
            if (parsed_address_length != address_text.size() ||
                address > std::numeric_limits<std::uint32_t>::max() ||
                (address & 3U) != 0U ||
                parsed_count_length != word_count_text.size() ||
                word_count == 0U || word_count > 64U ||
                address + word_count * sizeof(std::uint32_t) >
                    (std::uint64_t{1U} << 32U))
            {
                throw std::invalid_argument(
                    "--dump-code requires an aligned guest address and 1..64 words");
            }
            guest_code_dumps.emplace_back(
                static_cast<std::uint32_t>(address),
                static_cast<std::uint32_t>(word_count));
        }
        else
        {
            throw std::invalid_argument("Unknown or incomplete RPX diagnostic option: " +
                std::string(argument));
        }
    }

    const fs::path path = path_from_utf8(argv[2]);
    const auto read_start = std::chrono::steady_clock::now();
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("Could not open RPX file: " + path_utf8(path));
    }
    const auto length = input.tellg();
    if (length < 0)
    {
        throw std::runtime_error("Could not determine RPX file size");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    if (!bytes.empty() && !input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())))
    {
        throw std::runtime_error("Could not read the complete RPX file");
    }
    const auto read_end = std::chrono::steady_clock::now();
    const auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        read_end - read_start);

    affogato::Emulator emulator;
    const auto load_start = std::chrono::steady_clock::now();
    const auto image = emulator.load_rpx(
        bytes, path.parent_path().parent_path() / "content");
    // The command-line diagnostic has no scanout/display thread. Keep virtual
    // vsync accounting, but avoid a host sleep per guest frame.
    emulator.core().gx2.host_vsync_pacing_enabled = false;
    if (virtual_vpad_buttons.has_value())
    {
        emulator.core().virtual_vpad_connected = true;
        emulator.core().virtual_vpad_buttons = *virtual_vpad_buttons;
    }
    if (watched_guest_write_address.has_value())
    {
        emulator.core().memory.set_write_watchpoint(
            *watched_guest_write_address, sizeof(std::uint32_t));
        emulator.core().memory.set_write_watch_gpr_source(
            &emulator.core().state.gpr);
    }
    emulator.core().hle.set_recent_call_trace_enabled(true);
    if (hle_trace_filter.has_value())
    {
        emulator.core().hle.set_recent_call_trace_filter(*hle_trace_filter);
    }
    const auto load_end = std::chrono::steady_clock::now();
    const auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start);
    const auto execution_start = std::chrono::steady_clock::now();

    std::cout << "Loaded entry point: 0x" << std::hex << std::uppercase
              << image.entry_point << std::dec
              << "\nSections loaded: " << image.loaded_sections
              << "\nHLE data range: [0x" << std::hex << std::uppercase
              << emulator.core().hle.data_range_begin() << ",0x"
              << emulator.core().hle.data_range_end() << ") import range: [0x"
              << emulator.core().hle.import_range_begin() << ",0x"
              << emulator.core().hle.import_range_end() << ")" << std::dec
              << "\nExecution mode: " << (mode == affogato::ExecutionMode::recompiler
                    ? "recompiler" : "interpreter")
              << "\nVSync pacing: disabled (headless diagnostic)"
              << "\nInstruction limit: " << instruction_limit << '\n';
    if (virtual_vpad_buttons.has_value())
    {
        std::cout << "Virtual VPAD button mask: 0x" << std::hex << std::uppercase
                  << *virtual_vpad_buttons << std::dec << '\n';
    }

    constexpr std::size_t chunk_size = 1'000'000U;
    constexpr std::size_t progress_interval = 10'000'000U;
    std::size_t total_steps = 0U;
    std::size_t next_progress = progress_interval;
    affogato::EmulatorRunResult result{};
    while (total_steps < instruction_limit)
    {
        const std::size_t current_chunk = std::min(
            chunk_size, instruction_limit - total_steps);
        result = emulator.run(current_chunk, mode);
        total_steps += result.execution.steps;
        result.execution.steps = total_steps;

        if (total_steps >= next_progress &&
            result.execution.reason == affogato::cpu::espresso::StopReason::instruction_limit)
        {
            std::cout << "Progress: " << total_steps << " instructions; CIA=0x"
                      << std::hex << std::uppercase << result.execution.cia
                      << std::dec << "; r3=" << result.gpr3 << '\n';
            next_progress += progress_interval;
        }
        if (result.execution.reason !=
                affogato::cpu::espresso::StopReason::instruction_limit ||
            total_steps >= instruction_limit)
        {
            break;
        }
    }

    const auto execution_end = std::chrono::steady_clock::now();
    const auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        execution_end - execution_start);
    std::cout << format_result(result)
              << "\nGPR4: " << emulator.core().state.gpr[4]
              << "\nGPR5: " << emulator.core().state.gpr[5]
              << "\nCR: 0x" << std::hex << std::uppercase
              << emulator.core().state.cr
              << " LR: 0x" << emulator.core().state.lr
              << " CTR: 0x" << emulator.core().state.ctr
              << " XER: 0x" << emulator.core().state.xer << std::dec
              << "\nPhase timings: file read " << read_time.count()
              << " ms; RPX load " << load_time.count()
              << " ms; guest execution " << execution_time.count() << " ms.\n"
              << "Guest thread switches: " << emulator.core().guest_thread_switches
              << "; current thread: 0x" << std::hex << std::uppercase
              << emulator.core().current_thread_address << std::dec << '\n'
              << "Guest threads:\n";
    std::cout << "GPRs:";
    for (std::size_t reg = 0U; reg < emulator.core().state.gpr.size(); ++reg)
    {
        if ((reg % 4U) == 0U)
        {
            std::cout << '\n';
        }
        std::cout << "  r" << reg << "=0x" << std::hex << std::uppercase
                  << emulator.core().state.gpr[reg] << std::dec;
    }
    std::cout << '\n';
    if (watched_guest_write_address.has_value())
    {
        const auto watched_writes =
            emulator.core().memory.write_watch_events_snapshot();
        std::cout << "Guest-memory write watch at 0x" << std::hex
                  << std::uppercase << *watched_guest_write_address << std::dec
                  << ": " << watched_writes.size() << " overlapping write(s)\n";
        for (const auto& event : watched_writes)
        {
            std::cout << "  CIA 0x" << std::hex << std::uppercase
                      << event.writer_cia << " access 0x"
                      << event.access_address << "+0x" << event.access_width
                      << " -> watched value 0x" << event.watched_value
                      << std::dec << '\n';
            if (event.has_gpr_snapshot)
            {
                std::cout << "    GPRs at write:";
                for (std::size_t reg = 0U; reg < event.gpr_snapshot.size(); ++reg)
                {
                    if ((reg % 4U) == 0U)
                    {
                        std::cout << '\n';
                    }
                    std::cout << "      r" << reg << "=0x" << std::hex
                              << std::uppercase << event.gpr_snapshot[reg]
                              << std::dec;
                }
                std::cout << '\n';
            }
            for (std::int32_t offset = -32; offset <= 32; offset += 4)
            {
                const std::uint32_t cia = event.writer_cia +
                    static_cast<std::uint32_t>(offset);
                try
                {
                    const std::uint32_t word =
                        emulator.core().memory.read32_be(cia);
                    const auto instruction =
                        affogato::cpu::espresso::decode(word);
                    std::cout << "    0x" << std::hex << std::uppercase << cia
                              << (cia == event.writer_cia ? " * " : "   ")
                              << "0x" << word << ' '
                              << affogato::cpu::espresso::opcode_mnemonic(
                                     instruction.opcode)
                              << std::dec << '\n';
                }
                catch (const std::exception&)
                {
                    std::cout << "    0x" << std::hex << std::uppercase << cia
                              << "   <unmapped>" << std::dec << '\n';
                }
            }
        }
    }
    for (const auto& [dump_address, dump_size] : guest_memory_dumps)
    {
        std::cout << "Guest-memory dump at 0x" << std::hex << std::uppercase
                  << dump_address << " (" << dump_size
                  << " bytes, big-endian words):\n";
        try
        {
            for (std::uint32_t offset = 0U; offset < dump_size; offset += 16U)
            {
                std::cout << "  0x" << std::setw(8) << std::setfill('0')
                          << (dump_address + offset) << ':';
                for (std::uint32_t word_offset = 0U;
                     word_offset < 16U && offset + word_offset < dump_size;
                     word_offset += 4U)
                {
                    std::cout << " 0x" << std::setw(8) << std::setfill('0')
                              << emulator.core().memory.read32_be(
                                     dump_address + offset + word_offset);
                }
                std::cout << '\n';
            }
            std::cout << std::setfill(' ') << std::dec;
        }
        catch (const std::exception& error)
        {
            std::cout << "  <dump failed: " << error.what() << ">\n";
        }
    }
    for (const auto& [dump_address, word_count] : guest_code_dumps)
    {
        std::cout << "Guest-code dump at 0x" << std::hex << std::uppercase
                  << dump_address << " (" << std::dec << word_count
                  << " words):\n";
        if (const auto name = emulator.core().hle.import_name(dump_address);
            name.has_value())
        {
            std::cout << "  HLE import: " << *name << '\n';
        }
        for (std::uint32_t index = 0U; index < word_count; ++index)
        {
            const std::uint32_t cia =
                dump_address + index * sizeof(std::uint32_t);
            try
            {
                const std::uint32_t raw = emulator.core().memory.read32_be(cia);
                const auto instruction =
                    affogato::cpu::espresso::decode(raw);
                std::cout << "  0x" << std::hex << std::uppercase
                          << std::setw(8) << std::setfill('0') << cia
                          << ": 0x" << std::setw(8) << raw << std::setfill(' ')
                          << "  " << affogato::cpu::espresso::opcode_mnemonic(
                                 instruction.opcode);
                if (instruction.opcode !=
                    affogato::cpu::espresso::Opcode::unsupported)
                {
                    std::cout << " fields(rd=" << std::dec
                              << ((raw >> 21U) & 0x1FU)
                              << ",ra=" << ((raw >> 16U) & 0x1FU)
                              << ",rb=" << ((raw >> 11U) & 0x1FU)
                              << ",rc=" << ((raw >> 6U) & 0x1FU) << ')'
                              << " decoded-imm=" << instruction.immediate;
                    if (instruction.record)
                    {
                        std::cout << " Rc=1";
                    }
                }
                std::cout << '\n';
            }
            catch (const std::exception& error)
            {
                std::cout << "  0x" << std::hex << std::uppercase
                          << cia << " <dump failed: " << error.what() << ">\n"
                          << std::dec;
                break;
            }
        }
        std::cout << std::dec;
    }
    if (result.execution.hle_call == "coreinit::OSReceiveMessage")
    {
        const std::uint32_t queue = emulator.core().state.gpr[3];
        std::cout << "OSReceiveMessage args: queue=0x" << std::hex << std::uppercase
                  << queue << " output=0x" << emulator.core().state.gpr[4]
                  << " flags=0x" << emulator.core().state.gpr[5] << std::dec;
        try
        {
            std::cout << " messages=0x" << std::hex << std::uppercase
                      << emulator.core().memory.read32_be(queue + 0x2CU)
                      << " capacity=" << std::dec
                      << emulator.core().memory.read32_be(queue + 0x30U)
                      << " first="
                      << emulator.core().memory.read32_be(queue + 0x34U)
                      << " used="
                      << emulator.core().memory.read32_be(queue + 0x38U);
        }
        catch (const std::exception& error)
        {
            std::cout << " queue-state-error=" << error.what();
        }
        std::cout << '\n';
    }
    for (const auto& thread : emulator.core().guest_threads)
    {
        std::cout << "  0x" << std::hex << std::uppercase << thread.address
                  << " state=0x" << static_cast<unsigned>(emulator.core().memory.read8(
                         thread.address + 0x324U))
                  << " suspend=" << std::dec << emulator.core().memory.read32_be(
                         thread.address + 0x328U)
                  << " priority=" << emulator.core().memory.read32_be(
                         thread.address + 0x32CU)
                  << " affinity=0x" << std::hex << static_cast<unsigned>(
                         emulator.core().memory.read8(thread.address + 0x325U))
                  << " entry=0x" << emulator.core().memory.read32_be(
                         thread.address + 0x39CU)
                  << " wake=" << std::dec << thread.wake_tick
                  << " sleeping=" << (thread.sleeping ? "yes" : "no") << '\n';
    }
    std::cout << "HLE calls observed:\n";
    for (const auto& statistic : result.hle_call_statistics)
    {
        std::cout << "  " << statistic.name << ": " << statistic.calls << '\n';
    }
    const auto recent_hle_calls = emulator.core().hle.recent_call_trace_snapshot();
    constexpr std::size_t recent_hle_display_count = 512U;
    const std::size_t first_recent_hle = recent_hle_calls.size() > recent_hle_display_count
        ? recent_hle_calls.size() - recent_hle_display_count
        : 0U;
    std::cout << "Recent HLE calls (oldest first):\n";
    for (std::size_t index = first_recent_hle; index < recent_hle_calls.size(); ++index)
    {
        const auto& call = recent_hle_calls[index];
        std::cout << "  0x" << std::hex << std::uppercase << call.import_address
                  << " " << call.name << " return=0x" << call.return_address
                  << " args=";
        for (std::size_t argument = 0U; argument < call.arguments.size(); ++argument)
        {
            std::cout << (argument == 0U ? "(" : ",") << "r"
                      << (argument + 3U) << "=0x" << call.arguments[argument];
        }
        std::cout << ") ->";
        for (std::size_t result = 0U; result < call.results.size(); ++result)
        {
            std::cout << " r" << (result + 3U) << "=0x" << call.results[result];
        }
        std::cout << (call.completed ? " [returned]" :
                      call.implemented ? " [interrupted]" : " [unimplemented]")
                  << std::dec << '\n';
    }
    return 0;
}

struct WorkerResult
{
    std::mutex mutex;
    std::string message;
    std::optional<affogato::cpu::espresso::Gx2ClearColorRequest> latest_guest_clear;
    std::uint64_t guest_clear_sequence{};
    std::vector<affogato::cpu::espresso::Gx2Command> pending_gx2_commands;
    std::atomic_bool running{};
};

void launch_game(
    const fs::path& path,
    WorkerResult& worker,
    std::jthread& thread,
    affogato::ExecutionMode execution_mode)
{
    if (worker.running.exchange(true))
    {
        return;
    }
    if (thread.joinable()) thread.join();
    {
        std::lock_guard lock(worker.mutex);
        worker.message = "Reading RPX " + path_utf8(path) + " ...";
    }
    thread = std::jthread([path, &worker, execution_mode]
    {
        std::string message;
        try
        {
            const auto file_read_start = std::chrono::steady_clock::now();
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input)
            {
                throw std::runtime_error("Could not open RPX file.");
            }
            const auto length = input.tellg();
            if (length < 0)
            {
                throw std::runtime_error("Could not determine RPX file size.");
            }
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
            input.seekg(0);
            if (!bytes.empty() && !input.read(
                    reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())))
            {
                throw std::runtime_error("Could not read the complete RPX file.");
            }
            const auto file_read_end = std::chrono::steady_clock::now();
            const auto file_read_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                file_read_end - file_read_start);

            const auto rpx_load_start = std::chrono::steady_clock::now();
            affogato::Emulator emulator;
            static_cast<void>(emulator.load_rpx(
                bytes, path.parent_path().parent_path() / "content"));
            const auto rpx_load_end = std::chrono::steady_clock::now();
            const auto rpx_load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                rpx_load_end - rpx_load_start);

            constexpr std::size_t rpx_instruction_limit = 5'000'000'000ULL;
            constexpr std::size_t progress_chunk_size = 1'000'000U;
            std::size_t total_steps = 0;
            std::uint64_t observed_clear_sequence = 0U;
            auto guest_run_start = std::chrono::steady_clock::now();
            affogato::EmulatorRunResult run_result{};
            {
                std::lock_guard lock(worker.mutex);
                worker.message = "RPX loaded; starting guest execution.\nFile read: " +
                    std::to_string(file_read_time.count()) + " ms; RPX load: " +
                    std::to_string(rpx_load_time.count()) + " ms.";
            }
            while (total_steps < rpx_instruction_limit)
            {
                const std::size_t chunk_size = std::min(
                    progress_chunk_size, rpx_instruction_limit - total_steps);
                run_result = emulator.run(chunk_size, execution_mode);
                total_steps += run_result.execution.steps;
                run_result.execution.steps = total_steps;
                const auto& gx2 = emulator.core().gx2;
                if (gx2.clear_color_sequence != observed_clear_sequence &&
                    gx2.latest_clear_color.has_value())
                {
                    observed_clear_sequence = gx2.clear_color_sequence;
                    std::lock_guard lock(worker.mutex);
                    worker.latest_guest_clear = gx2.latest_clear_color;
                    ++worker.guest_clear_sequence;
                }
                if (!run_result.gx2_commands.empty())
                {
                    std::lock_guard lock(worker.mutex);
                    worker.pending_gx2_commands.insert(
                        worker.pending_gx2_commands.end(),
                        run_result.gx2_commands.begin(),
                        run_result.gx2_commands.end());
                }
                if (run_result.execution.reason !=
                    affogato::cpu::espresso::StopReason::instruction_limit)
                {
                    break;
                }

                const auto now = std::chrono::steady_clock::now();
                const auto guest_run_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - guest_run_start);
                message = format_progress(
                    run_result, total_steps, file_read_time, rpx_load_time, guest_run_time);
                std::lock_guard lock(worker.mutex);
                worker.message = message;
            }
            const auto guest_run_end = std::chrono::steady_clock::now();
            const auto guest_run_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                guest_run_end - guest_run_start);
            message = format_result(run_result) +
                "\n\nPhase timings: file read " + std::to_string(file_read_time.count()) +
                " ms; RPX load " + std::to_string(rpx_load_time.count()) +
                " ms; guest execution " + std::to_string(guest_run_time.count()) + " ms.";
        }
        catch (const std::exception& exception)
        {
            message = std::string("Launch failed: ") + exception.what();
        }
        catch (...)
        {
            message = "Launch failed with an unknown error.";
        }
        {
            std::lock_guard lock(worker.mutex);
            worker.message = std::move(message);
        }
        worker.running = false;
    });
}

struct FolderSelection
{
    std::mutex mutex;
    std::vector<std::string> pending_paths;
    std::string pending_error;
};

struct FolderDialogRequest
{
    std::shared_ptr<FolderSelection> selection;
};

void SDLCALL folder_selected(void* userdata, const char* const* filelist, int)
{
    std::unique_ptr<FolderDialogRequest> request(static_cast<FolderDialogRequest*>(userdata));
    if (!request || !request->selection) return;

    const auto selection = request->selection;
    try
    {
        if (filelist == nullptr)
        {
            const char* detail = SDL_GetError();
            std::lock_guard lock(selection->mutex);
            selection->pending_error = "Folder picker failed";
            if (detail != nullptr && detail[0] != '\0')
            {
                selection->pending_error += ": ";
                selection->pending_error += detail;
            }
            return;
        }

        // SDL gives us a NULL-terminated array of UTF-8 path pointers. Copy
        // each complete path while the callback's temporary strings exist.
        std::vector<std::string> copied_paths;
        for (const char* const* item = filelist; *item != nullptr; ++item)
        {
            copied_paths.emplace_back(*item);
        }

        std::lock_guard lock(selection->mutex);
        for (std::string& path : copied_paths)
        {
            selection->pending_paths.push_back(std::move(path));
        }
    }
    catch (const std::exception& exception)
    {
        try
        {
            std::lock_guard lock(selection->mutex);
            selection->pending_error = std::string("Could not copy folder dialog result: ") + exception.what();
        }
        catch (...)
        {
        }
    }
    catch (...)
    {
        try
        {
            std::lock_guard lock(selection->mutex);
            selection->pending_error = "Unknown error while copying folder dialog result.";
        }
        catch (...)
        {
        }
    }
}

void report_fatal_error(const char* message) noexcept
{
    try
    {
        std::ofstream log("AffogatoFrontend.log", std::ios::app);
        log << "Affogato frontend failure\n"
            << "Stage: " << current_stage << '\n'
            << "Error: " << message << "\n\n";
    }
    catch (...)
    {
    }

    std::cerr << "Affogato frontend failed during " << current_stage
              << ": " << message << '\n'
              << "Details were appended to AffogatoFrontend.log in the current directory.\n";
    if (sdl_initialized)
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Affogato error", message, nullptr);
    }
}

} // namespace

int run_frontend()
{
    current_stage = "SDL initialization";
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        throw std::runtime_error(std::string("SDL initialization failed: ") + SDL_GetError());
    }
    sdl_initialized = true;

    current_stage = "window creation";
    const float scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    SDL_Window* window = SDL_CreateWindow("Affogato", static_cast<int>(1050 * scale),
        static_cast<int>(720 * scale), SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (window == nullptr)
    {
        throw std::runtime_error(std::string("Window creation failed: ") + SDL_GetError());
    }
    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (renderer == nullptr)
    {
        SDL_DestroyWindow(window);
        throw std::runtime_error(std::string("Renderer creation failed: ") + SDL_GetError());
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(scale);
    ImGui::GetStyle().FontScaleDpi = scale;
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);

    current_stage = "Vulkan output initialization";
    std::optional<affogato::gpu::vulkan::VulkanRenderer> vulkan_output;
    try
    {
        vulkan_output.emplace();
        std::cout << "Vulkan output device: " << vulkan_output->device_name() << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan output unavailable; keeping the library frontend active: "
                  << error.what() << '\n';
    }

    current_stage = "loading saved game folders";
    char* preference_path = SDL_GetPrefPath("Affogato", "Affogato");
    const fs::path config_file = preference_path != nullptr
        ? fs::path(preference_path) / "game_folders.txt"
        : fs::temp_directory_path() / "affogato_game_folders.txt";
    SDL_free(preference_path);
    std::vector<fs::path> folders = load_folders(config_file);
    current_stage = "initial game library scan";
    bool library_dirty = false;
    std::vector<Game> games;
    std::string library_error;
    LibraryScan library_scan;
    std::jthread library_scan_thread;
    start_library_scan(folders, library_scan, library_scan_thread);
    const auto folder_selection = std::make_shared<FolderSelection>();
    std::optional<std::size_t> selected_game;
    WorkerResult worker;
    std::jthread launch_thread;
    bool grid_view = false;
    bool use_recompiler = false;
    bool done = false;
    std::uint64_t shown_guest_clear_sequence = 0U;
    while (!done)
    {
        if (vulkan_output)
        {
            vulkan_output->poll_events();
            if (vulkan_output->should_close()) done = true;
        }
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window)))
                done = true;
        }

        // SDL may invoke the folder dialog callback on a worker thread. Move
        // its UTF-8 results to the main thread before touching library state
        // or filesystem paths.
        std::vector<std::string> pending_paths;
        std::string dialog_error;
        {
            std::lock_guard lock(folder_selection->mutex);
            pending_paths.swap(folder_selection->pending_paths);
            dialog_error.swap(folder_selection->pending_error);
        }
        if (!dialog_error.empty())
        {
            std::lock_guard lock(worker.mutex);
            worker.message = std::move(dialog_error);
        }
        for (const std::string& selected_path : pending_paths)
        {
            current_stage = "converting a selected game folder path";
            try
            {
                const fs::path path = path_from_utf8(selected_path);
                std::error_code error;
                const fs::path canonical = fs::weakly_canonical(path, error);
                const fs::path& chosen = error ? path : canonical;
                if (std::find(folders.begin(), folders.end(), chosen) == folders.end())
                {
                    folders.push_back(chosen);
                    library_dirty = true;
                }
            }
            catch (const fs::filesystem_error& error)
            {
                std::lock_guard lock(worker.mutex);
                worker.message = std::string("Could not add game folder: ") + error.what();
            }
            catch (const std::exception& error)
            {
                std::lock_guard lock(worker.mutex);
                worker.message = std::string("Could not add game folder: ") + error.what();
            }
        }

        bool scan_running = false;
        {
            std::lock_guard lock(library_scan.mutex);
            if (library_scan.result_ready)
            {
                games = std::move(library_scan.games);
                library_error = std::move(library_scan.error);
                library_scan.result_ready = false;
                selected_game.reset();
            }
            scan_running = library_scan.running;
        }
        if (library_dirty && !scan_running)
        {
            current_stage = "starting a game library scan";
            library_error.clear();
            start_library_scan(folders, library_scan, library_scan_thread);
            library_dirty = false;
            save_folders(config_file, folders);
            scan_running = true;
        }

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Affogato Library", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse);

        if (ImGui::Button("Add Game Folder"))
        {
            auto* request = new FolderDialogRequest{folder_selection};
            SDL_ShowOpenFolderDialog(folder_selected, request, window, nullptr, true);
        }
        ImGui::SameLine();
        if (ImGui::Button("List View")) grid_view = false;
        ImGui::SameLine();
        if (ImGui::Button("Grid View")) grid_view = true;
        ImGui::SameLine();
        ImGui::Text("%zu titles", games.size());
        if (scan_running)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("Scanning folders...");
        }
        ImGui::Separator();

        ImGui::BeginChild("GameList", ImVec2(0, ImGui::GetContentRegionAvail().y * 0.60f), ImGuiChildFlags_Borders);
        if (!library_error.empty()) ImGui::TextWrapped("Library scan failed: %s", library_error.c_str());
        else if (games.empty() && !scan_running) ImGui::TextWrapped("No RPX titles found. Add a folder containing an extracted Wii U title or RPX.");
        else if (games.empty()) ImGui::TextWrapped("Scanning added folders for RPX titles...");
        else if (!grid_view)
        {
            for (std::size_t i = 0; i < games.size(); ++i)
            {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::Selectable(games[i].title.c_str(), selected_game == i, ImGuiSelectableFlags_AllowDoubleClick))
                {
                    selected_game = i;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        launch_game(games[i].rpx_path, worker, launch_thread,
                            use_recompiler ? affogato::ExecutionMode::recompiler
                                           : affogato::ExecutionMode::interpreter);
                }
                ImGui::SameLine(260);
                ImGui::TextDisabled("%s", path_utf8(games[i].rpx_path).c_str());
                ImGui::PopID();
            }
        }
        else
        {
            const float width = ImGui::GetContentRegionAvail().x;
            const int columns = std::max(1, static_cast<int>(width / 220.0f));
            if (ImGui::BeginTable("GameGrid", columns, ImGuiTableFlags_SizingStretchSame))
            {
                for (std::size_t i = 0; i < games.size(); ++i)
                {
                    ImGui::TableNextColumn();
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::BeginGroup();
                    if (ImGui::Selectable(games[i].title.c_str(), selected_game == i,
                        ImGuiSelectableFlags_AllowDoubleClick, ImVec2(-FLT_MIN, 70)))
                    {
                        selected_game = i;
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        launch_game(games[i].rpx_path, worker, launch_thread,
                            use_recompiler ? affogato::ExecutionMode::recompiler
                                           : affogato::ExecutionMode::interpreter);
                    }
                    ImGui::TextWrapped("%s", path_utf8(games[i].rpx_path).c_str());
                    ImGui::EndGroup();
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
        }
        ImGui::EndChild();

        if (selected_game && *selected_game < games.size())
        {
            ImGui::Text("Selected: %s", games[*selected_game].title.c_str());
            ImGui::SameLine();
            const bool launch_in_progress = worker.running.load();
            ImGui::BeginDisabled(launch_in_progress);
            if (ImGui::Button("Launch"))
                launch_game(games[*selected_game].rpx_path, worker, launch_thread,
                    use_recompiler ? affogato::ExecutionMode::recompiler
                                   : affogato::ExecutionMode::interpreter);
            ImGui::EndDisabled();
        }
        ImGui::BeginDisabled(worker.running.load());
        ImGui::Checkbox("Use experimental x86-64 recompiler", &use_recompiler);
        ImGui::EndDisabled();
        if (worker.running) ImGui::SameLine();
        if (worker.running) ImGui::Text("RPX loading / guest execution in progress...");
        ImGui::SeparatorText("Launch diagnostics");
        std::string message;
        {
            std::lock_guard lock(worker.mutex);
            message = worker.message;
        }
        ImGui::BeginChild("Diagnostics", ImVec2(0, 0), ImGuiChildFlags_Borders);
        std::string diagnostic_text = message.empty()
            ? "Select and launch an RPX to see its result."
            : std::move(message);
        ImGui::InputTextMultiline(
            "##launch-diagnostics",
            diagnostic_text.data(),
            diagnostic_text.size() + 1U,
            ImVec2(-FLT_MIN, -FLT_MIN),
            ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
        ImGui::EndChild();
        ImGui::End();

        current_stage = "rendering the frontend";
        ImGui::Render();
        const ImGuiIO& io = ImGui::GetIO();
        SDL_SetRenderScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        SDL_SetRenderDrawColor(renderer, 24, 27, 34, 255);
        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);

        std::vector<affogato::cpu::espresso::Gx2Command> guest_commands;
        {
            std::lock_guard lock(worker.mutex);
            guest_commands.swap(worker.pending_gx2_commands);
        }
        if (vulkan_output)
        {
            current_stage = "presenting Vulkan output";
            std::optional<affogato::cpu::espresso::Gx2ClearColorRequest> guest_clear;
            std::uint64_t guest_clear_sequence = 0U;
            {
                std::lock_guard lock(worker.mutex);
                guest_clear = worker.latest_guest_clear;
                guest_clear_sequence = worker.guest_clear_sequence;
            }
            if (!guest_commands.empty())
            {
                vulkan_output->consume_guest_commands(guest_commands);
            }
            if (guest_clear && guest_clear_sequence != shown_guest_clear_sequence)
            {
                vulkan_output->set_guest_clear_color(guest_clear->rgba);
                shown_guest_clear_sequence = guest_clear_sequence;
            }
            vulkan_output->render_frame();
        }
    }

    current_stage = "saving game folders and shutting down";
    save_folders(config_file, folders);
    if (library_scan_thread.joinable()) library_scan_thread.join();
    if (launch_thread.joinable()) launch_thread.join();
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

int run_vulkan_texture_upload_smoke_test()
{
    affogato::gpu::vulkan::VulkanRenderer renderer(false);
    affogato::gpu::Rgba8TextureStaging texture{
        2U, 2U,
        {
            0xFFU, 0x00U, 0x00U, 0xFFU,
            0x00U, 0xFFU, 0x00U, 0xFFU,
            0x00U, 0x00U, 0xFFU, 0xFFU,
            0xFFU, 0xFFU, 0xFFU, 0xFFU,
        }};
    const std::size_t texture_id = renderer.upload_rgba8_texture(texture);
    if (texture_id != 0U || renderer.uploaded_texture_count() != 1U ||
        renderer.uploaded_texture_view(texture_id) == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Vulkan texture upload did not produce a valid image view");
    }
    const affogato::gpu::Rgba8TextureStaging second_texture{
        1U, 1U, {0x20U, 0x40U, 0x80U, 0xFFU}};
    const std::size_t second_texture_id =
        renderer.upload_rgba8_texture(second_texture);
    if (second_texture_id != 1U || renderer.uploaded_texture_count() != 2U ||
        renderer.uploaded_texture_view(second_texture_id) == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Vulkan renderer did not retain multiple uploaded textures");
    }

    bool rejected_invalid_input = false;
    try
    {
        static_cast<void>(renderer.upload_rgba8_texture({}));
    }
    catch (const std::invalid_argument&)
    {
        rejected_invalid_input = true;
    }
    if (!rejected_invalid_input || renderer.uploaded_texture_count() != 2U)
    {
        throw std::runtime_error("Vulkan texture upload accepted invalid staging data");
    }
    std::cout << "Vulkan RGBA8 upload smoke test passed on " << renderer.device_name()
              << "; image view created and retained until renderer teardown.\n";
    return 0;
}

int main(int argc, char* argv[])
{
    const bool command_line_mode = argc > 1;
    try
    {
        if (command_line_mode)
        {
            if (std::string_view(argv[1]) == "--run-rpx")
            {
                return run_rpx_diagnostic(argc, argv);
            }
            if (std::string_view(argv[1]) == "--vulkan-texture-upload-smoke-test" &&
                argc == 2)
            {
                return run_vulkan_texture_upload_smoke_test();
            }
            std::cerr << "Unknown option: " << argv[1] << '\n'
                      << "Usage: AffogatoFrontend.exe [--run-rpx <file.rpx> "
                         "[--recompiler] [--limit <instructions>]]\n"
                         "       AffogatoFrontend.exe --vulkan-texture-upload-smoke-test\n";
            return 2;
        }
        return run_frontend();
    }
    catch (const std::exception& error)
    {
        if (command_line_mode)
        {
            std::cerr << "Command-line mode failed: " << error.what() << '\n';
            return 1;
        }
        report_fatal_error(error.what());
    }
    catch (...)
    {
        if (command_line_mode)
        {
            std::cerr << "Command-line mode failed with an unknown exception.\n";
            return 1;
        }
        report_fatal_error("Unknown non-standard exception.");
    }
    return 1;
}
