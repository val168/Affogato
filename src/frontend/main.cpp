#include "emulator.hpp"
#include "gpu/latte_shader_interpreter.hpp"
#include "gpu/vulkan/vulkan_renderer.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_filesystem.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
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
#include <span>
#include <stdexcept>
#include <sstream>
#include <stop_token>
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

std::optional<affogato::gpu::Rgba8TextureStaging> capture_tv_scanout(
    const affogato::Emulator& emulator,
    std::span<const affogato::cpu::espresso::Gx2Command> commands)
{
    using affogato::cpu::espresso::Gx2CommandType;
    const auto& gx2 = emulator.core().gx2;
    if (!gx2.tv_scan_buffer_configured || gx2.tv_output_enabled == false)
    {
        return std::nullopt;
    }

    std::uint32_t flip_request_count = 0U;
    bool has_scan_buffer_swap = false;
    for (const auto& command : commands)
    {
        if (command.type == Gx2CommandType::scan_buffer_swap)
        {
            flip_request_count = command.value;
            has_scan_buffer_swap = true;
        }
    }
    if (!has_scan_buffer_swap)
    {
        return std::nullopt;
    }

    std::uint32_t buffer_count = 0U;
    switch (gx2.tv_buffering_mode)
    {
    case 1U:
        buffer_count = 1U;
        break;
    case 2U:
        buffer_count = 2U;
        break;
    case 3U:
        buffer_count = 3U;
        break;
    default: return std::nullopt;
    }

    constexpr std::uint64_t bytes_per_pixel = 4U;
    if (gx2.tv_scan_width == 0U || gx2.tv_scan_height == 0U ||
        gx2.tv_scan_buffer_address == 0U)
    {
        return std::nullopt;
    }
    const std::uint64_t row_size =
        static_cast<std::uint64_t>(gx2.tv_scan_width) * bytes_per_pixel;
    if (gx2.tv_scan_height > std::numeric_limits<std::uint64_t>::max() / row_size)
    {
        return std::nullopt;
    }
    const std::uint64_t frame_size = row_size * gx2.tv_scan_height;
    const std::uint32_t buffer_index = flip_request_count == 0U
        ? 0U
        : (flip_request_count - 1U) % buffer_count;
    const std::uint64_t frame_offset = frame_size * buffer_index;
    if (frame_size > std::numeric_limits<std::uint32_t>::max() ||
        frame_offset + frame_size > gx2.tv_scan_buffer_size ||
        static_cast<std::uint64_t>(gx2.tv_scan_buffer_address) + frame_offset >
            std::numeric_limits<std::uint32_t>::max())
    {
        return std::nullopt;
    }

    std::vector<std::uint8_t> guest_pixels(static_cast<std::size_t>(frame_size));
    try
    {
        emulator.core().memory.read_bytes(
            static_cast<std::uint32_t>(gx2.tv_scan_buffer_address + frame_offset),
            guest_pixels);
    }
    catch (const affogato::cpu::espresso::GuestMemoryFault&)
    {
        return std::nullopt;
    }
    return affogato::gpu::stage_rgba8_scanout(
        gx2.tv_scan_width,
        gx2.tv_scan_height,
        gx2.tv_scan_width,
        guest_pixels);
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
                    if (rank(a) != rank(b))
                    {
                        return rank(a) < rank(b);
                    }
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
    std::ranges::sort(fallbacks, [](const auto& lhs, const auto& rhs)
    {
        return lhs.second > rhs.second;
    });
    if (fallbacks.empty())
    {
        return;
    }

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
    case StopReason::gx2_draw_captured:
        text << "\nPaused: GX2 draw captured";
        break;
    case StopReason::unsupported_instruction:
        text << "\nStopped: unsupported PowerPC instruction 0x" << std::hex
             << std::uppercase << result.execution.instruction_word;
        break;
    case StopReason::unsupported_architectural_state:
        text << "\nStopped: unsupported guest CPU mode";
        if (!result.execution.detail.empty())
        {
            text << "\nDetail: " << result.execution.detail;
        }
        break;
    case StopReason::unimplemented_hle_call:
        text << "\nStopped: unimplemented HLE call " << result.execution.hle_call;
        break;
    case StopReason::hle_error:
        text << "\nStopped: HLE error";
        if (!result.execution.detail.empty())
        {
            text << "\nDetail: " << result.execution.detail;
        }
        break;
    case StopReason::memory_fault:
        text << "\nStopped: memory fault";
        if (!result.execution.detail.empty())
        {
            text << "\nDetail: " << result.execution.detail;
        }
        break;
    }
    const std::string history =
        affogato::cpu::espresso::format_instruction_history(result.execution);
    if (!history.empty())
    {
        text << '\n' << history;
    }
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
    if (!history.empty())
    {
        text << "\nRecent instruction trace:\n" << history;
    }
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
    std::condition_variable_any draw_boundary_acknowledged;
    std::string message;
    std::string graphics_message;
    std::optional<affogato::cpu::espresso::Gx2ClearColorRequest> latest_guest_clear;
    std::uint64_t guest_clear_sequence{};
    std::optional<affogato::gpu::Rgba8TextureStaging> latest_guest_scanout;
    std::uint64_t guest_scanout_sequence{};
    std::vector<affogato::cpu::espresso::Gx2Command> pending_gx2_commands;
    std::vector<affogato::cpu::espresso::Gx2ColorBufferSnapshot>
        pending_guest_color_buffer_updates;
    std::uint64_t draw_boundary_sequence{};
    std::uint64_t processed_draw_boundary_sequence{};
    std::atomic_bool running{};
};

void launch_game(
    const fs::path& path,
    WorkerResult& worker,
    std::jthread& thread,
    affogato::ExecutionMode execution_mode,
    bool capture_guest_draw_vertex_data,
    bool synchronize_draw_capture,
    bool use_latte_vertex_shader_subset,
    bool use_latte_matched_shader_subset)
{
    if (worker.running.exchange(true))
    {
        return;
    }
    capture_guest_draw_vertex_data =
        capture_guest_draw_vertex_data || use_latte_vertex_shader_subset ||
        use_latte_matched_shader_subset;
    if (thread.joinable())
    {
        thread.request_stop();
        worker.draw_boundary_acknowledged.notify_all();
        thread.join();
    }
    {
        std::lock_guard lock(worker.mutex);
        worker.message = "Reading RPX " + path_utf8(path) + " ...";
        worker.graphics_message.clear();
        worker.latest_guest_scanout.reset();
        worker.draw_boundary_sequence = 0U;
        worker.processed_draw_boundary_sequence = 0U;
        worker.pending_guest_color_buffer_updates.clear();
        ++worker.guest_scanout_sequence;
    }
    thread = std::jthread([
        path,
        &worker,
        execution_mode,
        capture_guest_draw_vertex_data,
        synchronize_draw_capture](std::stop_token stop_token)
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
            emulator.core().gx2.capture_draw_vertex_data =
                capture_guest_draw_vertex_data;
            emulator.core().gx2.capture_draw_color_buffer_data =
                synchronize_draw_capture;
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
            while (total_steps < rpx_instruction_limit &&
                   !stop_token.stop_requested())
            {
                const std::size_t chunk_size = std::min(
                    progress_chunk_size, rpx_instruction_limit - total_steps);
                run_result = emulator.run(
                    chunk_size, execution_mode, synchronize_draw_capture);
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
                std::uint64_t draw_boundary_sequence = 0U;
                if (!run_result.gx2_commands.empty())
                {
                    if (auto scanout = capture_tv_scanout(
                            emulator, run_result.gx2_commands))
                    {
                        std::lock_guard lock(worker.mutex);
                        worker.latest_guest_scanout = std::move(*scanout);
                        ++worker.guest_scanout_sequence;
                    }
                    std::lock_guard lock(worker.mutex);
                    worker.pending_gx2_commands.insert(
                        worker.pending_gx2_commands.end(),
                        run_result.gx2_commands.begin(),
                        run_result.gx2_commands.end());
                    if (run_result.execution.reason ==
                            affogato::cpu::espresso::StopReason::gx2_draw_captured &&
                        synchronize_draw_capture)
                    {
                        draw_boundary_sequence = ++worker.draw_boundary_sequence;
                    }
                }
                if (draw_boundary_sequence != 0U)
                {
                    worker.draw_boundary_acknowledged.notify_all();
                    std::vector<
                        affogato::cpu::espresso::Gx2ColorBufferSnapshot> updates;
                    bool acknowledged = false;
                    {
                        std::unique_lock lock(worker.mutex);
                        acknowledged = worker.draw_boundary_acknowledged.wait(
                            lock,
                            stop_token,
                            [&worker, draw_boundary_sequence]
                            {
                                return worker.processed_draw_boundary_sequence >=
                                    draw_boundary_sequence;
                            });
                        if (acknowledged)
                        {
                            updates.swap(
                                worker.pending_guest_color_buffer_updates);
                        }
                    }
                    if (!acknowledged)
                    {
                        break;
                    }

                    for (const auto& update : updates)
                    {
                        if (update.status != affogato::cpu::espresso::
                                Gx2ColorBufferSnapshotStatus::captured)
                        {
                            throw std::runtime_error(
                                "Frontend received a failed GX2 color-buffer readback");
                        }
                        emulator.core().memory.write_bytes(
                            update.descriptor.image_address,
                            update.bytes);
                    }
                    continue;
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
    if (!request || !request->selection)
    {
        return;
    }

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
    bool use_diagnostic_guest_draws = false;
    bool use_latte_vertex_shader_subset = false;
    bool use_latte_matched_shader_subset = false;
    bool done = false;
    std::uint64_t shown_guest_clear_sequence = 0U;
    std::uint64_t shown_guest_scanout_sequence = 0U;
    while (!done)
    {
        if (vulkan_output)
        {
            vulkan_output->poll_events();
            if (vulkan_output->should_close())
            {
                done = true;
            }
        }
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                 event.window.windowID == SDL_GetWindowID(window)))
            {
                done = true;
            }
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
        if (ImGui::Button("List View"))
        {
            grid_view = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Grid View"))
        {
            grid_view = true;
        }
        ImGui::SameLine();
        ImGui::Text("%zu titles", games.size());
        if (scan_running)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("Scanning folders...");
        }
        ImGui::Separator();

        ImGui::BeginChild(
            "GameList",
            ImVec2(0, ImGui::GetContentRegionAvail().y * 0.60f),
            ImGuiChildFlags_Borders);
        if (!library_error.empty())
        {
            ImGui::TextWrapped("Library scan failed: %s", library_error.c_str());
        }
        else if (games.empty() && !scan_running)
        {
            ImGui::TextWrapped(
                "No RPX titles found. Add a folder containing an extracted Wii U title or RPX.");
        }
        else if (games.empty())
        {
            ImGui::TextWrapped("Scanning added folders for RPX titles...");
        }
        else if (!grid_view)
        {
            for (std::size_t i = 0; i < games.size(); ++i)
            {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::Selectable(
                        games[i].title.c_str(),
                        selected_game == i,
                        ImGuiSelectableFlags_AllowDoubleClick))
                {
                    selected_game = i;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    {
                        launch_game(games[i].rpx_path, worker, launch_thread,
                            use_recompiler ? affogato::ExecutionMode::recompiler
                                           : affogato::ExecutionMode::interpreter,
                            use_diagnostic_guest_draws ||
                                use_latte_vertex_shader_subset ||
                                use_latte_matched_shader_subset,
                            (use_diagnostic_guest_draws ||
                             use_latte_vertex_shader_subset ||
                             use_latte_matched_shader_subset) &&
                                vulkan_output.has_value(),
                            use_latte_vertex_shader_subset,
                            use_latte_matched_shader_subset);
                    }
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
                        {
                            launch_game(games[i].rpx_path, worker, launch_thread,
                                use_recompiler ? affogato::ExecutionMode::recompiler
                                               : affogato::ExecutionMode::interpreter,
                                use_diagnostic_guest_draws ||
                                    use_latte_vertex_shader_subset ||
                                    use_latte_matched_shader_subset,
                                (use_diagnostic_guest_draws ||
                                 use_latte_vertex_shader_subset ||
                                 use_latte_matched_shader_subset) &&
                                    vulkan_output.has_value(),
                                use_latte_vertex_shader_subset,
                                use_latte_matched_shader_subset);
                        }
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
            {
                launch_game(games[*selected_game].rpx_path, worker, launch_thread,
                    use_recompiler ? affogato::ExecutionMode::recompiler
                                   : affogato::ExecutionMode::interpreter,
                    use_diagnostic_guest_draws ||
                        use_latte_vertex_shader_subset ||
                        use_latte_matched_shader_subset,
                    (use_diagnostic_guest_draws ||
                     use_latte_vertex_shader_subset ||
                     use_latte_matched_shader_subset) &&
                        vulkan_output.has_value(),
                    use_latte_vertex_shader_subset,
                    use_latte_matched_shader_subset);
            }
            ImGui::EndDisabled();
        }
        ImGui::BeginDisabled(worker.running.load());
        ImGui::Checkbox("Use experimental x86-64 recompiler", &use_recompiler);
        if (ImGui::Checkbox(
                "Diagnostic GX2 draws (host shader, host window)",
                &use_diagnostic_guest_draws) && vulkan_output)
        {
            if (use_diagnostic_guest_draws)
            {
                use_latte_vertex_shader_subset = false;
                use_latte_matched_shader_subset = false;
            }
            vulkan_output->clear_guest_diagnostic_draws();
        }
        if (ImGui::Checkbox(
                "Experimental Latte vertex subset (diagnostic fragment)",
                &use_latte_vertex_shader_subset) && vulkan_output)
        {
            if (use_latte_vertex_shader_subset)
            {
                use_diagnostic_guest_draws = false;
                use_latte_matched_shader_subset = false;
            }
            vulkan_output->clear_guest_diagnostic_draws();
        }
        if (ImGui::Checkbox(
                "Experimental Latte vertex/fragment match (diagnostic target)",
                &use_latte_matched_shader_subset) && vulkan_output)
        {
            if (use_latte_matched_shader_subset)
            {
                use_diagnostic_guest_draws = false;
                use_latte_vertex_shader_subset = false;
            }
            vulkan_output->clear_guest_diagnostic_draws();
        }
        ImGui::EndDisabled();
        if (worker.running)
        {
            ImGui::SameLine();
        }
        if (worker.running)
        {
            ImGui::Text("RPX loading / guest execution in progress...");
        }
        ImGui::SeparatorText("Launch diagnostics");
        std::string message;
        std::string graphics_message;
        {
            std::lock_guard lock(worker.mutex);
            message = worker.message;
            graphics_message = worker.graphics_message;
        }
        ImGui::BeginChild("Diagnostics", ImVec2(0, 0), ImGuiChildFlags_Borders);
        std::string diagnostic_text = message.empty()
            ? "Select and launch an RPX to see its result."
            : std::move(message);
        if (!graphics_message.empty())
        {
            diagnostic_text += "\n\nGraphics diagnostics: " + graphics_message;
        }
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
        std::uint64_t draw_boundary_sequence = 0U;
        {
            std::lock_guard lock(worker.mutex);
            guest_commands.swap(worker.pending_gx2_commands);
            draw_boundary_sequence = worker.draw_boundary_sequence;
        }
        if (vulkan_output)
        {
            current_stage = "presenting Vulkan output";
            std::optional<affogato::cpu::espresso::Gx2ClearColorRequest> guest_clear;
            std::uint64_t guest_clear_sequence = 0U;
            std::optional<affogato::gpu::Rgba8TextureStaging> guest_scanout;
            std::uint64_t guest_scanout_sequence = 0U;
            {
                std::lock_guard lock(worker.mutex);
                guest_clear = worker.latest_guest_clear;
                guest_clear_sequence = worker.guest_clear_sequence;
                guest_scanout_sequence = worker.guest_scanout_sequence;
                if (guest_scanout_sequence != shown_guest_scanout_sequence &&
                    worker.latest_guest_scanout.has_value())
                {
                    guest_scanout = std::move(worker.latest_guest_scanout);
                }
            }
            auto draw_policy =
                affogato::gpu::vulkan::GuestDrawPolicy::capture_only;
            if (use_diagnostic_guest_draws)
            {
                draw_policy = affogato::gpu::vulkan::GuestDrawPolicy::
                    diagnostic_float2_position;
            }
            if (use_latte_vertex_shader_subset)
            {
                draw_policy = affogato::gpu::vulkan::GuestDrawPolicy::
                    diagnostic_latte_vertex_spirv_position_base_zero;
            }
            if (use_latte_matched_shader_subset)
            {
                draw_policy = affogato::gpu::vulkan::GuestDrawPolicy::
                    diagnostic_latte_vertex_fragment_matched_move_base_zero;
            }
            try
            {
                vulkan_output->consume_guest_commands(guest_commands, draw_policy);
                auto color_buffer_updates =
                    vulkan_output->take_guest_color_buffer_updates();
                if (!color_buffer_updates.empty())
                {
                    std::lock_guard lock(worker.mutex);
                    worker.pending_guest_color_buffer_updates.insert(
                        worker.pending_guest_color_buffer_updates.end(),
                        std::make_move_iterator(color_buffer_updates.begin()),
                        std::make_move_iterator(color_buffer_updates.end()));
                }
            }
            catch (const std::exception& error)
            {
                std::lock_guard lock(worker.mutex);
                if (worker.graphics_message.empty())
                {
                    worker.graphics_message = error.what();
                }
            }
            if (guest_clear && guest_clear_sequence != shown_guest_clear_sequence)
            {
                vulkan_output->set_guest_clear_color(guest_clear->rgba);
                shown_guest_clear_sequence = guest_clear_sequence;
            }
            if (guest_scanout)
            {
                if (!vulkan_output->set_guest_scanout(std::move(*guest_scanout)))
                {
                    std::lock_guard lock(worker.mutex);
                    worker.message =
                        "Vulkan surface does not support guest scanout presentation.";
                }
            }
            shown_guest_scanout_sequence = guest_scanout_sequence;
            vulkan_output->render_frame();
            if (draw_boundary_sequence > 0U)
            {
                bool acknowledged_new_boundary = false;
                {
                    std::lock_guard lock(worker.mutex);
                    if (worker.processed_draw_boundary_sequence <
                        draw_boundary_sequence)
                    {
                        worker.processed_draw_boundary_sequence =
                            draw_boundary_sequence;
                        acknowledged_new_boundary = true;
                    }
                }
                if (acknowledged_new_boundary)
                {
                    worker.draw_boundary_acknowledged.notify_all();
                }
            }
        }
    }

    current_stage = "saving game folders and shutting down";
    save_folders(config_file, folders);
    if (library_scan_thread.joinable())
    {
        library_scan_thread.join();
    }
    if (launch_thread.joinable())
    {
        launch_thread.request_stop();
        worker.draw_boundary_acknowledged.notify_all();
        launch_thread.join();
    }
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

int run_gx2_guest_scanout_smoke_test(bool visible_window = false)
{
    constexpr std::uint32_t color_buffer_address = 0x4000U;
    constexpr std::uint32_t source_image_address = 0x100000U;
    constexpr std::uint32_t scan_buffer_address = 0x600000U;
    constexpr std::uint32_t width = 1280U;
    constexpr std::uint32_t height = 720U;
    constexpr std::uint32_t pitch = width;
    constexpr std::uint32_t bytes_per_pixel = 4U;
    constexpr std::uint32_t frame_size = width * height * bytes_per_pixel;
    constexpr std::uint32_t buffer_count = 3U;
    constexpr std::uint32_t scan_buffer_size = frame_size * buffer_count;
    constexpr std::uint32_t guest_call_stub_address = 0x2000U;
    constexpr std::uint32_t guest_bctrl_instruction = 0x4E800421U;

    affogato::Emulator emulator;
    auto& core = emulator.core();
    core.configure_guest_heap(
        0x02000000U, static_cast<std::uint32_t>(core.memory.size()));
    constexpr std::array<std::uint32_t, 39U> descriptor_template{
        1U, width, height, 1U, 1U, 0x1AU, 0U, 0x0AU, frame_size,
        source_image_address, 0U, 0U, 1U, 0U, 256U, pitch};
    std::array<std::uint32_t, 39U> descriptor{};
    std::copy(
        descriptor_template.begin(), descriptor_template.end(),
        descriptor.begin());
    descriptor[31U] = 1U;
    for (std::size_t index = 0U; index < descriptor.size(); ++index)
    {
        core.memory.write32_be(
            color_buffer_address + static_cast<std::uint32_t>(index * 4U),
            descriptor[index]);
    }
    core.memory.zero_fill(source_image_address, frame_size);
    core.memory.zero_fill(scan_buffer_address, scan_buffer_size);
    core.memory.write32_be(guest_call_stub_address, guest_bctrl_instruction);

    const auto invoke = [&](
        const char* symbol,
        std::span<const std::uint32_t> arguments,
        const std::array<double, 4U>& floating_arguments = {}) {
        const std::uint32_t import = core.hle.bind_import("gx2", symbol);
        for (std::size_t index = 0U; index < arguments.size(); ++index)
        {
            core.state.gpr[index + 3U] = arguments[index];
        }
        for (std::size_t index = 0U; index < floating_arguments.size(); ++index)
        {
            core.state.fpr[index + 1U] =
                std::bit_cast<std::uint64_t>(floating_arguments[index]);
        }
        core.state.cia = guest_call_stub_address;
        core.state.ctr = import;
        const auto branch_result = core.run(1U);
        if (branch_result.reason !=
                affogato::cpu::espresso::StopReason::instruction_limit ||
            core.state.cia != import)
        {
            throw std::runtime_error(
                std::string("Guest bctrl did not reach GX2 HLE: ") + symbol);
        }

        const auto hle_result = core.run(1U);
        if (hle_result.reason !=
                affogato::cpu::espresso::StopReason::instruction_limit ||
            core.state.cia != guest_call_stub_address + 4U)
        {
            throw std::runtime_error(
                std::string("GX2 integration HLE failed: ") + symbol);
        }
    };

    invoke("GX2Init", std::array<std::uint32_t, 1U>{0U});
    invoke("GX2SetTVBuffer", std::array<std::uint32_t, 5U>{
        scan_buffer_address, scan_buffer_size, 3U, 0x1AU, buffer_count});
    invoke("GX2SetTVEnable", std::array<std::uint32_t, 1U>{1U});

    const auto execute_frame = [&](const std::array<double, 4U>& rgba) {
        invoke(
            "GX2ClearColor",
            std::array<std::uint32_t, 1U>{color_buffer_address},
            rgba);
        invoke(
            "GX2CopyColorBufferToScanBuffer",
            std::array<std::uint32_t, 2U>{color_buffer_address, 1U});
        invoke("GX2SwapScanBuffers", {});

        const auto commands = affogato::cpu::espresso::take_gx2_commands(core.gx2);
        const auto scanout = capture_tv_scanout(emulator, commands);
        if (!scanout || scanout->width != width || scanout->height != height)
        {
            throw std::runtime_error(
                "GX2 integration did not produce a configured TV scanout");
        }
        return *scanout;
    };

    auto first_frame = execute_frame({0.9, 0.1, 0.2, 1.0});
    constexpr std::array<std::uint8_t, 4U> first_expected{229U, 26U, 51U, 255U};
    if (!std::equal(
            first_expected.begin(), first_expected.end(),
            first_frame.pixels.begin()))
    {
        std::ostringstream message;
        message << "GX2 first-frame pixel was "
                << static_cast<unsigned int>(first_frame.pixels[0U]) << ','
                << static_cast<unsigned int>(first_frame.pixels[1U]) << ','
                << static_cast<unsigned int>(first_frame.pixels[2U]) << ','
                << static_cast<unsigned int>(first_frame.pixels[3U]);
        throw std::runtime_error(message.str());
    }

    auto second_frame = execute_frame({0.1, 0.8, 0.3, 1.0});
    constexpr std::array<std::uint8_t, 4U> second_expected{26U, 204U, 77U, 255U};
    if (!std::equal(
            second_expected.begin(), second_expected.end(),
            second_frame.pixels.begin()) ||
        first_frame.pixels == second_frame.pixels)
    {
        throw std::runtime_error(
            "GX2 guest clear changes did not change scanout pixels");
    }

    affogato::gpu::vulkan::VulkanRenderer renderer(visible_window);
    if (!renderer.set_guest_scanout(std::move(first_frame)))
    {
        throw std::runtime_error(
            "Vulkan renderer rejected the first GX2 integration scanout");
    }
    renderer.render_frame();
    if (!renderer.set_guest_scanout(std::move(second_frame)))
    {
        throw std::runtime_error(
            "Vulkan renderer rejected the changed GX2 integration scanout");
    }
    renderer.render_frame();
    while (visible_window && !renderer.should_close())
    {
        renderer.poll_events();
        renderer.render_frame();
    }
    std::cout << "Guest GX2 HLE clear/copy/swap produced changing TV scanout "
              << "pixels and presented them through Vulkan on "
              << renderer.device_name() << ". This diagnostic does not execute "
              << "a WUT RPX or guest draw call.\n";
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
    if (!renderer.set_guest_scanout(texture))
    {
        throw std::runtime_error(
            "Vulkan surface does not support guest scanout presentation");
    }
    renderer.render_frame();
    std::cout << "Vulkan RGBA8 upload and scanout presentation smoke test passed on "
              << renderer.device_name()
              << "; uploaded images remained valid through a presented scanout frame.\n";
    return 0;
}

enum class GuestTriangleShader
{
    cpu_mov,
    spirv_add_zero,
    spirv_add_one_x,
    spirv_add_negative_one_x,
    spirv_multiply_one,
    spirv_multiply_zero,
    spirv_maximum_self,
    spirv_minimum_self,
    spirv_maximum_zero,
    spirv_minimum_zero,
    spirv_conditional_move,
    spirv_integer_conditional_move,
    spirv_mixed_conditional_move,
    spirv_integer_compare_clip_z,
    spirv_integer_alu_identity,
    spirv_constant_pixel_color,
    spirv_default_pixel_input,
    spirv_matched_varying_move,
    spirv_matched_varying_add_half,
    spirv_matched_varying_multiply_half,
    spirv_matched_varying_maximum_half,
    spirv_matched_varying_minimum_half,
    spirv_matched_varying_add_negative_half,
    spirv_matched_varying_xor_zero,
    spirv_matched_varying_maximum_signed_integer_half,
    spirv_matched_varying_minimum_unsigned_integer_half,
    spirv_matched_varying_default_alpha
};

[[nodiscard]] bool contains_expected_interpolated_pixels(
    std::span<const std::uint8_t> rgba_pixels,
    GuestTriangleShader shader)
{
    constexpr std::size_t bytes_per_pixel = 4U;
    constexpr std::uint8_t maximum_zero_channel = 40U;
    constexpr std::uint8_t minimum_varying_green = 50U;
    constexpr std::uint8_t minimum_add_half_channel = 100U;
    constexpr std::uint8_t minimum_multiply_half_alpha = 110U;
    constexpr std::uint8_t maximum_multiply_half_alpha = 145U;
    constexpr std::uint8_t minimum_opaque_alpha = 200U;
    std::uint8_t minimum_green = 255U;
    std::uint8_t maximum_green = 0U;
    std::size_t matching_pixel_count = 0U;

    const bool is_add_half =
        shader == GuestTriangleShader::spirv_matched_varying_add_half;
    const bool is_maximum_half =
        shader == GuestTriangleShader::spirv_matched_varying_maximum_half ||
        shader == GuestTriangleShader::
            spirv_matched_varying_maximum_signed_integer_half;
    const bool is_add_negative_half =
        shader == GuestTriangleShader::spirv_matched_varying_add_negative_half;
    const bool is_default_alpha =
        shader == GuestTriangleShader::spirv_matched_varying_default_alpha;
    const bool is_multiply_or_minimum_half =
        shader == GuestTriangleShader::spirv_matched_varying_multiply_half ||
        shader == GuestTriangleShader::spirv_matched_varying_minimum_half ||
        shader == GuestTriangleShader::
            spirv_matched_varying_minimum_unsigned_integer_half ||
        shader == GuestTriangleShader::spirv_matched_varying_add_negative_half;
    const bool has_half_added_color =
        is_add_half || is_maximum_half || is_add_negative_half;

    for (std::size_t pixel_offset = 0U;
         pixel_offset + bytes_per_pixel <= rgba_pixels.size();
         pixel_offset += bytes_per_pixel)
    {
        const std::uint8_t red = rgba_pixels[pixel_offset];
        const std::uint8_t green = rgba_pixels[pixel_offset + 1U];
        const std::uint8_t blue = rgba_pixels[pixel_offset + 2U];
        const std::uint8_t alpha = rgba_pixels[pixel_offset + 3U];
        const bool has_expected_red_and_blue = has_half_added_color
            ? red >= minimum_add_half_channel &&
                blue >= minimum_add_half_channel
            : red <= maximum_zero_channel &&
                blue <= maximum_zero_channel;
        const bool has_expected_green = green >= minimum_varying_green;
        bool has_expected_alpha = false;
        if (is_default_alpha)
        {
            has_expected_alpha = alpha <= maximum_zero_channel;
        }
        else if (is_multiply_or_minimum_half || is_add_negative_half)
        {
            has_expected_alpha =
                alpha >= minimum_multiply_half_alpha &&
                alpha <= maximum_multiply_half_alpha;
        }
        else
        {
            has_expected_alpha = alpha >= minimum_opaque_alpha;
        }
        if (!has_expected_red_and_blue || !has_expected_green ||
            !has_expected_alpha)
        {
            continue;
        }

        minimum_green = std::min(minimum_green, green);
        maximum_green = std::max(maximum_green, green);
        ++matching_pixel_count;
    }

    return matching_pixel_count > 50U &&
        static_cast<unsigned int>(maximum_green - minimum_green) >= 20U;
}

int run_gx2_guest_triangle_test(
    bool visible_window = false,
    GuestTriangleShader shader = GuestTriangleShader::cpu_mov)
{
    const bool use_latte_spirv = shader != GuestTriangleShader::cpu_mov;
    const bool use_latte_pixel_spirv =
        shader == GuestTriangleShader::spirv_constant_pixel_color ||
        shader == GuestTriangleShader::spirv_default_pixel_input ||
        shader == GuestTriangleShader::spirv_matched_varying_move ||
        shader == GuestTriangleShader::spirv_matched_varying_add_half ||
        shader == GuestTriangleShader::spirv_matched_varying_multiply_half ||
        shader == GuestTriangleShader::spirv_matched_varying_maximum_half ||
        shader == GuestTriangleShader::spirv_matched_varying_minimum_half ||
        shader == GuestTriangleShader::spirv_matched_varying_add_negative_half ||
        shader == GuestTriangleShader::spirv_matched_varying_xor_zero ||
        shader == GuestTriangleShader::
            spirv_matched_varying_maximum_signed_integer_half ||
        shader == GuestTriangleShader::
            spirv_matched_varying_minimum_unsigned_integer_half ||
        shader == GuestTriangleShader::spirv_matched_varying_default_alpha;
    const bool use_varying_passthrough =
        shader == GuestTriangleShader::spirv_matched_varying_move ||
        shader == GuestTriangleShader::spirv_matched_varying_add_half ||
        shader == GuestTriangleShader::spirv_matched_varying_multiply_half ||
        shader == GuestTriangleShader::spirv_matched_varying_maximum_half ||
        shader == GuestTriangleShader::spirv_matched_varying_minimum_half ||
        shader == GuestTriangleShader::spirv_matched_varying_add_negative_half ||
        shader == GuestTriangleShader::spirv_matched_varying_xor_zero ||
        shader == GuestTriangleShader::
            spirv_matched_varying_maximum_signed_integer_half ||
        shader == GuestTriangleShader::
            spirv_matched_varying_minimum_unsigned_integer_half ||
        shader == GuestTriangleShader::spirv_matched_varying_default_alpha;
    constexpr std::uint32_t vertex_buffer_address = 0x10000U;
    constexpr std::uint32_t index_buffer_address = 0x4000U;
    constexpr std::uint32_t big_endian_index_buffer_address = 0x5000U;
    constexpr std::uint32_t pm4_display_list_address = 0x6000U;
    constexpr std::uint32_t call_stub_address = 0x2000U;
    constexpr std::uint32_t fetch_shader_descriptor_address = 0x3000U;
    constexpr std::uint32_t fetch_shader_program_address = 0x3100U;
    constexpr std::uint32_t attribute_stream_address = 0x3200U;
    constexpr std::uint32_t vertex_shader_descriptor_address = 0x3300U;
    constexpr std::uint32_t vertex_shader_program_address = 0x3400U;
    constexpr std::uint32_t pixel_shader_descriptor_address = 0x3500U;
    constexpr std::uint32_t pixel_shader_program_address = 0x3600U;
    constexpr std::size_t vertex_shader_output_configuration_word_index = 2U;
    constexpr std::size_t vertex_shader_output_id_count_word_index = 3U;
    constexpr std::size_t vertex_shader_first_output_id_word_index = 4U;
    constexpr std::size_t pixel_shader_input_count_word_index = 4U;
    constexpr std::size_t pixel_shader_first_input_control_word_index = 5U;
    constexpr std::uint32_t color_buffer_descriptor_address = 0x7000U;
    constexpr std::uint32_t color_buffer_image_address = 0x20000U;
    constexpr std::uint32_t color_buffer_scanout_address = 0x500000U;
    constexpr std::uint32_t color_buffer_width = 1280U;
    constexpr std::uint32_t color_buffer_height = 720U;
    constexpr std::uint32_t color_buffer_size =
        color_buffer_width * color_buffer_height * 4U;
    constexpr std::uint32_t guest_bctrl_instruction = 0x4E800421U;
    constexpr std::uint32_t gx2_primitive_mode_triangles = 4U;
    constexpr std::uint32_t gx2_primitive_mode_triangle_fan = 5U;
    constexpr std::uint32_t gx2_primitive_mode_triangle_strip = 6U;
    constexpr std::uint32_t vertex_buffer_index = 2U;
    constexpr std::uint32_t vertex_buffer_stride = 16U;
    constexpr std::uint32_t position_attribute_offset = 4U;
    constexpr std::uint32_t vertex_buffer_size = 4U * vertex_buffer_stride;
    constexpr std::uint16_t guest_position_gpr_selector = 1U;
    // OP2 ALU_WORD0 bit 25 marks the source-1 negate modifier.
    constexpr std::uint32_t latte_source1_negate_mask = 1U << 25U;
    constexpr std::array<affogato::gpu::vulkan::DiagnosticVertex, 3U> vertices{{
        {-0.65F, -0.60F}, {0.65F, -0.60F}, {0.0F, 0.70F}}};

    affogato::Emulator emulator;
    auto& core = emulator.core();
    core.configure_guest_heap(
        0x02000000U, static_cast<std::uint32_t>(core.memory.size()));
    core.memory.write32_be(call_stub_address, guest_bctrl_instruction);
    const auto write_vertex_positions = [&core](
        std::span<const affogato::gpu::vulkan::DiagnosticVertex> positions)
    {
        for (std::size_t index = 0U; index < positions.size(); ++index)
        {
            core.memory.write32_be(
                vertex_buffer_address + static_cast<std::uint32_t>(
                    index * vertex_buffer_stride + position_attribute_offset),
                std::bit_cast<std::uint32_t>(positions[index].x));
            core.memory.write32_be(
                vertex_buffer_address + static_cast<std::uint32_t>(
                    index * vertex_buffer_stride + position_attribute_offset + 4U),
                std::bit_cast<std::uint32_t>(positions[index].y));
        }
    };
    write_vertex_positions(vertices);

    const auto invoke = [&](const char* symbol,
                            std::span<const std::uint32_t> arguments) {
        const std::uint32_t import = core.hle.bind_import("gx2", symbol);
        for (std::size_t index = 0U; index < arguments.size(); ++index)
        {
            core.state.gpr[index + 3U] = arguments[index];
        }
        core.state.cia = call_stub_address;
        core.state.ctr = import;
        if (core.run(1U).reason !=
                affogato::cpu::espresso::StopReason::instruction_limit ||
            core.state.cia != import)
        {
            throw std::runtime_error(
                std::string("Guest bctrl did not reach GX2 HLE: ") + symbol);
        }
        if (core.run(1U).reason !=
                affogato::cpu::espresso::StopReason::instruction_limit ||
            core.state.cia != call_stub_address + 4U)
        {
            throw std::runtime_error(
                std::string("GX2 guest triangle HLE failed: ") + symbol);
        }
    };

    invoke("GX2Init", std::array<std::uint32_t, 1U>{0U});
    invoke("GX2SetTVBuffer", std::array<std::uint32_t, 5U>{
        color_buffer_scanout_address,
        color_buffer_size,
        3U,
        0x1AU,
        1U});
    core.gx2.capture_draw_vertex_data = true;
    core.gx2.capture_draw_color_buffer_data = true;

    std::array<std::uint32_t, 39U> color_buffer_descriptor{};
    color_buffer_descriptor[0U] = 1U;
    color_buffer_descriptor[1U] = color_buffer_width;
    color_buffer_descriptor[2U] = color_buffer_height;
    color_buffer_descriptor[3U] = 1U;
    color_buffer_descriptor[4U] = 1U;
    color_buffer_descriptor[5U] = 0x1AU;
    color_buffer_descriptor[7U] = 0x02U;
    color_buffer_descriptor[8U] = color_buffer_size;
    color_buffer_descriptor[9U] = color_buffer_image_address;
    color_buffer_descriptor[12U] =
        affogato::gpu::gx2_tile_mode_linear_aligned;
    color_buffer_descriptor[14U] = 1U;
    color_buffer_descriptor[15U] = color_buffer_width;
    color_buffer_descriptor[31U] = 1U;
    for (std::size_t index = 0U;
         index < color_buffer_descriptor.size();
         ++index)
    {
        core.memory.write32_be(
            color_buffer_descriptor_address +
                static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
            color_buffer_descriptor[index]);
    }
    std::vector<std::uint8_t> initial_color_buffer(color_buffer_size);
    for (std::size_t pixel_offset = 0U;
         pixel_offset < initial_color_buffer.size();
         pixel_offset += 4U)
    {
        initial_color_buffer[pixel_offset] = 0U;
        initial_color_buffer[pixel_offset + 1U] = 0U;
        initial_color_buffer[pixel_offset + 2U] = 180U;
        initial_color_buffer[pixel_offset + 3U] = 255U;
    }
    core.memory.write_bytes(color_buffer_image_address, initial_color_buffer);
    invoke("GX2SetColorBuffer", std::array<std::uint32_t, 2U>{
        color_buffer_descriptor_address,
        0U});

    // The CPU diagnostic uses MOV; the SPIR-V variant uses ADD with zero to
    // preserve the same position while exercising binary32 shader arithmetic.
    std::array<std::uint32_t, 0x134U / sizeof(std::uint32_t)>
        vertex_shader_descriptor{};
    vertex_shader_descriptor[0x40U / sizeof(std::uint32_t)] = 32U;
    vertex_shader_descriptor[0x44U / sizeof(std::uint32_t)] = 0U;
    vertex_shader_descriptor[
        vertex_shader_output_configuration_word_index] = 0U;
    vertex_shader_descriptor[
        vertex_shader_output_id_count_word_index] = 1U;
    vertex_shader_descriptor[
        vertex_shader_first_output_id_word_index] = 0U;
    if (use_varying_passthrough)
    {
        vertex_shader_descriptor[
            vertex_shader_first_output_id_word_index] = 0x31U;
    }
    for (std::size_t semantic_index = 1U; semantic_index < 32U; ++semantic_index)
    {
        vertex_shader_descriptor[
            0x44U / sizeof(std::uint32_t) + semantic_index] = 0xFFFFFFFFU;
    }
    vertex_shader_descriptor[0xD0U / sizeof(std::uint32_t)] = 76U * 4U;
    vertex_shader_descriptor[0xD4U / sizeof(std::uint32_t)] =
        vertex_shader_program_address;
    std::array<std::uint32_t, 76U> vertex_shader_program{};
    vertex_shader_program[0U] = 0x20000000U;
    vertex_shader_program[1U] = 0x00000CA0U;
    vertex_shader_program[2U] = 0x3CA00100U;
    vertex_shader_program[3U] = 0x88062094U;
    vertex_shader_program[64U] = 0x01000000U;
    vertex_shader_program[65U] = 0x900C6000U;
    vertex_shader_program[66U] = 0x01040000U;
    vertex_shader_program[67U] = 0x900C6020U;
    vertex_shader_program[68U] = 0x01080000U;
    vertex_shader_program[69U] = 0x900C6040U;
    vertex_shader_program[70U] = 0x010C0080U;
    vertex_shader_program[71U] = 0x900C6060U;
    if (use_varying_passthrough)
    {
        constexpr std::uint32_t parameter_export_word0 = 0x0001C000U;
        constexpr std::uint32_t parameter_export_word1 = 0x13800688U;
        vertex_shader_program[2U] = std::byteswap(parameter_export_word0);
        vertex_shader_program[3U] = std::byteswap(parameter_export_word1);
        vertex_shader_program[4U] = 0x3CA00100U;
        vertex_shader_program[5U] = 0x88062094U;
    }
    if (use_latte_spirv &&
        (shader == GuestTriangleShader::spirv_conditional_move ||
         shader == GuestTriangleShader::spirv_integer_conditional_move ||
         shader == GuestTriangleShader::spirv_mixed_conditional_move))
    {
        const std::uint16_t conditional_move_opcode =
            shader == GuestTriangleShader::spirv_integer_conditional_move
                ? affogato::gpu::
                    latte_alu_op3_conditional_move_greater_equal_integer_opcode
                : affogato::gpu::
                    latte_alu_op3_conditional_move_greater_equal_opcode;
        for (std::uint32_t component = 0U; component < 4U; ++component)
        {
            const std::uint32_t slot = 64U + component * 2U;
            if (shader == GuestTriangleShader::spirv_mixed_conditional_move &&
                component == 0U)
            {
                const std::uint32_t source_word = guest_position_gpr_selector;
                const std::uint32_t destination_word =
                    (static_cast<std::uint32_t>(
                         affogato::gpu::latte_alu_op2_move_opcode)
                     << 7U) |
                    (1U << 4U) |
                    (3U << 21U);
                vertex_shader_program[slot] = std::byteswap(source_word);
                vertex_shader_program[slot + 1U] =
                    std::byteswap(destination_word);
                continue;
            }
            const std::uint32_t source_word =
                guest_position_gpr_selector |
                (component << 10U) |
                (static_cast<std::uint32_t>(
                     affogato::gpu::latte_alu_source_one_float_selector)
                 << 13U) |
                (component << 23U) |
                (component == 3U ? 0x80000000U : 0U);
            const std::uint32_t destination_word =
                affogato::gpu::latte_alu_source_zero_selector |
                (component << 10U) |
                (static_cast<std::uint32_t>(conditional_move_opcode)
                 << affogato::gpu::latte_alu_opcode_encoding_shift) |
                (3U << 21U) |
                (component << 29U);
            vertex_shader_program[slot] = std::byteswap(source_word);
            vertex_shader_program[slot + 1U] =
                std::byteswap(destination_word);
        }
    }
    else if (use_latte_spirv)
    {
        std::uint32_t operation_opcode =
            affogato::gpu::latte_alu_op2_add_float_opcode;
        std::uint16_t second_operand_selector =
            affogato::gpu::latte_alu_source_zero_selector;
        if (shader == GuestTriangleShader::spirv_multiply_one)
        {
            operation_opcode =
                affogato::gpu::latte_alu_op2_multiply_ieee_opcode;
            second_operand_selector =
                affogato::gpu::latte_alu_source_one_float_selector;
        }
        else if (shader == GuestTriangleShader::spirv_multiply_zero)
        {
            operation_opcode =
                affogato::gpu::latte_alu_op2_multiply_ieee_opcode;
        }
        else if (shader == GuestTriangleShader::spirv_maximum_self)
        {
            operation_opcode =
                affogato::gpu::latte_alu_op2_maximum_dx10_opcode;
            second_operand_selector = guest_position_gpr_selector;
        }
        else if (shader == GuestTriangleShader::spirv_minimum_self)
        {
            operation_opcode =
                affogato::gpu::latte_alu_op2_minimum_dx10_opcode;
            second_operand_selector = guest_position_gpr_selector;
        }
        else if (shader == GuestTriangleShader::spirv_maximum_zero)
        {
            operation_opcode =
                affogato::gpu::latte_alu_op2_maximum_dx10_opcode;
        }
        else if (shader == GuestTriangleShader::spirv_minimum_zero)
        {
            operation_opcode =
                affogato::gpu::latte_alu_op2_minimum_dx10_opcode;
        }
        for (std::uint32_t component = 0U; component < 4U; ++component)
        {
            const std::uint32_t slot = 64U + component * 2U;
            std::uint32_t component_opcode = operation_opcode;
            std::uint16_t component_source0_selector = 1U;
            std::uint16_t component_source1_selector =
                second_operand_selector;
            if (shader == GuestTriangleShader::spirv_integer_compare_clip_z)
            {
                component_opcode = affogato::gpu::latte_alu_op2_move_opcode;
                if (component == 2U)
                {
                    component_opcode = affogato::gpu::
                        latte_alu_op2_set_greater_integer_opcode;
                    component_source0_selector = affogato::gpu::
                        latte_alu_source_zero_selector;
                    component_source1_selector = affogato::gpu::
                        latte_alu_source_one_integer_selector;
                }
            }
            if (shader == GuestTriangleShader::spirv_integer_alu_identity)
            {
                component_source1_selector = affogato::gpu::
                    latte_alu_source_zero_selector;
                switch (component)
                {
                case 0U:
                    component_opcode = affogato::gpu::
                        latte_alu_op2_and_integer_opcode;
                    component_source1_selector = affogato::gpu::
                        latte_alu_source_negative_one_integer_selector;
                    break;
                case 1U:
                    component_opcode = affogato::gpu::
                        latte_alu_op2_add_integer_opcode;
                    break;
                case 2U:
                    component_opcode = affogato::gpu::
                        latte_alu_op2_subtract_integer_opcode;
                    component_source0_selector = affogato::gpu::
                        latte_alu_source_one_integer_selector;
                    component_source1_selector = affogato::gpu::
                        latte_alu_source_one_integer_selector;
                    break;
                case 3U:
                    component_opcode = affogato::gpu::
                        latte_alu_op2_or_integer_opcode;
                    break;
                default:
                    throw std::runtime_error(
                        "GX2 integer ALU smoke test has an invalid component");
                }
            }
            if (shader == GuestTriangleShader::spirv_add_one_x &&
                component == 0U)
            {
                component_source1_selector =
                    affogato::gpu::latte_alu_source_one_float_selector;
            }
            if (shader == GuestTriangleShader::spirv_add_negative_one_x &&
                component == 0U)
            {
                component_source1_selector =
                    affogato::gpu::latte_alu_source_one_float_selector;
            }
            if (shader == GuestTriangleShader::spirv_multiply_zero &&
                component == 3U)
            {
                component_source1_selector =
                    affogato::gpu::latte_alu_source_one_float_selector;
            }
            const std::uint32_t source_word =
                component_source0_selector |
                (static_cast<std::uint32_t>(component_source1_selector) << 13U) |
                (component << 10U) |
                (component << 23U) |
                (shader == GuestTriangleShader::spirv_add_negative_one_x &&
                         component == 0U
                     ? latte_source1_negate_mask
                     : 0U) |
                (component == 3U ? 0x80000000U : 0U);
            const std::uint32_t destination_word =
                0x00600010U |
                (component_opcode << 7U) |
                (component << 29U);
            vertex_shader_program[slot] = std::byteswap(source_word);
            vertex_shader_program[slot + 1U] =
                std::byteswap(destination_word);
        }
    }
    for (std::size_t index = 0U; index < vertex_shader_descriptor.size(); ++index)
    {
        core.memory.write32_be(
            vertex_shader_descriptor_address + static_cast<std::uint32_t>(
                index * sizeof(std::uint32_t)),
            vertex_shader_descriptor[index]);
    }
    for (std::size_t index = 0U; index < vertex_shader_program.size(); ++index)
    {
        core.memory.write32_be(
            vertex_shader_program_address + static_cast<std::uint32_t>(
                index * sizeof(std::uint32_t)),
            vertex_shader_program[index]);
    }
    invoke("GX2SetVertexShader", std::array<std::uint32_t, 1U>{
        vertex_shader_descriptor_address});

    if (use_latte_pixel_spirv)
    {
        constexpr std::array<std::uint16_t, 4U> color_selectors{
            affogato::gpu::latte_alu_source_one_float_selector,
            affogato::gpu::latte_alu_source_zero_selector,
            affogato::gpu::latte_alu_source_one_float_selector,
            affogato::gpu::latte_alu_source_one_float_selector};
        std::array<std::uint32_t, 0xE8U / sizeof(std::uint32_t)>
            pixel_shader_descriptor{};
        if (shader == GuestTriangleShader::spirv_default_pixel_input ||
            use_varying_passthrough)
        {
            constexpr std::uint32_t pixel_input_semantic = 0x31U;
            const std::uint32_t pixel_input_default_value =
                shader == GuestTriangleShader::spirv_default_pixel_input
                    ? 1U
                    : 0U;
            constexpr std::uint32_t pixel_input_default_value_shift = 8U;
            pixel_shader_descriptor[pixel_shader_input_count_word_index] = 1U;
            pixel_shader_descriptor[
                pixel_shader_first_input_control_word_index] =
                    pixel_input_semantic |
                    (pixel_input_default_value <<
                     pixel_input_default_value_shift);
            if (shader ==
                GuestTriangleShader::spirv_matched_varying_default_alpha)
            {
                constexpr std::uint32_t default_alpha_input_semantic = 0x32U;
                constexpr std::uint32_t default_zero_alpha_value = 2U;
                pixel_shader_descriptor[
                    pixel_shader_input_count_word_index] = 2U;
                pixel_shader_descriptor[
                    pixel_shader_first_input_control_word_index + 1U] =
                        default_alpha_input_semantic |
                        (default_zero_alpha_value <<
                         pixel_input_default_value_shift);
            }
        }
        pixel_shader_descriptor[0xA4U / sizeof(std::uint32_t)] = 72U * 4U;
        pixel_shader_descriptor[0xA8U / sizeof(std::uint32_t)] =
            pixel_shader_program_address;

        std::array<std::uint32_t, 72U> pixel_shader_program{};
        pixel_shader_program[0U] = 0x20000000U;
        pixel_shader_program[1U] = 0x00000CA0U;
        pixel_shader_program[2U] = 0x00000100U;
        pixel_shader_program[3U] = 0x88062014U;
        for (std::uint32_t component = 0U; component < 4U; ++component)
        {
            const std::uint32_t slot = 64U + component * 2U;
            std::uint16_t source_selector =
                shader == GuestTriangleShader::spirv_default_pixel_input ||
                        use_varying_passthrough
                    ? 0U
                    : color_selectors[component];
            std::uint32_t source1_encoding = 0U;
            std::uint16_t pixel_opcode =
                affogato::gpu::latte_alu_op2_move_opcode;
            if (shader ==
                    GuestTriangleShader::spirv_matched_varying_default_alpha &&
                component == 3U)
            {
                source_selector = 1U;
            }
            if (shader == GuestTriangleShader::spirv_matched_varying_add_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode = affogato::gpu::latte_alu_op2_add_float_opcode;
            }
            else if (shader ==
                     GuestTriangleShader::spirv_matched_varying_add_negative_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode = affogato::gpu::latte_alu_op2_add_float_opcode;
            }
            else if (shader ==
                     GuestTriangleShader::spirv_matched_varying_xor_zero)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_zero_selector);
                pixel_opcode = affogato::gpu::latte_alu_op2_xor_integer_opcode;
            }
            else if (shader ==
                     GuestTriangleShader::spirv_matched_varying_multiply_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode =
                    affogato::gpu::latte_alu_op2_multiply_ieee_opcode;
            }
            else if (shader ==
                     GuestTriangleShader::spirv_matched_varying_maximum_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode =
                    affogato::gpu::latte_alu_op2_maximum_dx10_opcode;
            }
            else if (shader ==
                     GuestTriangleShader::spirv_matched_varying_minimum_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode =
                    affogato::gpu::latte_alu_op2_minimum_dx10_opcode;
            }
            else if (shader == GuestTriangleShader::
                         spirv_matched_varying_maximum_signed_integer_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode = affogato::gpu::
                    latte_alu_op2_maximum_signed_integer_opcode;
            }
            else if (shader == GuestTriangleShader::
                         spirv_matched_varying_minimum_unsigned_integer_half)
            {
                source1_encoding = static_cast<std::uint32_t>(
                    affogato::gpu::latte_alu_source_half_float_selector);
                pixel_opcode = affogato::gpu::
                    latte_alu_op2_minimum_unsigned_integer_opcode;
            }
            const std::uint32_t source1_modifier_encoding =
                shader ==
                        GuestTriangleShader::spirv_matched_varying_add_negative_half
                    ? (component == 3U ? (1U << 25U) : 0U)
                    : 0U;
            const std::uint32_t source_word =
                source_selector |
                (source1_encoding << 13U) |
                (component << 10U) |
                (component << 23U) |
                source1_modifier_encoding |
                (component == 3U ? 0x80000000U : 0U);
            const std::uint32_t source0_absolute_encoding =
                shader ==
                        GuestTriangleShader::spirv_matched_varying_add_negative_half
                    ? 1U
                    : 0U;
            const std::uint32_t destination_word =
                0x00400010U |
                (static_cast<std::uint32_t>(pixel_opcode) << 7U) |
                (component << 29U) |
                source0_absolute_encoding;
            pixel_shader_program[slot] = std::byteswap(source_word);
            pixel_shader_program[slot + 1U] =
                std::byteswap(destination_word);
        }

        for (std::size_t index = 0U;
             index < pixel_shader_descriptor.size();
             ++index)
        {
            core.memory.write32_be(
                pixel_shader_descriptor_address + static_cast<std::uint32_t>(
                    index * sizeof(std::uint32_t)),
                pixel_shader_descriptor[index]);
        }
        for (std::size_t index = 0U;
             index < pixel_shader_program.size();
             ++index)
        {
            core.memory.write32_be(
                pixel_shader_program_address + static_cast<std::uint32_t>(
                    index * sizeof(std::uint32_t)),
                pixel_shader_program[index]);
        }
        invoke("GX2SetPixelShader", std::array<std::uint32_t, 1U>{
            pixel_shader_descriptor_address});
    }

    invoke("GX2SetAttribBuffer", std::array<std::uint32_t, 4U>{
        vertex_buffer_index,
        vertex_buffer_size,
        vertex_buffer_stride,
        vertex_buffer_address});
    std::array<std::uint32_t, 8U> attribute_stream{
        0U, vertex_buffer_index, position_attribute_offset, 0x80DU,
        0U, 0U, 0x00010405U, 3U};
    for (std::size_t index = 0U; index < attribute_stream.size(); ++index)
    {
        core.memory.write32_be(
            attribute_stream_address +
                static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
            attribute_stream[index]);
    }
    invoke("GX2InitFetchShaderEx", std::array<std::uint32_t, 6U>{
        fetch_shader_descriptor_address,
        fetch_shader_program_address,
        1U,
        attribute_stream_address,
        0U,
        0U});
    invoke("GX2SetFetchShader", std::array<std::uint32_t, 1U>{
        fetch_shader_descriptor_address});
    invoke("GX2DrawEx", std::array<std::uint32_t, 4U>{
        gx2_primitive_mode_triangles, 3U, 0U, 1U});
    constexpr std::array<affogato::gpu::vulkan::DiagnosticVertex, 3U>
        second_vertices{{
            {0.72F, -0.35F}, {0.98F, -0.35F}, {0.85F, 0.20F}}};
    write_vertex_positions(second_vertices);
    constexpr std::array<std::uint8_t, 6U> indices{2U, 0U, 0U, 0U, 1U, 0U};
    core.memory.write16_be(index_buffer_address, 0xEEEEU);
    for (std::size_t byte = 0U; byte < indices.size(); ++byte)
    {
        core.memory.write8(
            index_buffer_address + 2U + static_cast<std::uint32_t>(byte),
            indices[byte]);
    }
    invoke("GX2DrawIndexedEx", std::array<std::uint32_t, 6U>{
        gx2_primitive_mode_triangles,
        3U,
        0U,
        index_buffer_address + 2U,
        0U,
        1U});

    constexpr std::array<affogato::gpu::vulkan::DiagnosticVertex, 3U>
        third_vertices{{
            {-0.98F, -0.35F}, {-0.72F, -0.35F}, {-0.85F, 0.20F}}};
    write_vertex_positions(third_vertices);
    constexpr std::array<std::uint8_t, 12U> big_endian_indices{
        0U, 0U, 0U, 2U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U};
    core.memory.write32_be(big_endian_index_buffer_address, 0xEEEEEEEEU);
    for (std::size_t byte = 0U; byte < big_endian_indices.size(); ++byte)
    {
        core.memory.write8(
            big_endian_index_buffer_address + 4U + static_cast<std::uint32_t>(byte),
            big_endian_indices[byte]);
    }
    invoke("GX2DrawIndexedEx", std::array<std::uint32_t, 6U>{
        gx2_primitive_mode_triangles,
        3U,
        9U,
        big_endian_index_buffer_address + 4U,
        0U,
        1U});

    constexpr std::array<affogato::gpu::vulkan::DiagnosticVertex, 4U>
        strip_vertices{{
            {0.55F, 0.35F}, {0.90F, 0.35F}, {0.55F, 0.65F}, {0.90F, 0.65F}}};
    write_vertex_positions(strip_vertices);
    invoke("GX2DrawEx", std::array<std::uint32_t, 4U>{
        gx2_primitive_mode_triangle_strip, 4U, 0U, 1U});

    constexpr std::array<affogato::gpu::vulkan::DiagnosticVertex, 4U>
        fan_vertices{{
            {-0.95F, 0.35F}, {-0.60F, 0.35F}, {-0.60F, 0.65F}, {-0.95F, 0.65F}}};
    write_vertex_positions(fan_vertices);
    invoke("GX2DrawEx", std::array<std::uint32_t, 4U>{
        gx2_primitive_mode_triangle_fan, 4U, 0U, 1U});

    attribute_stream[3U] = 0x08U;
    core.memory.write32_be(attribute_stream_address + 12U, attribute_stream[3U]);
    invoke("GX2InitFetchShaderEx", std::array<std::uint32_t, 6U>{
        fetch_shader_descriptor_address,
        fetch_shader_program_address,
        1U,
        attribute_stream_address,
        0U,
        0U});
    invoke("GX2SetFetchShader", std::array<std::uint32_t, 1U>{
        fetch_shader_descriptor_address});
    constexpr std::array<std::uint16_t, 8U> float16_positions{
        0xB400U, 0xBB00U, 0x3400U, 0xBB00U,
        0x0000U, 0xB900U, 0x0000U, 0x3900U};
    for (std::size_t vertex = 0U; vertex < 4U; ++vertex)
    {
        core.memory.write16_be(
            vertex_buffer_address + static_cast<std::uint32_t>(
                vertex * vertex_buffer_stride + position_attribute_offset),
            float16_positions[vertex * 2U]);
        core.memory.write16_be(
            vertex_buffer_address + static_cast<std::uint32_t>(
                vertex * vertex_buffer_stride + position_attribute_offset + 2U),
            float16_positions[vertex * 2U + 1U]);
    }
    invoke("GX2DrawEx", std::array<std::uint32_t, 4U>{
        gx2_primitive_mode_triangles, 3U, 0U, 1U});

    constexpr std::array<std::uint32_t, 13U> pm4_draw_words{
        0xC0016F00U, 0U, 0U,
        0xC0016800U, 0x256U, gx2_primitive_mode_triangles,
        0xC0002A00U, 9U,
        0xC0002F00U, 1U,
        0xC0012D00U, 3U, 0U};
    for (std::size_t index = 0U; index < pm4_draw_words.size(); ++index)
    {
        core.memory.write32_be(
            pm4_display_list_address + static_cast<std::uint32_t>(
                index * sizeof(std::uint32_t)),
            pm4_draw_words[index]);
    }
    invoke("GX2DirectCallDisplayList", std::array<std::uint32_t, 2U>{
        pm4_display_list_address,
        static_cast<std::uint32_t>(pm4_draw_words.size() *
                                   sizeof(std::uint32_t))});

    constexpr std::array<std::uint32_t, 16U> pm4_indexed_draw_words{
        0xC0016F00U, 0U, 1U,
        0xC0016800U, 0x256U, gx2_primitive_mode_triangles,
        0xC0002A00U, 0U,
        0xC0002F00U, 1U,
        0xC0042700U, 3U, index_buffer_address + 2U, 0U, 3U, 0U};
    for (std::size_t index = 0U; index < pm4_indexed_draw_words.size(); ++index)
    {
        core.memory.write32_be(
            pm4_display_list_address + static_cast<std::uint32_t>(
                index * sizeof(std::uint32_t)),
            pm4_indexed_draw_words[index]);
    }
    invoke("GX2DirectCallDisplayList", std::array<std::uint32_t, 2U>{
        pm4_display_list_address,
        static_cast<std::uint32_t>(pm4_indexed_draw_words.size() *
                                   sizeof(std::uint32_t))});

    constexpr std::array<std::uint32_t, 16U> pm4_immediate_indexed_draw_words{
        0xC0016F00U, 0U, 0U,
        0xC0016800U, 0x256U, gx2_primitive_mode_triangles,
        0xC0002A00U, 9U,
        0xC0002F00U, 1U,
        0xC0042E00U, 3U, 0U, 0U, 1U, 2U};
    for (std::size_t index = 0U;
         index < pm4_immediate_indexed_draw_words.size();
         ++index)
    {
        core.memory.write32_be(
            pm4_display_list_address + static_cast<std::uint32_t>(
                index * sizeof(std::uint32_t)),
            pm4_immediate_indexed_draw_words[index]);
    }
    invoke("GX2DirectCallDisplayList", std::array<std::uint32_t, 2U>{
        pm4_display_list_address,
        static_cast<std::uint32_t>(pm4_immediate_indexed_draw_words.size() *
                                   sizeof(std::uint32_t))});

    attribute_stream[3U] = 0x0AU;
    attribute_stream[7U] = 0U; // Explicit GX2_ENDIAN_SWAP_NONE.
    attribute_stream[6U] = 0x00010203U;
    core.memory.write32_be(attribute_stream_address + 12U, attribute_stream[3U]);
    core.memory.write32_be(attribute_stream_address + 24U, attribute_stream[6U]);
    core.memory.write32_be(attribute_stream_address + 28U, attribute_stream[7U]);
    invoke("GX2InitFetchShaderEx", std::array<std::uint32_t, 6U>{
        fetch_shader_descriptor_address,
        fetch_shader_program_address,
        1U,
        attribute_stream_address,
        0U,
        0U});
    invoke("GX2SetFetchShader", std::array<std::uint32_t, 1U>{
        fetch_shader_descriptor_address});
    constexpr std::array<std::array<std::uint8_t, 4U>, 3U> unorm8_positions{{
        {26U, 32U, 64U, 255U},
        {230U, 32U, 128U, 255U},
        {128U, 224U, 192U, 255U}}};
    for (std::size_t vertex = 0U; vertex < unorm8_positions.size(); ++vertex)
    {
        const std::uint32_t vertex_address = vertex_buffer_address +
            static_cast<std::uint32_t>(
                vertex * vertex_buffer_stride + position_attribute_offset);
        for (std::size_t component = 0U;
             component < unorm8_positions[vertex].size();
             ++component)
        {
            core.memory.write8(
                vertex_address + static_cast<std::uint32_t>(component),
                unorm8_positions[vertex][component]);
        }
    }
    invoke("GX2DrawEx", std::array<std::uint32_t, 4U>{
        gx2_primitive_mode_triangles, 3U, 0U, 1U});

    const auto commands = affogato::cpu::espresso::take_gx2_commands(core.gx2);
    affogato::gpu::Gx2RegisterState captured_state;
    captured_state.apply(commands);
    if (captured_state.draw_calls().size() != 10U ||
        captured_state.draw_vertex_buffers().size() != 10U ||
        !captured_state.draw_vertex_buffers()[0U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[1U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[2U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[3U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[4U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[5U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[6U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[7U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[8U][vertex_buffer_index] ||
        !captured_state.draw_vertex_buffers()[9U][vertex_buffer_index] ||
        captured_state.draw_fetch_shaders().size() != 10U ||
        !captured_state.draw_fetch_shaders()[0U] ||
        !captured_state.draw_fetch_shaders()[1U] ||
        !captured_state.draw_fetch_shaders()[2U] ||
        !captured_state.draw_fetch_shaders()[3U] ||
        !captured_state.draw_fetch_shaders()[4U] ||
        !captured_state.draw_fetch_shaders()[5U] ||
        !captured_state.draw_fetch_shaders()[6U] ||
        !captured_state.draw_fetch_shaders()[7U] ||
        !captured_state.draw_fetch_shaders()[8U] ||
        !captured_state.draw_fetch_shaders()[9U] ||
        captured_state.draw_calls()[0U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[1U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[2U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[3U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[4U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[5U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[6U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[7U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[8U].draw_vertex_data.size() != 1U ||
        captured_state.draw_calls()[9U].draw_vertex_data.size() != 1U ||
        !captured_state.draw_calls()[0U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[1U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[2U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[3U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[4U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[5U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[6U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[7U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[8U].draw_vertex_data[0U].is_valid ||
        !captured_state.draw_calls()[9U].draw_vertex_data[0U].is_valid ||
        captured_state.draw_fetch_shaders()[5U]->program_words[0U] !=
            (1U | (vertex_buffer_index << 8U) | (3U << 26U)) ||
        captured_state.draw_fetch_shaders()[5U]->program_words[1U] !=
            ((1U << 12U) | (4U << 15U) | (5U << 18U) | (0x08U << 22U)) ||
        captured_state.draw_fetch_shaders()[5U]->program_words[2U] !=
            (position_attribute_offset | (1U << 16U) | (1U << 19U)) ||
        captured_state.draw_calls()[0U].draw_vertex_data[0U].bytes ==
            captured_state.draw_calls()[1U].draw_vertex_data[0U].bytes ||
        captured_state.draw_calls()[1U].draw_vertex_data[0U].bytes ==
            captured_state.draw_calls()[2U].draw_vertex_data[0U].bytes ||
        !captured_state.draw_calls()[1U].draw_index_data ||
        !captured_state.draw_calls()[1U].draw_index_data->is_valid ||
        captured_state.draw_calls()[1U].draw_index_data->bytes !=
            std::vector<std::uint8_t>{2U, 0U, 0U, 0U, 1U, 0U} ||
        captured_state.draw_calls()[1U].draw_index_data->byte_offset != 0U ||
        !captured_state.draw_calls()[2U].draw_index_data ||
        !captured_state.draw_calls()[2U].draw_index_data->is_valid ||
        captured_state.draw_calls()[2U].draw_index_data->bytes !=
            std::vector<std::uint8_t>{
                0U, 0U, 0U, 2U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U} ||
        captured_state.draw_calls()[2U].draw_index_data->byte_offset != 0U ||
        captured_state.draw_calls()[0U].draw_call !=
            std::optional<affogato::cpu::espresso::Gx2DrawCall>{
                affogato::cpu::espresso::Gx2DrawCall{
                    gx2_primitive_mode_triangles, 3U, false, 0U, 0U, 0U, 1U, 0U}} ||
        captured_state.draw_calls()[1U].draw_call !=
            std::optional<affogato::cpu::espresso::Gx2DrawCall>{
                affogato::cpu::espresso::Gx2DrawCall{
                    gx2_primitive_mode_triangles,
                    3U,
                    true,
                    0U,
                    index_buffer_address + 2U,
                    0U,
                    1U,
                    0U,
                    0U}} ||
        captured_state.draw_calls()[2U].draw_call !=
            std::optional<affogato::cpu::espresso::Gx2DrawCall>{
                affogato::cpu::espresso::Gx2DrawCall{
                    gx2_primitive_mode_triangles,
                    3U,
                    true,
                    9U,
                    big_endian_index_buffer_address + 4U,
                    0U,
                    1U,
                    0U,
                    0U}} ||
        captured_state.draw_calls()[3U].draw_call !=
            std::optional<affogato::cpu::espresso::Gx2DrawCall>{
                affogato::cpu::espresso::Gx2DrawCall{
                    gx2_primitive_mode_triangle_strip,
                    4U,
                    false,
                    0U,
                    0U,
                    0U,
                    1U,
                    0U}} ||
        captured_state.draw_calls()[4U].draw_call !=
            std::optional<affogato::cpu::espresso::Gx2DrawCall>{
                affogato::cpu::espresso::Gx2DrawCall{
                    gx2_primitive_mode_triangle_fan,
                    4U,
                    false,
                    0U,
                    0U,
                    0U,
                    1U,
                    0U}} ||
        captured_state.draw_calls()[5U].draw_call !=
            std::optional<affogato::cpu::espresso::Gx2DrawCall>{
                affogato::cpu::espresso::Gx2DrawCall{
                    gx2_primitive_mode_triangles,
                    3U,
                    false,
                    0U,
                    0U,
                    0U,
                    1U,
                    0U}})
    {
        throw std::runtime_error(
            "GX2 triangle call or vertex-buffer binding was not captured correctly");
    }

    const affogato::cpu::espresso::Gx2DrawCall expected_pm4_draw{
        gx2_primitive_mode_triangles, 3U, false, 0U, 0U, 0U, 1U, 0U, 0U};
    if (captured_state.draw_calls()[6U].draw_call != expected_pm4_draw)
    {
        throw std::runtime_error(
            "GX2DirectCallDisplayList draw packet was not captured correctly");
    }

    const auto& captured_pm4_indexed_draw = captured_state.draw_calls()[7U];
    if (!captured_pm4_indexed_draw.draw_call ||
        *captured_pm4_indexed_draw.draw_call !=
            affogato::cpu::espresso::Gx2DrawCall{
                gx2_primitive_mode_triangles,
                3U,
                true,
                0U,
                index_buffer_address + 2U,
                0U,
                1U,
                0U,
                0U,
                1U} ||
        !captured_pm4_indexed_draw.draw_index_data ||
        !captured_pm4_indexed_draw.draw_index_data->is_valid ||
        captured_pm4_indexed_draw.draw_index_data->bytes !=
            std::vector<std::uint8_t>{2U, 0U, 0U, 0U, 1U, 0U})
    {
        throw std::runtime_error(
            "GX2DirectCallDisplayList indexed packet was not captured correctly");
    }

    const auto& captured_pm4_immediate_draw = captured_state.draw_calls()[8U];
    if (!captured_pm4_immediate_draw.draw_call ||
        *captured_pm4_immediate_draw.draw_call !=
            affogato::cpu::espresso::Gx2DrawCall{
                gx2_primitive_mode_triangles,
                3U,
                true,
                9U,
                0U,
                0U,
                1U,
                0U,
                0U,
                0U} ||
        !captured_pm4_immediate_draw.draw_index_data ||
        !captured_pm4_immediate_draw.draw_index_data->is_valid ||
        captured_pm4_immediate_draw.draw_index_data->bytes !=
            std::vector<std::uint8_t>{
                0U, 0U, 0U, 0U,
                0U, 0U, 0U, 1U,
                0U, 0U, 0U, 2U} ||
        captured_pm4_immediate_draw.draw_vertex_data.size() != 1U ||
        !captured_pm4_immediate_draw.draw_vertex_data.front().is_valid)
    {
        throw std::runtime_error(
            "GX2DirectCallDisplayList immediate-index packet was not captured correctly");
    }

    if (!captured_state.draw_calls()[9U].draw_call ||
        *captured_state.draw_calls()[9U].draw_call !=
            affogato::cpu::espresso::Gx2DrawCall{
                gx2_primitive_mode_triangles,
                3U,
                false,
                0U,
                0U,
                0U,
                1U,
                0U,
                0U} ||
        ((captured_state.draw_fetch_shaders()[9U]->program_words[1U] >> 22U) &
         0x3FU) != 0x0AU)
    {
        throw std::runtime_error(
            "GX2 UNORM8x4 position format was not captured correctly");
    }

    affogato::gpu::vulkan::VulkanRenderer renderer(visible_window, true);
    affogato::gpu::vulkan::GuestDrawPolicy draw_policy =
        affogato::gpu::vulkan::GuestDrawPolicy::
            diagnostic_latte_vertex_position_base_zero;
    if (use_latte_spirv)
    {
        draw_policy = affogato::gpu::vulkan::GuestDrawPolicy::
            diagnostic_latte_vertex_spirv_position_base_zero;
    }
    if (use_latte_pixel_spirv)
    {
        draw_policy = affogato::gpu::vulkan::GuestDrawPolicy::
            diagnostic_latte_vertex_fragment_spirv_position_base_zero;
    }
    if (use_varying_passthrough)
    {
        draw_policy = affogato::gpu::vulkan::GuestDrawPolicy::
            diagnostic_latte_vertex_fragment_matched_move_base_zero;
    }
    renderer.consume_guest_commands(commands, draw_policy);
    const auto color_buffer_updates =
        renderer.take_guest_color_buffer_updates();
    if (color_buffer_updates.size() != 1U ||
        color_buffer_updates.front().status !=
            affogato::cpu::espresso::Gx2ColorBufferSnapshotStatus::captured)
    {
        throw std::runtime_error(
            "Vulkan guest draw did not return a color-buffer readback");
    }
    core.memory.write_bytes(
        color_buffer_image_address,
        color_buffer_updates.front().bytes);
    std::array<std::uint8_t, 4U> untouched_pixel{};
    core.memory.read_bytes(color_buffer_image_address, untouched_pixel);
    if (untouched_pixel != std::array<std::uint8_t, 4U>{0U, 0U, 180U, 255U})
    {
        throw std::runtime_error(
            "Vulkan guest color-buffer readback changed a pixel outside the diagnostic draws");
    }
    const std::uint32_t color_sample_x = color_buffer_width / 2U;
    const std::uint32_t color_sample_y = (color_buffer_height * 5U) / 12U;
    const std::size_t color_sample_offset =
        (static_cast<std::size_t>(color_sample_y) * color_buffer_width +
         color_sample_x) * 4U;
    std::array<std::uint8_t, 4U> diagnostic_pixel{};
    core.memory.read_bytes(
        color_buffer_image_address +
            static_cast<std::uint32_t>(color_sample_offset),
        diagnostic_pixel);
    bool has_diagnostic_color =
        diagnostic_pixel[0U] >= 200U && diagnostic_pixel[2U] <= 40U;
    if (use_latte_pixel_spirv)
    {
        has_diagnostic_color =
            diagnostic_pixel[0U] >= 200U &&
            diagnostic_pixel[1U] <= 40U &&
            diagnostic_pixel[2U] >= 200U;
    }
    if (shader == GuestTriangleShader::spirv_default_pixel_input)
    {
        has_diagnostic_color =
            diagnostic_pixel[0U] <= 40U &&
            diagnostic_pixel[1U] <= 40U &&
            diagnostic_pixel[2U] <= 40U;
    }
    if (use_varying_passthrough)
    {
        has_diagnostic_color = contains_expected_interpolated_pixels(
            color_buffer_updates.front().bytes,
            shader);
    }
    if (!has_diagnostic_color)
    {
        throw std::runtime_error(
            "Vulkan guest color-buffer pixel (" +
            std::to_string(color_sample_x) + ", " +
            std::to_string(color_sample_y) + ") was (" +
            std::to_string(diagnostic_pixel[0U]) + ", " +
            std::to_string(diagnostic_pixel[1U]) + ", " +
            std::to_string(diagnostic_pixel[2U]) + ", " +
            std::to_string(diagnostic_pixel[3U]) + ")");
    }

    invoke("GX2CopyColorBufferToScanBuffer", std::array<std::uint32_t, 2U>{
        color_buffer_descriptor_address,
        1U});
    const auto scanout_commands =
        affogato::cpu::espresso::take_gx2_commands(core.gx2);
    if (scanout_commands.size() != 1U ||
        scanout_commands.front().type !=
            affogato::cpu::espresso::Gx2CommandType::
                copy_color_buffer_to_scan_buffer)
    {
        throw std::runtime_error(
            "GX2 color-buffer copy did not produce the expected scanout command");
    }
    std::vector<std::uint8_t> scanout_bytes(color_buffer_size);
    core.memory.read_bytes(color_buffer_scanout_address, scanout_bytes);
    const auto scanout = affogato::gpu::stage_rgba8_scanout(
        color_buffer_width,
        color_buffer_height,
        color_buffer_width,
        scanout_bytes);
    if (!scanout)
    {
        throw std::runtime_error(
            "GX2 color-buffer copy did not produce readable RGBA8 scanout pixels");
    }
    const bool verify_shader_output =
        shader == GuestTriangleShader::spirv_constant_pixel_color ||
        shader == GuestTriangleShader::spirv_default_pixel_input ||
        shader == GuestTriangleShader::spirv_add_one_x ||
        shader == GuestTriangleShader::spirv_add_negative_one_x ||
        shader == GuestTriangleShader::spirv_multiply_zero ||
        shader == GuestTriangleShader::spirv_maximum_zero ||
        shader == GuestTriangleShader::spirv_minimum_zero ||
        shader == GuestTriangleShader::spirv_conditional_move ||
        shader == GuestTriangleShader::spirv_integer_conditional_move ||
        shader == GuestTriangleShader::spirv_mixed_conditional_move ||
        shader == GuestTriangleShader::spirv_integer_compare_clip_z ||
        shader == GuestTriangleShader::spirv_integer_alu_identity ||
        use_varying_passthrough;
    if (verify_shader_output)
    {
        renderer.render_frame();
        const auto& shader_frame = renderer.frame_readback();
        if (!shader_frame || shader_frame->width == 0U ||
            shader_frame->height == 0U)
        {
            throw std::runtime_error(
                "Vulkan Latte shader output readback is unavailable");
        }

        if (shader == GuestTriangleShader::spirv_mixed_conditional_move)
        {
            std::size_t red_pixel_count = 0U;
            for (std::size_t pixel_offset = 0U;
                 pixel_offset + 3U < shader_frame->pixels.size();
                 pixel_offset += 4U)
            {
                if (shader_frame->pixels[pixel_offset] >= 200U &&
                    shader_frame->pixels[pixel_offset + 2U] <= 40U)
                {
                    ++red_pixel_count;
                }
            }
            if (red_pixel_count == 0U)
            {
                throw std::runtime_error(
                    "Vulkan mixed OP2/OP3 shader produced no red readback pixels");
            }
        }
        else
        {
            std::uint32_t shader_sample_x = shader_frame->width / 2U;
            std::uint32_t shader_sample_y =
                (shader_frame->height * 5U) / 12U;
            bool expect_red = false;
            if (shader == GuestTriangleShader::spirv_add_one_x)
            {
                shader_sample_x = (shader_frame->width * 90U) / 100U;
                shader_sample_y = (shader_frame->height * 7U) / 12U;
                expect_red = true;
            }
            else if (shader == GuestTriangleShader::spirv_add_negative_one_x)
            {
                shader_sample_x = (shader_frame->width * 10U) / 100U;
                shader_sample_y = (shader_frame->height * 7U) / 12U;
                expect_red = true;
            }
            else if (shader == GuestTriangleShader::spirv_maximum_zero)
            {
                shader_sample_x = (shader_frame->width * 70U) / 100U;
                shader_sample_y = (shader_frame->height * 7U) / 12U;
                expect_red = true;
            }
            else if (shader == GuestTriangleShader::spirv_minimum_zero)
            {
                shader_sample_x = (shader_frame->width * 60U) / 100U;
                shader_sample_y = (shader_frame->height * 23U) / 40U;
            }
            else if (shader == GuestTriangleShader::spirv_conditional_move ||
                     shader == GuestTriangleShader::spirv_integer_conditional_move)
            {
                shader_sample_x = (shader_frame->width * 80U) / 100U;
                shader_sample_y = (shader_frame->height * 60U) / 100U;
                expect_red = true;
            }
            else if (shader ==
                         GuestTriangleShader::spirv_integer_compare_clip_z ||
                     shader == GuestTriangleShader::spirv_integer_alu_identity)
            {
                expect_red = true;
            }
            const std::size_t shader_sample_offset =
                (static_cast<std::size_t>(shader_sample_y) * shader_frame->width +
                 shader_sample_x) * 4U;
            if (shader_frame->pixels.size() < shader_sample_offset + 4U)
            {
                throw std::runtime_error(
                    "Vulkan Latte shader output readback is smaller than the expected sample");
            }
            const bool shader_sample_is_red =
                shader_frame->pixels[shader_sample_offset] >= 200U &&
                shader_frame->pixels[shader_sample_offset + 2U] <= 40U;
            const bool shader_sample_is_magenta =
                shader_frame->pixels[shader_sample_offset] >= 200U &&
                shader_frame->pixels[shader_sample_offset + 1U] <= 40U &&
                shader_frame->pixels[shader_sample_offset + 2U] >= 200U;
            const bool shader_sample_is_black =
                shader_frame->pixels[shader_sample_offset] <= 40U &&
                shader_frame->pixels[shader_sample_offset + 1U] <= 40U &&
                shader_frame->pixels[shader_sample_offset + 2U] <= 40U;
            bool shader_sample_matches =
                shader_sample_is_red == expect_red;
            if (shader == GuestTriangleShader::spirv_constant_pixel_color)
            {
                shader_sample_matches = shader_sample_is_magenta;
            }
            if (shader == GuestTriangleShader::spirv_default_pixel_input)
            {
                shader_sample_matches = shader_sample_is_black;
            }
            if (use_varying_passthrough)
            {
                shader_sample_matches = contains_expected_interpolated_pixels(
                    shader_frame->pixels,
                    shader);
            }
            if (!shader_sample_matches)
            {
                throw std::runtime_error(
                    "Vulkan Latte shader output pixel at (" +
                    std::to_string(shader_sample_x) + ", " +
                    std::to_string(shader_sample_y) + ") was (" +
                    std::to_string(shader_frame->pixels[shader_sample_offset]) +
                    ", " +
                    std::to_string(shader_frame->pixels[shader_sample_offset + 1U]) +
                    ", " +
                    std::to_string(shader_frame->pixels[shader_sample_offset + 2U]) +
                    ")");
            }
        }

        if (shader == GuestTriangleShader::spirv_conditional_move ||
            shader == GuestTriangleShader::spirv_integer_conditional_move)
        {
            const std::uint32_t outside_x =
                (shader_frame->width * 60U) / 100U;
            const std::uint32_t outside_y =
                (shader_frame->height * 80U) / 100U;
            const std::size_t outside_offset =
                (static_cast<std::size_t>(outside_y) * shader_frame->width +
                 outside_x) * 4U;
            if (shader_frame->pixels.size() < outside_offset + 4U)
            {
                throw std::runtime_error(
                    "Vulkan OP3 shader output readback is smaller than the expected outside sample");
            }
            const bool outside_sample_is_red =
                shader_frame->pixels[outside_offset] >= 200U &&
                shader_frame->pixels[outside_offset + 2U] <= 40U;
            if (outside_sample_is_red)
            {
                throw std::runtime_error(
                    "Vulkan OP3 conditional-move output covered an outside sample");
            }
        }
    }
    renderer.consume_guest_commands(
        scanout_commands,
        affogato::gpu::vulkan::GuestDrawPolicy::capture_only);
    renderer.clear_guest_diagnostic_draws();
    if (!renderer.set_guest_scanout(*scanout))
    {
        throw std::runtime_error(
            "Vulkan surface does not support the guest color-buffer scanout");
    }
    renderer.set_guest_clear_color({0.08F, 0.75F, 0.2F, 1.0F});
    renderer.render_frame();
    const auto& rendered_frame = renderer.frame_readback();
    if (!rendered_frame || rendered_frame->width == 0U ||
        rendered_frame->height == 0U)
    {
        throw std::runtime_error("Vulkan guest triangle frame readback is unavailable");
    }
    std::uint32_t sample_x = rendered_frame->width / 2U;
    std::uint32_t sample_y = (rendered_frame->height * 5U) / 12U;
    const std::size_t sample_offset =
        (static_cast<std::size_t>(sample_y) * rendered_frame->width + sample_x) * 4U;
    if (rendered_frame->pixels.size() < sample_offset + 4U)
    {
        throw std::runtime_error(
            "Vulkan guest triangle readback is smaller than the expected shader sample");
    }
    bool presented_diagnostic_color =
        rendered_frame->pixels[sample_offset] >= 200U &&
        rendered_frame->pixels[sample_offset + 2U] <= 40U;
    if (use_latte_pixel_spirv)
    {
        presented_diagnostic_color =
            rendered_frame->pixels[sample_offset] >= 200U &&
            rendered_frame->pixels[sample_offset + 1U] <= 40U &&
            rendered_frame->pixels[sample_offset + 2U] >= 200U;
    }
    if (shader == GuestTriangleShader::spirv_default_pixel_input)
    {
        presented_diagnostic_color =
            rendered_frame->pixels[sample_offset] <= 40U &&
            rendered_frame->pixels[sample_offset + 1U] <= 40U &&
            rendered_frame->pixels[sample_offset + 2U] <= 40U;
    }
    if (use_varying_passthrough)
    {
        presented_diagnostic_color = contains_expected_interpolated_pixels(
            rendered_frame->pixels,
            shader);
    }
    if (!presented_diagnostic_color)
    {
        throw std::runtime_error(
            "Vulkan guest triangle readback did not match the expected shader result");
    }
    while (visible_window && !renderer.should_close())
    {
        renderer.poll_events();
        renderer.render_frame();
    }
    const char* shader_subset_description =
        "the supported CPU MOV/export shader subset";
    if (shader == GuestTriangleShader::spirv_add_zero)
    {
        shader_subset_description =
            "the bounded MOV/ADD-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_add_one_x)
    {
        shader_subset_description =
            "the bounded MOV/ADD-offset-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_add_negative_one_x)
    {
        shader_subset_description =
            "the bounded MOV/ADD-negated-offset-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_multiply_one)
    {
        shader_subset_description =
            "the bounded MOV/MUL-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_multiply_zero)
    {
        shader_subset_description =
            "the bounded MOV/MUL-collapse-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_maximum_self)
    {
        shader_subset_description =
            "the bounded MOV/MAX_DX10-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_minimum_self)
    {
        shader_subset_description =
            "the bounded MOV/MIN_DX10-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_maximum_zero)
    {
        shader_subset_description =
            "the bounded MOV/MAX_DX10-clamp-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_minimum_zero)
    {
        shader_subset_description =
            "the bounded MOV/MIN_DX10-clamp-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_conditional_move)
    {
        shader_subset_description =
            "the bounded OP3 floating conditional-move-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_integer_conditional_move)
    {
        shader_subset_description =
            "the bounded OP3 signed-integer conditional-move-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_mixed_conditional_move)
    {
        shader_subset_description =
            "the bounded mixed OP2 MOV/OP3 conditional-move-to-position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_integer_compare_clip_z)
    {
        shader_subset_description =
            "the bounded OP2 SETGT_INT position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_integer_alu_identity)
    {
        shader_subset_description =
            "the bounded OP2 integer ALU position SPIR-V vertex subset";
    }
    if (shader == GuestTriangleShader::spirv_constant_pixel_color)
    {
        shader_subset_description =
            "the bounded MOV/inline-constant-to-MRT0 SPIR-V fragment subset";
    }
    if (shader == GuestTriangleShader::spirv_default_pixel_input)
    {
        shader_subset_description =
            "the bounded MOV/default-input-to-MRT0 SPIR-V fragment subset";
    }
    if (use_varying_passthrough)
    {
        shader_subset_description =
            "the bounded guest vertex position/parameter and pixel MOV subsets";
    }
    const char* fragment_shader_status = use_varying_passthrough
        ? "the captured guest fragment MOV subset executed on the matched "
          "varying; other guest pixel operations remain unsupported.\n"
        : (use_latte_pixel_spirv
        ? "the captured guest pixel shader subset was translated; general Latte "
          "shader translation and WWHD graphics remain unsupported.\n"
        : "the fragment shader remains diagnostic; general Latte shader "
          "translation and WWHD graphics remain unsupported.\n");
    std::cout << "GX2 float32/float16/UNORM8x4 triangle-list, indexed, strip, and fan "
              << "geometry, "
              << "including unpredicated PM4 auto-index, indexed, and immediate-index "
              << "draws used captured guest vertex input and "
              << shader_subset_description
              << "; Vulkan readback and guest-memory writeback verified the "
                 "diagnostic color-buffer path on "
              << renderer.device_name()
              << ". " << fragment_shader_status;
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
            if (std::string_view(argv[1]) == "--gx2-guest-scanout-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_scanout_smoke_test();
            }
            if (std::string_view(argv[1]) == "--gx2-guest-scanout-display-test" &&
                argc == 2)
            {
                return run_gx2_guest_scanout_smoke_test(true);
            }
            if (std::string_view(argv[1]) == "--gx2-guest-triangle-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test();
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_add_zero);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-multiply-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_multiply_one);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-add-one-x-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_add_one_x);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-add-negative-one-x-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_add_negative_one_x);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-multiply-zero-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_multiply_zero);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-maximum-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_maximum_self);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-minimum-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_minimum_self);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-maximum-zero-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_maximum_zero);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-minimum-zero-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_minimum_zero);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-op3-conditional-move-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_conditional_move);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-op3-integer-conditional-move-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_integer_conditional_move);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-mixed-alu-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_mixed_conditional_move);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-integer-compare-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_integer_compare_clip_z);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-integer-alu-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_integer_alu_identity);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-pixel-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_constant_pixel_color);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-pixel-default-spirv-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_default_pixel_input);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-move-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_move);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-add-half-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_add_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-multiply-half-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_multiply_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-maximum-half-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_maximum_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-minimum-half-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_minimum_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-add-negative-half-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_add_negative_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-xor-zero-smoke-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_xor_zero);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-signed-max-readback-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::
                        spirv_matched_varying_maximum_signed_integer_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-unsigned-min-readback-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::
                        spirv_matched_varying_minimum_unsigned_integer_half);
            }
            if (std::string_view(argv[1]) ==
                    "--gx2-guest-triangle-matched-varying-default-alpha-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(
                    false,
                    GuestTriangleShader::spirv_matched_varying_default_alpha);
            }
            if (std::string_view(argv[1]) == "--gx2-guest-triangle-display-test" &&
                argc == 2)
            {
                return run_gx2_guest_triangle_test(true);
            }
            std::cerr << "Unknown option: " << argv[1] << '\n'
                      << "Usage: AffogatoFrontend.exe [--run-rpx <file.rpx> "
                         "[--recompiler] [--limit <instructions>]]\n"
                         "       AffogatoFrontend.exe --vulkan-texture-upload-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-scanout-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-scanout-display-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-multiply-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-add-one-x-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-add-negative-one-x-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-multiply-zero-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-maximum-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-minimum-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-maximum-zero-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-minimum-zero-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-op3-conditional-move-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-op3-integer-conditional-move-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-mixed-alu-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-integer-compare-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-integer-alu-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-pixel-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-pixel-default-spirv-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-matched-varying-move-smoke-test\n"
                         "       AffogatoFrontend.exe --gx2-guest-triangle-display-test\n";
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
