#include "emulator.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_filesystem.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
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

std::string format_result(const affogato::EmulatorRunResult& result)
{
    using affogato::cpu::espresso::StopReason;
    std::ostringstream text;
    text << "Loaded entry point: 0x" << std::hex << std::uppercase << result.image.entry_point
         << "\nSections loaded: " << std::dec << result.image.loaded_sections
         << "\nExecuted instructions: " << result.execution.steps
         << "\nStop CIA: 0x" << std::hex << std::uppercase << result.execution.cia
         << "\nGPR3: " << std::dec << result.gpr3;
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
    case StopReason::memory_fault:
        text << "\nStopped: memory fault";
        if (!result.execution.detail.empty()) text << "\nDetail: " << result.execution.detail;
        break;
    }
    if (result.gpr3 == 42)
    {
        text << "\nReturned value observed in r3: 42";
    }
    return text.str();
}

struct WorkerResult
{
    std::mutex mutex;
    std::string message;
    std::atomic_bool running{};
};

void launch_game(const fs::path& path, WorkerResult& worker, std::jthread& thread)
{
    if (worker.running.exchange(true))
    {
        return;
    }
    if (thread.joinable()) thread.join();
    {
        std::lock_guard lock(worker.mutex);
        worker.message = "Loading " + path_utf8(path) + " ...";
    }
    thread = std::jthread([path, &worker]
    {
        std::string message;
        try
        {
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
            if (!bytes.empty() && !input.read(reinterpret_cast<char*>(bytes.data()), length))
            {
                throw std::runtime_error("Could not read the complete RPX file.");
            }
            affogato::Emulator emulator;
            (void)emulator.load_rpx(bytes);
            message = format_result(emulator.run());
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
    bool done = false;
    while (!done)
    {
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
                        launch_game(games[i].rpx_path, worker, launch_thread);
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
                            launch_game(games[i].rpx_path, worker, launch_thread);
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
            if (worker.running) ImGui::BeginDisabled();
            if (ImGui::Button("Launch")) launch_game(games[*selected_game].rpx_path, worker, launch_thread);
            if (worker.running) ImGui::EndDisabled();
        }
        if (worker.running) ImGui::SameLine();
        if (worker.running) ImGui::Text("Running RPX...");
        ImGui::SeparatorText("Launch diagnostics");
        std::string message;
        {
            std::lock_guard lock(worker.mutex);
            message = worker.message;
        }
        ImGui::BeginChild("Diagnostics", ImVec2(0, 0), ImGuiChildFlags_Borders);
        ImGui::TextWrapped("%s", message.empty() ? "Select and launch an RPX to see its result." : message.c_str());
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

int main()
{
    try
    {
        return run_frontend();
    }
    catch (const std::exception& error)
    {
        report_fatal_error(error.what());
    }
    catch (...)
    {
        report_fatal_error("Unknown non-standard exception.");
    }
    return 1;
}
