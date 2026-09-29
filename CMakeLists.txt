cmake_minimum_required(VERSION 3.25)

project(Affogato LANGUAGES CXX)

include(CTest)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Affogato currently requires a 64-bit host.")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(
            FATAL_ERROR
            "Affogato requires GCC 14 or newer. "
            "Detected GCC ${CMAKE_CXX_COMPILER_VERSION}."
        )
    endif()
endif()

set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/zlib)

add_library(affogato_cpu STATIC
    src/cpu/espresso/cafe_os_hle.cpp
    src/cpu/espresso/decoder.cpp
    src/cpu/espresso/elf_loader.cpp
    src/cpu/espresso/hle_dispatcher.cpp
    src/cpu/espresso/guest_mutex.cpp
    src/cpu/espresso/rpx_loader.cpp
    src/cpu/espresso/interpreter.cpp
    src/emulator.cpp
)

target_include_directories(affogato_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(affogato_cpu PRIVATE ZLIB::ZLIBSTATIC)

set_target_properties(affogato_cpu PROPERTIES
    OUTPUT_NAME "Affogato"
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

add_executable(affogato
    src/main.cpp
)

target_link_libraries(affogato PRIVATE affogato_cpu)

set_target_properties(affogato PROPERTIES
    OUTPUT_NAME "Affogato"
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# The desktop frontend is deliberately a separate executable: it talks to the
# public Emulator session API and does not add UI dependencies to the core.
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/SDL EXCLUDE_FROM_ALL)

add_executable(affogato_frontend
    src/frontend/main.cpp
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_sdl3.cpp
    third_party/imgui/backends/imgui_impl_sdlrenderer3.cpp
)
target_include_directories(affogato_frontend PRIVATE
    third_party/imgui
    third_party/imgui/backends
)
target_link_libraries(affogato_frontend PRIVATE affogato_cpu SDL3::SDL3-static)
set_target_properties(affogato_frontend PROPERTIES
    OUTPUT_NAME "AffogatoFrontend"
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if(MSVC)
    target_compile_options(affogato_cpu PRIVATE
        /W4
        /permissive-
    )
    target_compile_options(affogato PRIVATE
        /W4
        /permissive-
    )
    target_compile_options(affogato_frontend PRIVATE
        /W4
        /permissive-
    )
else()
    target_compile_options(affogato_cpu PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    target_compile_options(affogato PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
    target_compile_options(affogato_frontend PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

if(BUILD_TESTING)
    add_executable(affogato_cpu_tests
        tests/cpu_test.cpp
    )

    target_include_directories(affogato_cpu_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(affogato_cpu_tests PRIVATE affogato_cpu)
    target_link_libraries(affogato_cpu_tests PRIVATE ZLIB::ZLIBSTATIC)

    set_target_properties(affogato_cpu_tests PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )

    if(MSVC)
        target_compile_options(affogato_cpu_tests PRIVATE
            /W4
            /permissive-
        )
    else()
        target_compile_options(affogato_cpu_tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(NAME affogato_cpu_tests COMMAND affogato_cpu_tests)
endif()
