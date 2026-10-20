cmake_minimum_required(VERSION 3.25)

project(Affogato LANGUAGES CXX)

include(CTest)
include(FetchContent)

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
    src/cpu/espresso/gx2_hle.cpp
    src/cpu/espresso/rpx_loader.cpp
    src/cpu/espresso/interpreter.cpp
    src/cpu/espresso/recompiler.cpp
    src/cpu/espresso/nn_olv_hle.cpp
    src/cpu/espresso/nn_save_hle.cpp
    src/emulator.cpp
)

# Pin the JIT backend so local builds and CI use the same assembler/compiler
# implementation. The interpreter remains available on every build.
set(ASMJIT_NO_INSTALL ON CACHE BOOL "" FORCE)
set(ASMJIT_STATIC ON CACHE BOOL "" FORCE)
FetchContent_Declare(asmjit
    GIT_REPOSITORY https://github.com/asmjit/asmjit.git
    GIT_TAG 0464e1c8f1f43c861a0c73d4e59437fff255cdd9
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(asmjit)

target_include_directories(affogato_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(affogato_cpu PRIVATE ZLIB::ZLIBSTATIC asmjit::asmjit)

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
            /UNDEBUG
        )
    else()
        target_compile_options(affogato_cpu_tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
        )
    endif()

    add_test(NAME affogato_cpu_tests COMMAND affogato_cpu_tests)

    add_executable(affogato_recompiler_tests
        tests/recompiler_test.cpp
    )
    target_link_libraries(affogato_recompiler_tests PRIVATE affogato_cpu)
    set_target_properties(affogato_recompiler_tests PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    if(MSVC)
        target_compile_options(affogato_recompiler_tests PRIVATE /W4 /permissive- /UNDEBUG)
    else()
        target_compile_options(affogato_recompiler_tests PRIVATE
            -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()
    add_test(NAME affogato_recompiler_tests COMMAND affogato_recompiler_tests)

    add_executable(affogato_recompiler_benchmark
        tests/recompiler_benchmark.cpp
    )
    target_link_libraries(affogato_recompiler_benchmark PRIVATE affogato_cpu)
    set_target_properties(affogato_recompiler_benchmark PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
endif()
