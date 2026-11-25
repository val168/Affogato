cmake_minimum_required(VERSION 3.25)

project(Affogato LANGUAGES CXX)

# GLFW is built from source with the same MinGW toolchain as Affogato. It is
# used by the graphics frontend for Vulkan-capable windows and surfaces.
enable_language(C)

include(CTest)
include(FetchContent)

option(AFFOGATO_ENABLE_VULKAN_INTEGRATION_TESTS
    "Run Vulkan tests that require an available graphics device and window system"
    OFF
)

find_package(Vulkan 1.3 REQUIRED)

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
    src/cpu/espresso/nn_act_hle.cpp
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

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.5.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

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
    src/gpu/vulkan/vulkan_renderer.cpp
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
target_link_libraries(affogato_frontend PRIVATE
    affogato_cpu
    SDL3::SDL3-static
    glfw
    Vulkan::Vulkan
)
set_target_properties(affogato_frontend PROPERTIES
    OUTPUT_NAME "AffogatoFrontend"
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Keep Debug symbols and assertions, but optimize the CPU execution library.
# Without this, even the recompiler's C++ dispatch and interpreter fallbacks
# run at -O0 and dominate time spent executing generated guest code.
if(MSVC)
    target_compile_options(affogato_cpu PRIVATE
        /W4
        /permissive-
        $<$<CONFIG:Debug>:/O2>
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
        $<$<CONFIG:Debug>:-O2>
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

    add_executable(affogato_latte_spirv_tests
        tests/latte_spirv_test.cpp
    )
    target_include_directories(affogato_latte_spirv_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(affogato_latte_spirv_tests PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    if(MSVC)
        target_compile_options(affogato_latte_spirv_tests PRIVATE
            /W4
            /permissive-
            /UNDEBUG
        )
    else()
        target_compile_options(affogato_latte_spirv_tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
        )
    endif()

    set(AFFOGATO_LATTE_SPIRV_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset.spv")
    set(AFFOGATO_LATTE_SPIRV_MODIFIERS_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_modifiers.spv")
    set(AFFOGATO_LATTE_SPIRV_ADD_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_add.spv")
    set(AFFOGATO_LATTE_SPIRV_MULTIPLY_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_multiply.spv")
    set(AFFOGATO_LATTE_SPIRV_MAXIMUM_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_maximum.spv")
    set(AFFOGATO_LATTE_SPIRV_MINIMUM_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_minimum.spv")
    set(AFFOGATO_LATTE_SPIRV_CONDITIONAL_MOVE_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_conditional_move.spv")
    set(AFFOGATO_LATTE_SPIRV_INTEGER_CONDITIONAL_MOVE_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_integer_conditional_move.spv")
    set(AFFOGATO_LATTE_SPIRV_MIXED_ALU_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_mixed_alu.spv")
    set(AFFOGATO_LATTE_SPIRV_INTEGER_ALU_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_subset_integer_alu.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_constant_color.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_DEFAULT_INPUT_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_default_input.spv")
    set(AFFOGATO_LATTE_SPIRV_VERTEX_PARAMETER_OUTPUT_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_vertex_parameter_output.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_VARYING_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_varying_passthrough.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MOVE_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_move.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_ADD_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_add.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MULTIPLY_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_multiply.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MAXIMUM_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_maximum.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MINIMUM_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_minimum.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_ARITHMETIC_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_integer_arithmetic.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_LOGIC_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_integer_logic.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_COMPARISON_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_integer_comparison.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INPUTS_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_matched_inputs.spv")
    set(AFFOGATO_LATTE_SPIRV_FRAGMENT_MULTIPLE_VARYINGS_TEST_MODULE
        "${CMAKE_CURRENT_BINARY_DIR}/latte_fragment_multiple_varyings.spv")
    add_test(NAME affogato_latte_spirv_tests
        COMMAND affogato_latte_spirv_tests
            --write-modules
            "${AFFOGATO_LATTE_SPIRV_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_MODIFIERS_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_ADD_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_MULTIPLY_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_MAXIMUM_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_MINIMUM_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_CONDITIONAL_MOVE_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_INTEGER_CONDITIONAL_MOVE_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_MIXED_ALU_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_INTEGER_ALU_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_DEFAULT_INPUT_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_VERTEX_PARAMETER_OUTPUT_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_VARYING_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MOVE_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_ADD_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MULTIPLY_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MAXIMUM_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MINIMUM_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_ARITHMETIC_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_LOGIC_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_COMPARISON_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INPUTS_TEST_MODULE}"
            "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MULTIPLE_VARYINGS_TEST_MODULE}")
    set_tests_properties(affogato_latte_spirv_tests PROPERTIES
        FIXTURES_SETUP affogato_latte_spirv_module
    )

    find_program(AFFOGATO_SPIRV_VAL_EXECUTABLE
        NAMES spirv-val
        HINTS "$ENV{VULKAN_SDK}/Bin"
    )
    if(AFFOGATO_SPIRV_VAL_EXECUTABLE)
        add_test(NAME affogato_latte_spirv_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_vertex_parameter_output_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_VERTEX_PARAMETER_OUTPUT_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_vertex_parameter_output_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_varying_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_VARYING_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_varying_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_matched_move_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MOVE_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_move_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_matched_add_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_ADD_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_add_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_matched_multiply_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MULTIPLY_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_multiply_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_matched_maximum_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MAXIMUM_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_maximum_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_matched_minimum_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_MINIMUM_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_minimum_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(
            NAME affogato_latte_spirv_fragment_matched_integer_arithmetic_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_ARITHMETIC_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_integer_arithmetic_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(
            NAME affogato_latte_spirv_fragment_matched_integer_logic_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_LOGIC_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_integer_logic_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(
            NAME affogato_latte_spirv_fragment_matched_integer_comparison_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INTEGER_COMPARISON_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_integer_comparison_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(
            NAME affogato_latte_spirv_fragment_matched_inputs_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MATCHED_INPUTS_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_matched_inputs_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(
            NAME affogato_latte_spirv_fragment_multiple_varyings_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_MULTIPLE_VARYINGS_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_multiple_varyings_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_modifier_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_MODIFIERS_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_modifier_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_integer_alu_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_INTEGER_ALU_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_integer_alu_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_fragment_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_fragment_default_input_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_FRAGMENT_DEFAULT_INPUT_TEST_MODULE}")
        set_tests_properties(
            affogato_latte_spirv_fragment_default_input_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_add_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_ADD_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_add_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_multiply_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_MULTIPLY_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_multiply_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_maximum_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_MAXIMUM_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_maximum_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_minimum_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_MINIMUM_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_minimum_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_conditional_move_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_CONDITIONAL_MOVE_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_conditional_move_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_integer_conditional_move_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_INTEGER_CONDITIONAL_MOVE_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_integer_conditional_move_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
        add_test(NAME affogato_latte_spirv_mixed_alu_validation
            COMMAND "${AFFOGATO_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.0
                "${AFFOGATO_LATTE_SPIRV_MIXED_ALU_TEST_MODULE}")
        set_tests_properties(affogato_latte_spirv_mixed_alu_validation PROPERTIES
            FIXTURES_REQUIRED affogato_latte_spirv_module
        )
    endif()

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

    if(AFFOGATO_ENABLE_VULKAN_INTEGRATION_TESTS)
        add_test(
            NAME affogato_vulkan_guest_triangle_readback
            COMMAND affogato_frontend --gx2-guest-triangle-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_multiply_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-multiply-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_add_offset_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-add-one-x-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_add_negated_offset_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-add-negative-one-x-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_multiply_zero_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-multiply-zero-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_maximum_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-maximum-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_minimum_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-minimum-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_maximum_zero_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-maximum-zero-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_minimum_zero_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-minimum-zero-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_op3_conditional_move_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-op3-conditional-move-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_op3_integer_conditional_move_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-op3-integer-conditional-move-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_mixed_alu_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-mixed-alu-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_integer_compare_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-integer-compare-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_integer_alu_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-integer-alu-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_pixel_shader_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-pixel-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_pixel_default_input_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-pixel-default-spirv-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_move_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-move-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_add_half_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-add-half-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_multiply_half_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-multiply-half-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_maximum_half_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-maximum-half-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_minimum_half_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-minimum-half-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_add_negative_half_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-add-negative-half-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_xor_zero_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-xor-zero-smoke-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_signed_max_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-signed-max-readback-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_unsigned_min_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-unsigned-min-readback-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_varying_default_input_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-varying-default-alpha-test
        )
        add_test(
            NAME affogato_vulkan_guest_matched_two_varyings_readback
            COMMAND affogato_frontend
                --gx2-guest-triangle-matched-two-varyings-test
        )
    endif()

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
