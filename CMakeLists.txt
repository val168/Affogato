cmake_minimum_required(VERSION 3.25)

project(Affogato LANGUAGES CXX)

include(CTest)
include(FetchContent)

find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        zlib
        URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(zlib)
    set(AFFOGATO_ZLIB_TARGET zlibstatic)
else()
    set(AFFOGATO_ZLIB_TARGET ZLIB::ZLIB)
endif()

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

add_library(affogato_cpu STATIC
    src/cpu/espresso/decoder.cpp
    src/cpu/espresso/elf_loader.cpp
    src/cpu/espresso/rpx_loader.cpp
    src/cpu/espresso/interpreter.cpp
)

target_include_directories(affogato_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(affogato_cpu PRIVATE ${AFFOGATO_ZLIB_TARGET})

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

if(MSVC)
    target_compile_options(affogato_cpu PRIVATE
        /W4
        /permissive-
    )
    target_compile_options(affogato PRIVATE
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
endif()

if(BUILD_TESTING)
    add_executable(affogato_cpu_tests
        tests/cpu_test.cpp
    )

    target_include_directories(affogato_cpu_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(affogato_cpu_tests PRIVATE affogato_cpu)
    target_link_libraries(affogato_cpu_tests PRIVATE ${AFFOGATO_ZLIB_TARGET})

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
