cmake_minimum_required(VERSION 3.25)

project(Affogato LANGUAGES CXX)

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

add_executable(affogato
    src/main.cpp
)

set_target_properties(affogato PROPERTIES
    OUTPUT_NAME "Affogato"
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if(MSVC)
    target_compile_options(affogato PRIVATE
        /W4
        /permissive-
    )
else()
    target_compile_options(affogato PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()