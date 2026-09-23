cmake_minimum_required(VERSION 3.13)

project(minimal_wut_rpx LANGUAGES C)

set(DEVKITPRO "C:/devkitPro" CACHE PATH "devkitPro installation root")
set(WUT_ROOT "${DEVKITPRO}/wut")

add_executable(minimal_wut_rpx main.c)
set_target_properties(minimal_wut_rpx PROPERTIES SUFFIX ".elf")

target_include_directories(minimal_wut_rpx PRIVATE "${WUT_ROOT}/include")
target_compile_options(minimal_wut_rpx PRIVATE
    -mcpu=750
    -meabi
    -mhard-float
    -ffunction-sections
    -fdata-sections
    -DESPRESSO
    -D__WIIU__
    -D__WUT__
)
target_link_options(minimal_wut_rpx PRIVATE "-specs=${WUT_ROOT}/share/wut.specs")
target_link_libraries(minimal_wut_rpx PRIVATE "${WUT_ROOT}/lib/libwut.a" m)

add_custom_command(TARGET minimal_wut_rpx POST_BUILD
    COMMAND "${DEVKITPRO}/tools/bin/elf2rpl.exe"
        "$<TARGET_FILE:minimal_wut_rpx>"
        "${CMAKE_CURRENT_BINARY_DIR}/minimal_wut_rpx.rpx"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/minimal_wut_rpx.rpx"
    COMMENT "Converting the devkitPro/WUT executable to RPX format"
    VERBATIM
)

