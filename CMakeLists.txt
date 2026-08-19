cmake_minimum_required(VERSION 3.20)

project(miniLLM_V1 VERSION 1.0.0 LANGUAGES C)

# 本项目故意只使用 ISO C11。关闭编译器扩展后，GCC 会使用 -std=c11，
# 而不是默认的 -std=gnu11；这样能及时发现不属于标准 C 的写法。
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(minillm STATIC
        src/core/memory.c
        src/core/random.c
        src/core/model.c
        src/core/cache.c
        src/layers/linear.c
        src/layers/rmsnorm.c
        src/layers/attention.c
        src/model/forward.c
        src/model/backward.c
        src/training/optimizer.c
        src/inference/generate.c
        src/io/checkpoint.c
)
target_include_directories(minillm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(minillm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
    target_compile_options(minillm PRIVATE /W4)
else ()
    target_compile_options(minillm PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif ()

# 在大多数 Unix 系统上，标准数学函数位于单独的 libm 中。
# Windows/MSVC/MinGW 通常不需要显式链接它。
if (UNIX)
    target_link_libraries(minillm PUBLIC m)
endif ()

add_executable(minillm_cli src/main.c)
target_link_libraries(minillm_cli PRIVATE minillm)
set_target_properties(minillm_cli PROPERTIES OUTPUT_NAME minillm)

include(CTest)
if (BUILD_TESTING)
    add_executable(minillm_tests tests/test_minillm.c)
    target_link_libraries(minillm_tests PRIVATE minillm)
    add_test(NAME minillm_unit_tests COMMAND minillm_tests)
endif ()
