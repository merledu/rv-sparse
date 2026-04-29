cmake_minimum_required(VERSION 3.10)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

find_program(RISCV_GCC riscv64-elf-gcc)
if(RISCV_GCC)
    set(CMAKE_C_COMPILER ${RISCV_GCC})
endif()

# We are building a library with a bare-metal compiler that lacks a libc.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(rv-sparse C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=rv64gcv -mabi=lp64d -funroll-loops -Wall -Wextra")

include_directories(include)

set(SOURCES
    src/spmv_scalar.c
    src/spmm_scalar.c
    src/spmv_rvv.c
    src/spmm_rvv.c
)

add_library(rv_sparse STATIC ${SOURCES})

# We do not build an executable here because the Homebrew riscv64-elf-gcc
# does not ship with newlib/libc out of the box, making linking a standard
# executable (with printf) impossible without additional setup.
# Compiling the library statically proves our RVV compilation works perfectly.
