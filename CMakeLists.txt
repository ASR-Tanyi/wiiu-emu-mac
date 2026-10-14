# wiiu-emu — root build.
#
# Target platform is macOS on Apple silicon (see docs/decisions/ADR-002).
# Platform-independent libraries also build on Linux so the core can be
# compiled and tested in CI without a Mac; anything touching Metal, CoreAudio
# or the Game Controller framework is guarded by WIIU_PLATFORM_APPLE.

cmake_minimum_required(VERSION 3.24)

# Must precede project(): these seed the Apple toolchain.
if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS architectures")
endif()
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    # macOS 13 Ventura — the Metal 3 baseline (plan Phase 7).
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
endif()

project(wiiu-emu
    VERSION 0.1.0
    DESCRIPTION "A Wii U emulator for macOS"
    LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
include(CMakeDependentOption)

option(WIIU_BUILD_TESTS   "Build the test suite"                     ON)
option(WIIU_WERROR        "Treat compiler warnings as errors"        ON)
option(WIIU_ENABLE_LTO    "Link-time optimisation in Release (P38)"  ON)
option(WIIU_SANITIZE      "Build with ASan + UBSan"                  OFF)
option(WIIU_EXPORT_COMPILE_COMMANDS "Emit compile_commands.json"     ON)

if(WIIU_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# --------------------------------------------------------------------------
# Platform gate
# --------------------------------------------------------------------------
if(APPLE)
    set(WIIU_PLATFORM_APPLE ON)
    enable_language(OBJCXX)   # the CAMetalLayer platform layer (Phase 1.4)

    if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        message(FATAL_ERROR
            "wiiu-emu targets Apple silicon only (CMAKE_OSX_ARCHITECTURES=arm64).\n"
            "Mesh shaders require GPU family Apple7, which no Intel Mac provides — "
            "see docs/decisions/ADR-002-apple-silicon-only.md.\n"
            "Requested: ${CMAKE_OSX_ARCHITECTURES}")
    endif()
else()
    set(WIIU_PLATFORM_APPLE OFF)
    message(STATUS
        "wiiu-emu: non-Apple host — building the platform-independent core only. "
        "The emulator itself requires macOS 13+ on Apple silicon.")
endif()

# --------------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# P38 — free speed, on from day one. Guest memory is type-punned all over the
# place (Phase 3), so strict aliasing is off globally and deliberately.
add_compile_options(-fno-strict-aliasing)

# The host compiler does not get to decide how guest arithmetic rounds.
#
# Clang defaults to contracting `a * b + c` into a fused multiply-add wherever
# the target has one. On arm64 it always does; on x86-64 without -mfma it
# never does. That is the same source producing two different answers on two
# hosts -- which for an emulator is not an optimisation, it is a correctness
# bug that only appears on one machine. Espresso's `fmadd` really is fused and
# its `fmul`/`fadd` really are not, so the distinction is modelled explicitly
# with std::fma (Phase 4.6) and the compiler must leave it alone.
#
# It also protects what 10.4 and 12.7 are built on: a replay only reproduces a
# run, and a JIT only matches the interpreter bit-for-bit, if the arithmetic
# does not depend on which machine compiled it. Found by the one test that
# compared a fused result against a deliberately unfused one and watched the
# compiler fuse both.
add_compile_options(-ffp-contract=off)

if(WIIU_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT WIIU_IPO_OK OUTPUT WIIU_IPO_MSG)
    if(WIIU_IPO_OK)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    else()
        message(STATUS "wiiu-emu: LTO unavailable — ${WIIU_IPO_MSG}")
    endif()
endif()

if(WIIU_PLATFORM_APPLE)
    # P38 — tune for Apple silicon. apple-m1 is the baseline; newer chips are
    # backward compatible with it.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-mcpu=apple-m1>)
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

include(CompilerWarnings)
include(Dependencies)
include(AddWiiuLibrary)

# --------------------------------------------------------------------------
# Subsystems
#
# Each is enabled by the plan phase that first gives it content, so the build
# never contains placeholder code. Directories exist from the start; the
# add_subdirectory() line is uncommented when the phase lands.
# --------------------------------------------------------------------------
add_subdirectory(src/common)                 # Phase 2
add_subdirectory(src/memory)                 # Phase 3
add_subdirectory(src/cpu)                    # Phase 4
add_subdirectory(src/loader)                 # Phase 5
add_subdirectory(src/hle)                    # Phase 6
add_subdirectory(src/gpu)                    # Phase 7-8
# add_subdirectory(src/audio)                # Phase 9
# add_subdirectory(src/input)                # Phase 9
add_subdirectory(src/frontend)               # Phase 1

add_subdirectory(tools)                      # ppcdis; rpl-dump in Phase 5.10

if(WIIU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------
message(STATUS "")
message(STATUS "wiiu-emu ${PROJECT_VERSION}")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler ........... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(WIIU_PLATFORM_APPLE)
    message(STATUS "  apple target ....... ${CMAKE_OSX_ARCHITECTURES} / macOS ${CMAKE_OSX_DEPLOYMENT_TARGET}")
else()
    message(STATUS "  apple target ....... n/a (core-only build)")
endif()
message(STATUS "  tests .............. ${WIIU_BUILD_TESTS}")
message(STATUS "  warnings as errors . ${WIIU_WERROR}")
message(STATUS "  sanitizers ......... ${WIIU_SANITIZE}")
message(STATUS "")
