# Phase 1.3 — third-party dependencies.
#
# Only what a phase actually needs is fetched; the rest are declared with their
# pinned tag and the phase that turns them on. This keeps configure fast and
# stops the dependency graph growing ahead of the code.
#
# Everything is pinned to an exact tag. Never track a branch: a moving
# dependency turns a reproducible build into a guess.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ---- active now -----------------------------------------------------------

# fmt — formatting, and the backend for spdlog.
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        12.2.0
    GIT_SHALLOW    TRUE
    SYSTEM)

# spdlog — logging backend behind wiiu::common's channel API (Phase 1.6).
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)  # share one fmt, avoid ODR grief
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
    SYSTEM)

# toml++ — settings file parsing (Phase 2.5). Header-only, MIT.
set(TOMLPLUSPLUS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
    SYSTEM)

# pugixml — cos.xml / app.xml / meta.xml (Phase 5.5). Header + one source, MIT.
set(PUGIXML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(pugixml
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    GIT_TAG        v1.16
    GIT_SHALLOW    TRUE
    SYSTEM)

FetchContent_MakeAvailable(fmt spdlog tomlplusplus pugixml)

# addrlib — AMD's surface addressing library, for Latte's tiling (Phase 8.5).
# MIT, AMD's notice. GX2 itself is built on AMD's addrlib; this is Mesa's R800
# addrlib adapted to the R600/R700 variant in the Wii U, as decaf maintains it
# (see docs/decisions/ADR-005-addrlib.md). It publishes no tags, so the pin is
# a commit, which also rules out a shallow clone.
#
# No CMakeLists either: the target below is ours, built without the project's
# warning set and exposed with SYSTEM includes, like metal-cpp.
FetchContent_Declare(addrlib
    GIT_REPOSITORY https://github.com/decaf-emu/addrlib.git
    GIT_TAG        194162c47469ce620dd2470eb767ff5e42f5954a)
FetchContent_MakeAvailable(addrlib)

if(NOT TARGET addrlib)
    add_library(addrlib STATIC
        "${addrlib_SOURCE_DIR}/src/addrinterface.cpp"
        "${addrlib_SOURCE_DIR}/src/core/addrelemlib.cpp"
        "${addrlib_SOURCE_DIR}/src/core/addrlib.cpp"
        "${addrlib_SOURCE_DIR}/src/core/addrobject.cpp"
        "${addrlib_SOURCE_DIR}/src/r600/r600addrlib.cpp")
    target_include_directories(addrlib
        SYSTEM PUBLIC "${addrlib_SOURCE_DIR}/include"
        PRIVATE "${addrlib_SOURCE_DIR}/include/addrlib"
                "${addrlib_SOURCE_DIR}/src"
                "${addrlib_SOURCE_DIR}/src/core"
                "${addrlib_SOURCE_DIR}/src/r600")
    # Its sources draw Clang's default warnings (unhandled enum values in
    # switches, missing `override`); not ours to fix, so not ours to print.
    target_compile_options(addrlib PRIVATE -w)
    set_target_properties(addrlib PROPERTIES FOLDER "third_party" POSITION_INDEPENDENT_CODE ON)
endif()

# metal-cpp — Apple's header-only C++ interface to Metal (Phase 1.4).
#
# Apple-only, and fetched only there: a Linux core build has no use for it.
# The repository publishes no releases, only tags of the form
# `release/metal-cpp_macOS<n>_iOS<n>`, and the tag has to match the SDK being
# built against rather than being the newest -- headers for an SDK newer than
# the one installed reference symbols that are not there.
#
# There is no CMakeLists in the repository, so `MakeAvailable` only downloads
# it and the interface target below is ours. The include root is the
# repository root: `#include <Metal/Metal.hpp>`.
if(WIIU_PLATFORM_APPLE)
    FetchContent_Declare(metalcpp
        GIT_REPOSITORY https://github.com/apple/metal-cpp.git
        GIT_TAG        release/metal-cpp_macOS26_iOS26
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(metalcpp)

    if(NOT TARGET metal-cpp)
        add_library(metal-cpp INTERFACE)
        # SYSTEM, because these headers are not ours and do not survive the
        # project's warning set (`-Wall -Wextra -Wpedantic -Werror`).
        target_include_directories(metal-cpp SYSTEM INTERFACE "${metalcpp_SOURCE_DIR}")
    endif()
endif()

if(WIIU_BUILD_TESTS)
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.16.0
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()

# ---- declared, enabled by a later phase -----------------------------------
#
# SDL3             release-3.4.8             Phase 9    input / audio device
#                  https://github.com/libsdl-org/SDL
#                  (Phase 1.4 was going to take SDL3's Metal view for the
#                   window. It turned out to be about ninety lines of
#                   Objective-C++ to make an NSWindow and a CAMetalLayer by
#                   hand, against a dependency that also wants to own the
#                   event loop -- so the window is ours and SDL3 waits for the
#                   phase that actually needs a gamepad and an audio device.)
# imgui            v1.92.9-docking           Phase 10.1 debug UI (Metal backend)
#                  https://github.com/ocornut/imgui
# zlib             v1.3.1                    Phase 5.1  RPL SHF_DEFLATED sections
#                  https://github.com/madler/zlib
# mbedtls          v3.6.7                    Phase 5.7  AES-128-CBC, SHA, RSA verify
#                  https://github.com/Mbed-TLS/mbedtls
# mimalloc         v2.1.9                    Phase 12   P38, once there is a hot path
#                  https://github.com/microsoft/mimalloc
