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
# metal-cpp        metal-cpp_macOS26_iOS26   Phase 1.4  (Apple only, headers)
#                  https://github.com/apple/metal-cpp
# SDL3             release-3.4.8             Phase 1.4  window / input / audio
#                  https://github.com/libsdl-org/SDL
# imgui            v1.92.9-docking           Phase 10.1 debug UI (Metal backend)
#                  https://github.com/ocornut/imgui
# zlib             v1.3.1                    Phase 5.1  RPL SHF_DEFLATED sections
#                  https://github.com/madler/zlib
# mbedtls          v3.6.7                    Phase 5.7  AES-128-CBC, SHA, RSA verify
#                  https://github.com/Mbed-TLS/mbedtls
# mimalloc         v2.1.9                    Phase 12   P38, once there is a hot path
#                  https://github.com/microsoft/mimalloc
