# wiiu::common — types, bit helpers, logging, version.
# No dependency on any other subsystem; everything else depends on this.

# --- generated version header ---------------------------------------------
# Captured at configure time. Stale after a commit until the next configure,
# which is fine for a banner; nothing correctness-bearing reads it.
set(WIIU_GIT_HASH "unknown")
set(WIIU_GIT_BRANCH "unknown")
set(WIIU_GIT_DIRTY "false")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE WIIU_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE WIIU_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE WIIU_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT WIIU_GIT_STATUS STREQUAL "")
        set(WIIU_GIT_DIRTY "true")
    endif()
    if(WIIU_GIT_HASH STREQUAL "")
        set(WIIU_GIT_HASH "unknown")
    endif()
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/wiiu/common/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/wiiu/common/version.hpp"
    @ONLY)

# --- library ---------------------------------------------------------------
wiiu_add_library(common
    SOURCES
        src/config.cpp
        src/log.cpp
        src/version.cpp
    HEADERS
        include/wiiu/common/bitfield.hpp
        include/wiiu/common/config.hpp
        include/wiiu/common/bits.hpp
        include/wiiu/common/endian.hpp
        include/wiiu/common/log.hpp
        include/wiiu/common/types.hpp
        "${CMAKE_CURRENT_BINARY_DIR}/include/wiiu/common/version.hpp"
    DEPS
        fmt::fmt
    PRIVATE_DEPS
        spdlog::spdlog
        tomlplusplus::tomlplusplus)
