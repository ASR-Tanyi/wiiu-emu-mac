# Phase 1.2 — one helper so every subsystem library is built the same way.
#
#   wiiu_add_library(common
#       SOURCES  src/log.cpp
#       HEADERS  include/wiiu/common/log.hpp
#       DEPS     fmt::fmt
#       PRIVATE_DEPS spdlog::spdlog)
#
# Creates target `wiiu_common` plus alias `wiiu::common`.

function(wiiu_add_library name)
    cmake_parse_arguments(ARG "" "" "SOURCES;HEADERS;DEPS;PRIVATE_DEPS;DEFINES" ${ARGN})

    set(target "wiiu_${name}")
    add_library(${target} STATIC ${ARG_SOURCES} ${ARG_HEADERS})
    add_library(wiiu::${name} ALIAS ${target})

    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

    if(ARG_DEPS)
        target_link_libraries(${target} PUBLIC ${ARG_DEPS})
    endif()
    if(ARG_PRIVATE_DEPS)
        target_link_libraries(${target} PRIVATE ${ARG_PRIVATE_DEPS})
    endif()
    if(ARG_DEFINES)
        target_compile_definitions(${target} PUBLIC ${ARG_DEFINES})
    endif()

    if(WIIU_PLATFORM_APPLE)
        target_compile_definitions(${target} PUBLIC WIIU_PLATFORM_APPLE=1)
    endif()

    wiiu_set_warnings(${target})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${target}
        FOLDER "libs")
endfunction()
