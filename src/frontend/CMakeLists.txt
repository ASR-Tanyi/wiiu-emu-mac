# The emulator executable.
#
# Phase 1.2 built a command-line shell; Phase 1.4 adds the platform layer. It
# still builds on a non-Apple host: the window and the JIT probe each have an
# implementation that reports it cannot do the thing, rather than being absent
# and taking the whole frontend out of the core-only build with them.

set(WIIU_FRONTEND_SOURCES main.cpp display.cpp)

if(WIIU_PLATFORM_APPLE)
    list(APPEND WIIU_FRONTEND_SOURCES
        platform/window_macos.mm
        platform/jit_probe_macos.cpp)
else()
    list(APPEND WIIU_FRONTEND_SOURCES
        platform/window_absent.cpp
        platform/jit_probe_absent.cpp)
endif()

add_executable(wiiu-emu ${WIIU_FRONTEND_SOURCES})

target_include_directories(wiiu-emu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(wiiu-emu PRIVATE wiiu::common wiiu::gpu wiiu::hle)
wiiu_set_warnings(wiiu-emu)

set_target_properties(wiiu-emu PROPERTIES
    OUTPUT_NAME "wiiu-emu"
    FOLDER "app")

if(WIIU_PLATFORM_APPLE)
    # Cocoa for the window, QuartzCore for CAMetalLayer. Both are system
    # frameworks, so nothing is shipped alongside the binary (ADR-001).
    target_link_libraries(wiiu-emu PRIVATE "-framework Cocoa" "-framework QuartzCore")

    set_target_properties(wiiu-emu PROPERTIES
        XCODE_ATTRIBUTE_ENABLE_HARDENED_RUNTIME YES)

    # Phase 1.8 — sign on every build, not at release.
    #
    # `--sign -` is an ad-hoc signature, which is all that is needed to make
    # the JIT entitlement real on this machine; notarization needs a Developer
    # ID certificate and belongs to Phase 13.5. `--options runtime` turns on
    # the Hardened Runtime, without which the entitlement is ignored and
    # MAP_JIT quietly works for the wrong reason — which would make the Phase
    # 1.8 probe pass here and fail for everyone else.
    set(WIIU_ENTITLEMENTS "${CMAKE_CURRENT_SOURCE_DIR}/wiiu-emu.entitlements")
    add_custom_command(TARGET wiiu-emu POST_BUILD
        COMMAND codesign --force --sign - --timestamp=none --options runtime
                --entitlements "${WIIU_ENTITLEMENTS}"
                "$<TARGET_FILE:wiiu-emu>"
        COMMENT "Signing wiiu-emu ad-hoc with the JIT entitlement (Phase 1.8)"
        VERBATIM)
endif()
