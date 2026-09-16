# The emulator executable.
#
# Phase 1.2 builds a command-line shell. Phase 1.4 adds the macOS platform
# layer (CAMetalLayer window + metal-cpp RAII), which is when this target
# stops building on non-Apple hosts.

set(WIIU_FRONTEND_SOURCES main.cpp)

# if(WIIU_PLATFORM_APPLE)                       # Phase 1.4
#     list(APPEND WIIU_FRONTEND_SOURCES platform/window_macos.mm)
# endif()

add_executable(wiiu-emu ${WIIU_FRONTEND_SOURCES})

target_link_libraries(wiiu-emu PRIVATE wiiu::common wiiu::hle)
wiiu_set_warnings(wiiu-emu)

set_target_properties(wiiu-emu PROPERTIES
    OUTPUT_NAME "wiiu-emu"
    FOLDER "app")

if(WIIU_PLATFORM_APPLE)
    # Phase 1.8 proves this pipeline on the empty app, long before the JIT
    # (Phase 12) actually needs the entitlement.
    set_target_properties(wiiu-emu PROPERTIES
        XCODE_ATTRIBUTE_ENABLE_HARDENED_RUNTIME YES)
endif()
