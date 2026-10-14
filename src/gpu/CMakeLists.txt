# wiiu::gpu — the backend interface (Phase 7.1) and its implementations.
#
# The interface, the null backend and the descriptor validation are
# platform-independent and build everywhere, which is what lets the Phase 8
# command processor be tested on a machine with no GPU at all. Only the Metal
# backend is Apple-only, and on any other host it is replaced by a
# `create_metal_backend` that says so rather than failing to link.

set(WIIU_GPU_SOURCES
    src/backend.cpp
    src/null_backend.cpp
    src/presenter.cpp
    src/shader_diagnostics.cpp)

set(WIIU_GPU_HEADERS
    include/wiiu/gpu/backend.hpp
    include/wiiu/gpu/null_backend.hpp
    include/wiiu/gpu/presenter.hpp
    include/wiiu/gpu/shader_diagnostics.hpp
    include/wiiu/gpu/types.hpp)

if(WIIU_PLATFORM_APPLE)
    list(APPEND WIIU_GPU_SOURCES src/metal/metal_backend.cpp)
    list(APPEND WIIU_GPU_HEADERS include/wiiu/gpu/metal/raii.hpp)
else()
    list(APPEND WIIU_GPU_SOURCES src/metal/metal_backend_absent.cpp)
endif()

wiiu_add_library(gpu
    SOURCES ${WIIU_GPU_SOURCES}
    HEADERS ${WIIU_GPU_HEADERS}
    DEPS wiiu::common)

if(WIIU_PLATFORM_APPLE)
    target_link_libraries(wiiu_gpu PUBLIC metal-cpp)

    # Metal and QuartzCore are the frameworks metal-cpp is a shim over;
    # Foundation carries NSString, NSError and the autorelease pool. They are
    # part of the OS, so unlike the MoltenVK path this adds nothing to ship
    # (ADR-001).
    target_link_libraries(wiiu_gpu PUBLIC
        "-framework Metal"
        "-framework QuartzCore"
        "-framework Foundation")

    # metal-cpp's headers are not ours and are not warning-clean under the
    # project's set. They are included as SYSTEM by the metal-cpp target, so
    # this is only about the one translation unit that instantiates them.
    set_source_files_properties(src/metal/metal_backend.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-parameter")
endif()
