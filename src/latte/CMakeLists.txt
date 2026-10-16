# wiiu::latte — the Latte GPU's own formats (Phase 8): PM4 command streams,
# the register file, and later the shader microcode and surface tiling.
#
# Kept out of wiiu::gpu on purpose. That library is the backend interface, and
# no second backend would want Latte's encodings in it (7.1); this is the
# translator that turns them into calls on it. Platform-independent, so all of
# it is tested on Linux too.

wiiu_add_library(latte
    SOURCES
        src/command_processor.cpp
        src/control_flow.cpp
        src/formats.cpp
        src/pm4.cpp
        src/registers.cpp
        src/shader_program.cpp
        src/surface.cpp
        src/timeline.cpp
    HEADERS
        include/wiiu/latte/color_buffer.hpp
        include/wiiu/latte/command_processor.hpp
        include/wiiu/latte/control_flow.hpp
        include/wiiu/latte/depth_buffer.hpp
        include/wiiu/latte/draw.hpp
        include/wiiu/latte/formats.hpp
        include/wiiu/latte/microcode.hpp
        include/wiiu/latte/pm4.hpp
        include/wiiu/latte/registers.hpp
        include/wiiu/latte/render_state.hpp
        include/wiiu/latte/resources.hpp
        include/wiiu/latte/shader_program.hpp
        include/wiiu/latte/shader_state.hpp
        include/wiiu/latte/surface.hpp
        include/wiiu/latte/timeline.hpp
        include/wiiu/latte/viewport.hpp
    DEPS
        wiiu::common
        wiiu::memory
    PRIVATE_DEPS
        addrlib)
