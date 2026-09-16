# wiiu::memory — the guest address space, page permissions, MMIO routing and
# the cache model.

wiiu_add_library(memory
    SOURCES
        src/address_space.cpp
        src/map.cpp
        src/memory.cpp
    HEADERS
        include/wiiu/memory/address_space.hpp
        include/wiiu/memory/map.hpp
        include/wiiu/memory/memory.hpp
    DEPS
        wiiu::common)
