# wiiu::hle — Cafe OS libraries, reimplemented natively.

wiiu_add_library(hle
    SOURCES
        src/coreinit.cpp
        src/coreinit_thread.cpp
        src/coreinit_time.cpp
        src/filesystem.cpp
        src/heap.cpp
        src/loader.cpp
        src/recording.cpp
        src/registry.cpp
        src/scheduler.cpp
    HEADERS
        include/wiiu/hle/coreinit.hpp
        include/wiiu/hle/filesystem.hpp
        include/wiiu/hle/heap.hpp
        include/wiiu/hle/loader.hpp
        include/wiiu/hle/recording.hpp
        include/wiiu/hle/registry.hpp
        include/wiiu/hle/scheduler.hpp
    DEPS
        wiiu::common
        wiiu::cpu
        wiiu::loader
        wiiu::memory)
