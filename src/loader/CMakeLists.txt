# wiiu::loader — RPX/RPL parsing, and later discs and the virtual filesystem.

find_package(ZLIB REQUIRED)

wiiu_add_library(loader
    SOURCES
        src/metadata.cpp
        src/relocations.cpp
        src/rpl.cpp
        src/vfs.cpp
    HEADERS
        include/wiiu/loader/metadata.hpp
        include/wiiu/loader/relocations.hpp
        include/wiiu/loader/rpl.hpp
        include/wiiu/loader/vfs.hpp
    DEPS
        wiiu::common
    PRIVATE_DEPS
        ZLIB::ZLIB
        pugixml::pugixml)
