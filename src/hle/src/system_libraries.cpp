#include <wiiu/hle/system_libraries.hpp>

namespace wiiu::hle {

void register_system_libraries(Registry& registry) {
    // nsysnet: the BSD sockets layer. `socket_lib_init` returning zero is
    // success, and a title that only initialises the library -- which is most
    // of them, at start-up -- gets past it. Anything that then tries to open a
    // socket meets the unimplemented-import log, which is the right place to
    // find out that networking is not here yet.
    for (const char* name : {"socket_lib_init", "socket_lib_finish"}) {
        registry.add_stub("nsysnet", name);
    }

    // nn_save: save data. Zero is `SAVE_STATUS_OK`. The mount itself is 9.9;
    // until then a title can initialise the library and will be told no when
    // it asks for a directory, rather than being told yes and finding nothing.
    for (const char* name : {"SAVEInit", "SAVEShutdown"}) {
        registry.add_stub("nn_save", name);
    }

    // nn_acp: the application control protocol -- device emulation checks and
    // metadata. Zero is success here too.
    for (const char* name : {"ACPInitialize", "ACPFinalize"}) {
        registry.add_stub("nn_acp", name);
    }

    // nn_act is deliberately absent. Its exports are C++ symbols with mangled
    // names, and a stub registered under a name no title actually imports is
    // dead code that reads like a specification of what the library offers.
    // The automatic policy names them correctly when a real title calls one,
    // which is also the first moment there is anything to check the name
    // against.
}

} // namespace wiiu::hle
