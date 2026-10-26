// Phase 6.11 — the system libraries a title touches on its way to a frame.
//
// `nsysnet`, `nn_act`, `nn_save`, `nn_acp`: networking, the user account, save
// data and the application control protocol. A title calls into these during
// start-up whether or not it uses them for anything, and what it needs back is
// "yes, that worked" rather than any real behaviour.
//
// Most of that happens without this file. The missing-function policy (6.12)
// already registers any unresolved import as a named stub, logs it loudly with
// its library, name and caller the first time it is called, and returns zero
// -- which for these libraries is success, an empty account slot, or a null
// handle, all of which a title copes with. So what is registered here is only
// the handful whose names are known well enough to be worth writing down, and
// the ones that need something other than zero.
//
// Guessing at the rest would be worse than leaving them out: a stub registered
// under a name no title imports is dead code that reads like a specification.
//
// Two get more than a name, because WHB's UDP log -- which most homebrew
// starts first -- calls them: nsysnet's sockets open, take options and send,
// into no network (what goes to WHB's log port is logged here), and nn_ac
// accepts a connection request it never completes.
#pragma once

#include <wiiu/hle/registry.hpp>

namespace wiiu::hle {

/// Register `nsysnet`, `nn_ac`, and the start-up stubs for `nn_save` and
/// `nn_acp`.
void register_system_libraries(Registry& registry);

} // namespace wiiu::hle
