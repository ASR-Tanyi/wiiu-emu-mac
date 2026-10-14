// Phase 7.6 — making a shader compiler's complaints readable.
//
// Metal reports a compile error the way clang does, against a file called
// `program_source` that does not exist anywhere a person can open. When the
// source is generated -- and from Phase 8.6 on, all of it is -- a line number
// is the least useful thing an error can say. This puts the line back under
// the complaint, so a decompiler bug reads as the code that was wrong.
//
// Platform-independent on purpose: it is the one part of shader compilation
// that can be tested without a GPU.
#pragma once

#include <string>
#include <string_view>

namespace wiiu::gpu {

/// `output`, with the source line quoted (and a caret under the column)
/// after each diagnostic of the form `<file>:<line>:<column>: ...` that names
/// `file`. Diagnostics whose next line already quotes the source are left
/// alone, and a line number past the end of the source is left alone too.
[[nodiscard]] std::string annotate_shader_errors(std::string_view source,
                                                 std::string_view output,
                                                 std::string_view file = "program_source");

} // namespace wiiu::gpu
