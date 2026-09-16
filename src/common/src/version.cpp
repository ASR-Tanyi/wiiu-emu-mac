#include <wiiu/common/version.hpp>

#include <fmt/format.h>

#include <string>

namespace wiiu::version {

std::string_view full() noexcept {
    static const std::string cached = [] {
        std::string out = fmt::format("{} {}", project, string);
        if (!git_hash.empty() && git_hash != "unknown") {
            out += fmt::format(" ({}{}", git_hash, git_dirty ? "-dirty" : "");
            if (!build_type.empty()) {
                out += fmt::format(", {}", build_type);
            }
            out += ")";
        } else if (!build_type.empty()) {
            out += fmt::format(" ({})", build_type);
        }
        return out;
    }();
    return cached;
}

} // namespace wiiu::version
