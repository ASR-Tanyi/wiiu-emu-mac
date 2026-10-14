// Phase 7.6 — see shader_diagnostics.hpp.

#include <wiiu/gpu/shader_diagnostics.hpp>

#include <fmt/format.h>

#include <charconv>
#include <optional>
#include <vector>

namespace wiiu::gpu {

namespace {

[[nodiscard]] std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    while (!text.empty()) {
        const std::size_t end = text.find('\n');
        if (end == std::string_view::npos) {
            lines.push_back(text);
            break;
        }
        lines.push_back(text.substr(0, end));
        text.remove_prefix(end + 1);
    }
    return lines;
}

/// Leading and trailing whitespace off, for comparing a quoted line with the
/// source line it might already be quoting.
[[nodiscard]] std::string_view trim(std::string_view text) {
    const std::size_t first = text.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1);
}

struct Location {
    std::size_t line = 0;   ///< 1-based
    std::size_t column = 0; ///< 1-based
};

/// `<file>:<line>:<column>:` at the start of a diagnostic, if it is there.
[[nodiscard]] std::optional<Location> parse_location(std::string_view text, std::string_view file) {
    if (text.substr(0, file.size()) != file || text.size() <= file.size() ||
        text[file.size()] != ':') {
        return std::nullopt;
    }
    text.remove_prefix(file.size() + 1);

    Location location;
    const auto parse = [&text](std::size_t& out) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
        if (error != std::errc{} || end == text.data() || end == text.data() + text.size() ||
            *end != ':') {
            return false;
        }
        text.remove_prefix(static_cast<std::size_t>(end - text.data()) + 1);
        return true;
    };
    if (!parse(location.line) || !parse(location.column) || location.line == 0) {
        return std::nullopt;
    }
    return location;
}

} // namespace

std::string
annotate_shader_errors(std::string_view source, std::string_view output, std::string_view file) {
    const std::vector<std::string_view> source_lines = split_lines(source);
    const std::vector<std::string_view> output_lines = split_lines(output);

    std::string result;
    for (std::size_t index = 0; index < output_lines.size(); ++index) {
        const std::string_view line = output_lines[index];
        result.append(line);
        result.push_back('\n');

        const std::optional<Location> location = parse_location(line, file);
        if (!location || location->line > source_lines.size()) {
            continue;
        }
        const std::string_view quoted = source_lines[location->line - 1];

        // Some compilers quote the line themselves. Saying it twice is noise.
        if (index + 1 < output_lines.size() && trim(output_lines[index + 1]) == trim(quoted) &&
            !trim(quoted).empty()) {
            continue;
        }

        const std::string number = fmt::format("{:>5}", location->line);
        result += fmt::format("{} | {}\n", number, quoted);
        const std::size_t caret = location->column == 0 ? 0 : location->column - 1;
        result +=
            fmt::format("{} | {}^\n", std::string(number.size(), ' '), std::string(caret, ' '));
    }
    if (!result.empty() && output.back() != '\n') {
        result.pop_back();
    }
    return result;
}

} // namespace wiiu::gpu
