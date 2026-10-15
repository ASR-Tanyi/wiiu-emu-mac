// Phase 8.4 — see registers.hpp.

#include <wiiu/latte/registers.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace wiiu::latte {

WindowRange window_range(Window window) noexcept {
    switch (window) {
        case Window::Config:
            return {0x08000, 0x0AC00};
        case Window::Context:
            return {0x28000, 0x29000};
        case Window::AluConst:
            return {0x30000, 0x32000};
        case Window::Resource:
            return {0x38000, 0x3C000};
        case Window::Sampler:
            return {0x3C000, 0x3CFF0};
        case Window::CtlConst:
            return {0x3CFF0, 0x3E200};
        case Window::LoopConst:
            return {0x3E200, 0x3E380};
        case Window::BoolConst:
            return {0x3E380, 0x40000};
    }
    return {};
}

std::string_view window_name(Window window) noexcept {
    switch (window) {
        case Window::Config:
            return "config";
        case Window::Context:
            return "context";
        case Window::AluConst:
            return "ALU constant";
        case Window::Resource:
            return "resource";
        case Window::Sampler:
            return "sampler";
        case Window::CtlConst:
            return "control constant";
        case Window::LoopConst:
            return "loop constant";
        case Window::BoolConst:
            return "bool constant";
    }
    return "?";
}

u32 RegisterFile::read(u32 address) const noexcept {
    if (address >= register_space_bytes || (address % 4) != 0) {
        return 0;
    }
    return values_[address / 4];
}

bool RegisterFile::write(u32 address, u32 value) noexcept {
    if (address >= register_space_bytes || (address % 4) != 0) {
        return false;
    }
    values_[address / 4] = value;
    ++writes_;
    return true;
}

bool RegisterFile::write_type0(u32 base_index, std::span<const u32> values, std::string& error) {
    // Checked in u64: a base near the top plus a long run must not wrap
    // round to a small, valid-looking index.
    if (static_cast<u64>(base_index) + values.size() > register_count) {
        error = fmt::format("a Type 0 write of {} registers from {:#07x} runs past the register "
                            "space",
                            values.size(),
                            static_cast<u64>(base_index) * 4);
        return false;
    }
    std::copy(values.begin(), values.end(), values_.begin() + base_index);
    writes_ += values.size();
    return true;
}

bool RegisterFile::write_window(Window window,
                                u32 offset,
                                std::span<const u32> values,
                                std::string& error) {
    const WindowRange range = window_range(window);
    if (static_cast<u64>(offset) + values.size() > range.words()) {
        error = fmt::format("a {} write of {} registers at offset {:#x} runs past the window "
                            "({:#07x}-{:#07x})",
                            window_name(window),
                            values.size(),
                            offset,
                            range.base,
                            range.end);
        return false;
    }
    std::copy(values.begin(), values.end(), values_.begin() + (range.base / 4) + offset);
    writes_ += values.size();
    return true;
}

void RegisterFile::reset() noexcept {
    std::fill(values_.begin(), values_.end(), 0U);
    writes_ = 0;
}

} // namespace wiiu::latte
