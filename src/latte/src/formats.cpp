#include <wiiu/latte/formats.hpp>

#include <algorithm>
#include <cmath>

namespace wiiu::latte {

namespace {

[[nodiscard]] u8 unorm8(f32 value) noexcept {
    if (std::isnan(value)) {
        return 0;
    }
    return static_cast<u8>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
}

/// Linear to sRGB, IEC 61966-2-1's curve.
[[nodiscard]] f32 srgb_encode(f32 linear) noexcept {
    if (std::isnan(linear)) {
        return 0.0F;
    }
    const f32 c = std::clamp(linear, 0.0F, 1.0F);
    return c <= 0.0031308F ? c * 12.92F : (1.055F * std::pow(c, 1.0F / 2.4F)) - 0.055F;
}

} // namespace

std::optional<EncodedColor> encode_color(u32 surface_format,
                                         const std::array<f32, 4>& rgba) noexcept {
    EncodedColor out;
    switch (surface_format) {
        case 0x001: // UNORM_R8
            out.size = 1;
            break;
        case 0x007: // UNORM_R8_G8
            out.size = 2;
            break;
        case 0x01A: // UNORM_R8_G8_B8_A8
            out.size = 4;
            break;
        case 0x41A: // SRGB_R8_G8_B8_A8
            out.bytes = {unorm8(srgb_encode(rgba[0])),
                         unorm8(srgb_encode(rgba[1])),
                         unorm8(srgb_encode(rgba[2])),
                         unorm8(rgba[3])};
            out.size = 4;
            return out;
        default:
            return std::nullopt;
    }
    for (u32 i = 0; i < out.size; ++i) {
        out.bytes[i] = unorm8(rgba[i]);
    }
    return out;
}

} // namespace wiiu::latte
