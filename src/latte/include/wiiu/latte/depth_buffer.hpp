// Latte's depth-buffer registers: DB_DEPTH_SIZE, DB_DEPTH_VIEW and
// DB_DEPTH_INFO, the hierarchical-Z and prefetch registers beside them, the
// polygon-offset format control, and the two clear values.
//
// The colour buffer's twin (color_buffer.hpp): GX2 works these out once, in
// `GX2InitDepthBufferRegs`, keeps them in the title's `GX2DepthBuffer`, and
// `GX2SetDepthBuffer` writes them. The size, view and info fields are the
// R600 family's -- Linux's `r600d.h` and decaf's tables agree on every bit --
// and so are the depth formats. The hierarchical-Z, prefetch, preload and
// polygon-offset fields are decaf's reading alone; nothing reads them before
// the translator (8.11).
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>
#include <wiiu/latte/surface.hpp>

#include <optional>

namespace wiiu::latte {

/// DB_DEPTH_INFO's depth formats.
enum class DbFormat : u32 {
    Invalid = 0,
    Depth16 = 1,
    DepthX8_24 = 2,
    Depth8_24 = 3,
    DepthX8_24Float = 4,
    Depth8_24Float = 5,
    Depth32Float = 6,
    DepthX24_8_32Float = 7,
};

/// The depth format a GX2 surface format is used as: decaf's mapping, the
/// five formats GX2 accepts for a depth buffer. Nothing for any other.
[[nodiscard]] constexpr std::optional<DbFormat> depth_format(u32 surface_format) noexcept {
    switch (surface_format) {
        case 0x005: // UNORM_R16
            return DbFormat::Depth16;
        case 0x011: // UNORM_R24_X8
            return DbFormat::Depth8_24;
        case 0x80E: // FLOAT_R32
            return DbFormat::Depth32Float;
        case 0x811: // FLOAT_D24_S8
            return DbFormat::Depth8_24Float;
        case 0x81C: // FLOAT_X8_X24
            return DbFormat::DepthX24_8_32Float;
        default:
            return std::nullopt;
    }
}

/// DB_DEPTH_SIZE: the level's pitch and slice, in 8x8 micro-tiles, minus one.
union DbDepthSize {
    u32 raw;
    BitField<0, 10, u32> pitch_tile_max;
    BitField<10, 20, u32> slice_tile_max;
};

/// DB_DEPTH_VIEW: the slices drawn to, first and last.
union DbDepthView {
    u32 raw;
    BitField<0, 11, u32> slice_start;
    BitField<13, 11, u32> slice_max;
};

/// DB_DEPTH_INFO.
union DbDepthInfo {
    u32 raw;
    BitField<0, 3, DbFormat> format;
    BitField<3, 1, bool> read_size_512; ///< 512-bit reads rather than 256
    BitField<15, 4, TileMode> array_mode;
    BitField<25, 1, bool> tile_surface_enable; ///< hierarchical Z on
    BitField<26, 1, bool> tile_compact;
    BitField<31, 1, bool> zrange_precision;
};

/// DB_HTILE_SURFACE.
union DbHtileSurface {
    u32 raw;
    BitField<0, 1, bool> htile_width;
    BitField<1, 1, bool> htile_height;
    BitField<2, 1, bool> linear;
    BitField<3, 1, bool> full_cache;
    BitField<4, 1, bool> htile_uses_preload_window;
    BitField<5, 1, bool> preload;
    BitField<6, 6, u32> prefetch_width;
    BitField<12, 6, u32> prefetch_height;
};

/// DB_PREFETCH_LIMIT.
union DbPrefetchLimit {
    u32 raw;
    BitField<0, 10, u32> depth_height_tile_max;
};

/// DB_PRELOAD_CONTROL: the hierarchical-Z preload window, in 32-pixel units.
union DbPreloadControl {
    u32 raw;
    BitField<0, 8, u32> start_x;
    BitField<8, 8, u32> start_y;
    BitField<16, 8, u32> max_x;
    BitField<24, 8, u32> max_y;
};

/// PA_SU_POLY_OFFSET_DB_FMT_CNTL: what polygon offset is scaled against.
union PaSuPolyOffsetDbFmtCntl {
    u32 raw;
    BitField<0, 8, u32> neg_num_db_bits;
    BitField<8, 1, bool> db_is_float;
};

namespace reg {
/// The values a depth clear writes, as `GX2ClearDepthStencilEx` asks:
/// stencil in the low byte, depth as a float's bits. Byte addresses; decaf's
/// and Cemu's tables agree.
inline constexpr u32 db_stencil_clear = 0x28028;
inline constexpr u32 db_depth_clear = 0x2802C;
} // namespace reg

} // namespace wiiu::latte
