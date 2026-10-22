// Phase 8.1 — GX2's textures.
//
// What is guarded: the five words GX2InitTextureRegs packs, field by field
// against decaf's and Cemu's packing -- shape, pitch (a compressed one in
// pixels), size, format, the format's type bits, the component map, the view
// of levels and slices, cubes, arrays and multisampled textures, the depth
// bit, and the zero counts held at one; and GX2Set…Texture writing all seven
// to the right stage's unit, with level 0 and level 1 where the GPU sees
// them and the swizzle applied as Cemu applies it, read back by the
// translator's `texture_resource`. And units past eighteen refused.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_surface.hpp>
#include <wiiu/hle/gx2_texture.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/resources.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
namespace res = latte::resource;

namespace {

/// A texture as a title makes one: the surface described, laid out by
/// GX2CalcSurfaceSizeAndAlignment, the image placed, the component map set.
[[nodiscard]] gx2::Texture make_texture(
    gx2::SurfaceDim dim, u32 width, u32 height, u32 depth, u32 format, u32 mip_levels = 1) {
    gx2::Texture texture{};
    texture.surface.dim = static_cast<u32>(dim);
    texture.surface.width = width;
    texture.surface.height = height;
    texture.surface.depth = depth;
    texture.surface.mip_levels = mip_levels;
    texture.surface.format = format;
    texture.surface.use = gx2::surface_use_texture;
    texture.surface.tile_mode = gx2::tile_mode_default;
    REQUIRE(gx2::calc_surface_size_and_alignment(texture.surface));
    texture.view_num_mips = mip_levels;
    texture.view_num_slices = depth;
    texture.comp_map = 0x00010203; // GX2_COMP_MAP(X, Y, Z, W)
    return texture;
}

} // namespace

TEST_CASE("GX2InitTextureRegs packs a 2D texture as decaf and Cemu do", "[hle][gx2][texture]") {
    gx2::Texture texture = make_texture(gx2::SurfaceDim::Texture2D, 256, 128, 1, 0x01A);
    REQUIRE(texture.surface.tile_mode == 4); // 2D thin1, GX2's default for this
    const u32 pitch = texture.surface.pitch;
    REQUIRE(pitch >= 256);
    gx2::init_texture_regs(texture);

    // WORD0: 2D, 2D thin1, colour, the pitch in eights less one, width less one.
    CHECK(texture.word0 == (1U | (4U << 3) | (((pitch / 8) - 1) << 8) | (255U << 19)));
    // WORD1: height less one, no depth, FMT_8_8_8_8.
    CHECK(texture.word1 == (127U | (0x1AU << 26)));
    // WORD4: unsigned, normalised, no swap, request size 2, X Y Z W, level 0.
    CHECK(texture.word4 == ((2U << 14) | (0U << 16) | (1U << 19) | (2U << 22) | (3U << 25)));
    // WORD5: level 0 alone, slice 0 alone.
    CHECK(texture.word5 == 0);
    // WORD6: anisotropy 4, modulation 7, a valid texture.
    CHECK(texture.word6 == 0x800000F0);

    // The same through the call, on the title's copy.
    memory::Memory memory;
    std::string error;
    REQUIRE(memory.initialize(memory::default_regions(), error));
    gx2::Texture guest = make_texture(gx2::SurfaceDim::Texture2D, 256, 128, 1, 0x01A);
    constexpr ppcaddr_t kTexture = 0x10003000;
    REQUIRE(gx2::write_guest(memory, kTexture, guest));
    Registry registry;
    coreinit::Context os;
    wiiu::log::set_all_levels(wiiu::log::Level::Off);
    coreinit::register_all(registry, os);
    Function* const init = registry.find("gx2", "GX2InitTextureRegs");
    REQUIRE(init != nullptr);
    cpu::CoreState state;
    state.gpr[3] = kTexture;
    Call call{.state = state, .memory = memory};
    init->handler(call);
    REQUIRE(gx2::read_guest(memory, kTexture, guest));
    CHECK(guest.word0 == texture.word0);
    CHECK(guest.word1 == texture.word1);
    CHECK(guest.word4 == texture.word4);
    CHECK(guest.word5 == texture.word5);
    CHECK(guest.word6 == texture.word6);
}

TEST_CASE("a texture's format bits, view and shape reach its words", "[hle][gx2][texture]") {
    const auto word4 = [](u32 format) {
        gx2::Texture texture = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, format);
        gx2::init_texture_regs(texture);
        return latte::TexResourceWord4{.raw = texture.word4};
    };
    // sRGB: degamma'd.
    CHECK(word4(0x41A).force_degamma);
    CHECK_FALSE(word4(0x01A).force_degamma);
    // SNORM: signed components, normalised.
    CHECK(word4(0x21A).format_comp_x == latte::FormatComp::Signed);
    CHECK(word4(0x21A).format_comp_w == latte::FormatComp::Signed);
    CHECK(word4(0x21A).num_format_all == latte::NumFormat::Norm);
    // UINT and SINT: integers.
    CHECK(word4(0x11A).num_format_all == latte::NumFormat::Int);
    CHECK(word4(0x11A).format_comp_y == latte::FormatComp::Unsigned);
    CHECK(word4(0x31A).num_format_all == latte::NumFormat::Int);
    CHECK(word4(0x31A).format_comp_z == latte::FormatComp::Signed);
    // FLOAT: scaled.
    CHECK(word4(0x80E).num_format_all == latte::NumFormat::Scaled);

    // A component map: one, zero, red, green.
    gx2::Texture swizzled = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, 0x01A);
    swizzled.comp_map = 0x05040001;
    gx2::init_texture_regs(swizzled);
    const latte::TexResourceWord4 map{.raw = swizzled.word4};
    CHECK(map.dst_sel_x == latte::Sel::One);
    CHECK(map.dst_sel_y == latte::Sel::Zero);
    CHECK(map.dst_sel_z == latte::Sel::X);
    CHECK(map.dst_sel_w == latte::Sel::Y);

    // BC1: the pitch, in blocks, counted in pixels.
    gx2::Texture bc1 = make_texture(gx2::SurfaceDim::Texture2D, 256, 256, 1, 0x031);
    const u32 blocks = bc1.surface.pitch;
    gx2::init_texture_regs(bc1);
    CHECK(latte::TexResourceWord0{.raw = bc1.word0}.pitch == ((blocks * 4) / 8) - 1);
    CHECK(latte::TexResourceWord1{.raw = bc1.word1}.data_format == 0x31);

    // A cube: six faces, one cube. A cube array: twelve faces, two, and YUV_CONV.
    gx2::Texture cube = make_texture(gx2::SurfaceDim::TextureCube, 64, 64, 6, 0x01A);
    gx2::init_texture_regs(cube);
    CHECK(latte::TexResourceWord0{.raw = cube.word0}.dim == latte::TexDim::Cube);
    CHECK(latte::TexResourceWord1{.raw = cube.word1}.depth == 0);
    CHECK(latte::TexResourceWord5{.raw = cube.word5}.yuv_conv == 0);
    gx2::Texture cubes = make_texture(gx2::SurfaceDim::TextureCube, 64, 64, 12, 0x01A);
    gx2::init_texture_regs(cubes);
    CHECK(latte::TexResourceWord1{.raw = cubes.word1}.depth == 1);
    CHECK(latte::TexResourceWord5{.raw = cubes.word5}.yuv_conv == 1);

    // An array, viewed from slice 1 for two slices and from level 1 for
    // three levels.
    gx2::Texture array = make_texture(gx2::SurfaceDim::Texture2DArray, 64, 64, 4, 0x01A, 4);
    array.view_first_slice = 1;
    array.view_num_slices = 2;
    array.view_first_mip = 1;
    array.view_num_mips = 3;
    gx2::init_texture_regs(array);
    CHECK(latte::TexResourceWord1{.raw = array.word1}.depth == 3);
    const latte::TexResourceWord5 view{.raw = array.word5};
    CHECK(view.base_array == 1);
    CHECK(view.last_array == 2);
    CHECK(view.last_level == 3);
    CHECK(latte::TexResourceWord4{.raw = array.word4}.base_level == 1);

    // A 2D texture's depth says nothing, whatever the surface holds.
    gx2::Texture flat = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, 0x01A);
    flat.surface.depth = 5;
    gx2::init_texture_regs(flat);
    CHECK(latte::TexResourceWord1{.raw = flat.word1}.depth == 0);

    // Multisampled, 4x: the last level is the samples' log2.
    gx2::Texture msaa = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, 0x01A);
    msaa.surface.dim = static_cast<u32>(gx2::SurfaceDim::Texture2DMsaa);
    msaa.surface.aa = 2;
    gx2::init_texture_regs(msaa);
    CHECK(latte::TexResourceWord5{.raw = msaa.word5}.last_level == 2);

    // A depth buffer read as a texture: TILE_TYPE.
    gx2::Texture depth = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, 0x80E);
    depth.surface.use = gx2::surface_use_texture | gx2::surface_use_depth_buffer;
    gx2::init_texture_regs(depth);
    CHECK(latte::TexResourceWord0{.raw = depth.word0}.tile_type);

    // Zero counts are held at one, in the title's structure too; a zero
    // width wraps, as Cemu's arithmetic does.
    gx2::Texture zero{};
    zero.surface.dim = static_cast<u32>(gx2::SurfaceDim::Texture2D);
    zero.surface.format = 0x01A;
    gx2::init_texture_regs(zero);
    CHECK(zero.view_num_mips == 1);
    CHECK(zero.view_num_slices == 1);
    CHECK(zero.surface.height == 1);
    CHECK(zero.surface.depth == 1);
    CHECK(zero.surface.mip_levels == 1);
    CHECK(latte::TexResourceWord0{.raw = zero.word0}.width == 0x1FFF);
    CHECK(latte::TexResourceWord0{.raw = zero.word0}.pitch == 0x7FF);
    CHECK(latte::TexResourceWord1{.raw = zero.word1}.height == 0);
}

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kTexture = 0x10003000;
constexpr ppcaddr_t kImage = 0x10100000;
constexpr ppcaddr_t kMipmaps = 0x10200000;

struct TextureFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    TextureFixture() {
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        coreinit::register_all(registry, os);
        os.memory = &memory;
        state.core_id = 1;
        state.timebase = 10'000;
        static_cast<void>(os.scheduler.adopt_main(state, kMain));
        call("GX2Init", {0});
    }

    void call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
    }

    void run_gpu() {
        call("GX2Flush", {});
        state.timebase += os.gpu.timeline().latency();
        static_cast<void>(os.scheduler.reschedule(state));
    }

    /// The texture written to guest memory and its registers worked out,
    /// as a title does.
    void put(gx2::Texture texture) {
        REQUIRE(gx2::write_guest(memory, kTexture, texture));
        call("GX2InitTextureRegs", {kTexture});
    }

    [[nodiscard]] latte::TextureResource resource(u32 id) const {
        return latte::texture_resource(os.gpu.registers(), id);
    }

    [[nodiscard]] physaddr_t physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }
};

} // namespace

TEST_CASE_METHOD(TextureFixture,
                 "GX2SetPixelTexture writes all seven words to the pixel stage's unit",
                 "[hle][gx2][texture]") {
    gx2::Texture texture = make_texture(gx2::SurfaceDim::Texture2D, 256, 128, 1, 0x01A);
    texture.surface.image = kImage;
    put(texture);
    call("GX2SetPixelTexture", {kTexture, 2});
    run_gpu();

    gx2::Texture packed{};
    REQUIRE(gx2::read_guest(memory, kTexture, packed));
    const latte::TextureResource unit = resource(res::ps_base + 2);
    CHECK(unit.valid());
    CHECK(unit.word0.raw == packed.word0);
    CHECK(unit.word1.raw == packed.word1);
    CHECK(unit.word4.raw == packed.word4);
    CHECK(unit.word5.raw == packed.word5);
    CHECK(unit.word6.raw == packed.word6);
    CHECK(unit.word1.data_format == 0x1A);
    CHECK(unit.word0.width == 255);
    // Level 0 where the GPU sees it; level 1, with no mip chain, the image.
    CHECK(unit.base == physical(kImage));
    CHECK(unit.mip == physical(kImage));
    CHECK_FALSE(resource(res::ps_base + 1).valid());
    CHECK(os.gpu.processor().stats().errors == 0);
}

TEST_CASE_METHOD(TextureFixture,
                 "each stage's texture goes to its own share of the window",
                 "[hle][gx2][texture]") {
    gx2::Texture texture = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, 0x01A);
    texture.surface.image = kImage;
    put(texture);
    call("GX2SetVertexTexture", {kTexture, 0});
    call("GX2SetGeometryTexture", {kTexture, 17});
    call("GX2SetComputeTexture", {kTexture, 3}); // the vertex stage's, as Cemu has it
    run_gpu();
    CHECK(resource(res::vs_base).valid());
    CHECK(resource(res::gs_base + 17).valid());
    CHECK(resource(res::vs_base + 3).valid());
    CHECK_FALSE(resource(res::ps_base).valid());

    // Unit 18 is past the stage's units: nothing is written.
    call("GX2SetPixelTexture", {kTexture, 18});
    run_gpu();
    CHECK_FALSE(resource(res::ps_base + 18).valid());
}

TEST_CASE_METHOD(TextureFixture,
                 "a texture's swizzle goes into each base while its level is macro-tiled",
                 "[hle][gx2][texture]") {
    // A mip chain whose level 1 is still macro-tiled, with a bank and pipe
    // swizzle of 3 (GX2SetSurfaceSwizzle puts it in the second byte).
    gx2::Texture texture = make_texture(gx2::SurfaceDim::Texture2D, 1024, 1024, 1, 0x01A, 3);
    REQUIRE(((texture.surface.swizzle >> 16) & 0xFF) > 1);
    texture.surface.swizzle = texture.surface.swizzle | (3U << 8);
    texture.surface.image = kImage;
    texture.surface.mipmaps = kMipmaps;
    put(texture);
    call("GX2SetPixelTexture", {kTexture, 0});
    run_gpu();
    CHECK(resource(res::ps_base).base == (physical(kImage) ^ 0x300));
    CHECK(resource(res::ps_base).mip == (physical(kMipmaps) ^ 0x300));

    // Macro-tiling stops at level 1: level 1's base has no swizzle.
    texture.surface.swizzle = (texture.surface.swizzle & 0xFF00FFFF) | (1U << 16);
    put(texture);
    call("GX2SetPixelTexture", {kTexture, 1});
    run_gpu();
    CHECK(resource(res::ps_base + 1).base == (physical(kImage) ^ 0x300));
    CHECK(resource(res::ps_base + 1).mip == physical(kMipmaps));

    // A linear texture has none at all.
    gx2::Texture linear = make_texture(gx2::SurfaceDim::Texture2D, 64, 64, 1, 0x01A);
    linear.surface.tile_mode = static_cast<u32>(latte::TileMode::LinearAligned);
    linear.surface.swizzle = 0x000D0300;
    linear.surface.image = kImage;
    put(linear);
    call("GX2SetPixelTexture", {kTexture, 2});
    run_gpu();
    CHECK(resource(res::ps_base + 2).base == physical(kImage));
    CHECK(os.gpu.processor().stats().errors == 0);
}
