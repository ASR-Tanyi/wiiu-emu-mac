// Phase 8.1 — GX2's samplers.
//
// What is guarded: the three words GX2InitSampler makes, against Cemu's
// (point Z and mip filters and TEX_ARRAY_OVERRIDE included); each
// GX2InitSampler… call changing its own fields and nothing else -- the
// anisotropic filters, the LODs and bias floored to fixed point and held to
// their fields, NaN read as zero, and decaf's three adjustments; GX2Set…
// Sampler writing the words to the right stage's sampler, and a unit past
// eighteen not spilling into the next stage's; and the border colours as
// config registers, read back by the translator's `border_color`.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_sampler.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/samplers.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
namespace smp = latte::sampler;

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kSampler = 0x10003000;
constexpr ppcaddr_t kOther = 0x10003100;

/// GX2's enums, by the numbers a title passes.
constexpr u32 kClampWrap = 0;
constexpr u32 kClampMirror = 1;
constexpr u32 kClampLastTexel = 2;
constexpr u32 kClampBorder = 6;
constexpr u32 kFilterPoint = 0;
constexpr u32 kFilterLinear = 1;

struct SamplerFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    SamplerFixture() {
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

    void call(std::string_view name, std::vector<u32> arguments, std::vector<f32> floats = {}) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        for (std::size_t i = 0; i < floats.size(); ++i) {
            state.fpr[1 + i].ps0 = static_cast<f64>(floats[i]);
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
    }

    void run_gpu() {
        call("GX2Flush", {});
        state.timebase += os.gpu.timeline().latency();
        static_cast<void>(os.scheduler.reschedule(state));
    }

    /// The title's sampler, as the translator's view of the words.
    [[nodiscard]] latte::TexSampler guest(ppcaddr_t address = kSampler) const {
        gx2::Sampler sampler{};
        REQUIRE(gx2::read_guest(memory, address, sampler));
        latte::TexSampler out;
        out.word0.raw = sampler.word0;
        out.word1.raw = sampler.word1;
        out.word2.raw = sampler.word2;
        return out;
    }

    [[nodiscard]] latte::TexSampler registers(u32 id) const {
        return latte::texture_sampler(os.gpu.registers(), id);
    }
};

} // namespace

TEST_CASE_METHOD(SamplerFixture,
                 "GX2InitSampler makes a sampler as Cemu's GX2 does",
                 "[hle][gx2][sampler]") {
    // Whatever was there before goes.
    const gx2::Sampler garbage{.word0 = 0xFFFFFFFF, .word1 = 0xFFFFFFFF, .word2 = 0x7FFFFFFF};
    REQUIRE(gx2::write_guest(memory, kSampler, garbage));
    call("GX2InitSampler", {kSampler, kClampLastTexel, kFilterLinear});

    const latte::TexSampler sampler = guest();
    // Clamp 2 on all three axes, bilinear in and out, point Z and mip
    // filters, TEX_ARRAY_OVERRIDE.
    CHECK(sampler.word0.raw == 0x02029292);
    CHECK(sampler.word0.clamp_z == latte::TexClamp::ClampLastTexel);
    CHECK(sampler.word0.xy_min_filter == latte::TexXYFilter::Bilinear);
    CHECK(sampler.word0.mip_filter == latte::TexZFilter::Point);
    CHECK(sampler.word0.depth_compare_function == latte::CompareFunction::Never);
    // LOD 0 to 1023/64, no bias.
    CHECK(sampler.word1.raw == (1023U << 10));
    CHECK(sampler.min_lod() == 0.0F);
    CHECK(sampler.max_lod() == 1023.0F / 64.0F);
    CHECK(sampler.lod_bias() == 0.0F);
    // TYPE.
    CHECK(sampler.word2.raw == 0x80000000);
}

TEST_CASE_METHOD(SamplerFixture,
                 "each GX2InitSampler call changes its own fields",
                 "[hle][gx2][sampler]") {
    call("GX2InitSampler", {kSampler, kClampWrap, kFilterLinear});

    // Filters: as given without anisotropy; their anisotropic forms with it.
    call("GX2InitSamplerXYFilter", {kSampler, kFilterPoint, kFilterLinear, 0});
    CHECK(guest().word0.xy_mag_filter == latte::TexXYFilter::Point);
    CHECK(guest().word0.xy_min_filter == latte::TexXYFilter::Bilinear);
    CHECK(guest().word0.max_aniso_ratio == 0);
    call("GX2InitSamplerXYFilter", {kSampler, kFilterPoint, kFilterLinear, 2}); // 4:1
    CHECK(guest().word0.xy_mag_filter == latte::TexXYFilter::AnisoPoint);
    CHECK(guest().word0.xy_min_filter == latte::TexXYFilter::AnisoBilinear);
    CHECK(guest().word0.max_aniso_ratio == 2);

    call("GX2InitSamplerZMFilter", {kSampler, 2, 0});
    CHECK(guest().word0.z_filter == latte::TexZFilter::Linear);
    CHECK(guest().word0.mip_filter == latte::TexZFilter::None);

    call("GX2InitSamplerClamping", {kSampler, kClampWrap, kClampMirror, kClampBorder});
    CHECK(guest().word0.clamp_x == latte::TexClamp::Wrap);
    CHECK(guest().word0.clamp_y == latte::TexClamp::Mirror);
    CHECK(guest().word0.clamp_z == latte::TexClamp::ClampBorder);

    call("GX2InitSamplerBorderType", {kSampler, 2});
    CHECK(guest().word0.border_color_type == latte::TexBorderColor::OpaqueWhite);
    call("GX2InitSamplerDepthCompare", {kSampler, 3});
    CHECK(guest().word0.depth_compare_function == latte::CompareFunction::LessEqual);

    // None of that touched the filters set before it, or the other words.
    CHECK(guest().word0.xy_min_filter == latte::TexXYFilter::AnisoBilinear);
    CHECK(guest().word0.tex_array_override);
    CHECK(guest().word1.max_lod == 1023);
    CHECK(guest().word2.raw == 0x80000000);

    // LODs and bias, floored to sixty-fourths.
    call("GX2InitSamplerLOD", {kSampler}, {0.5F, 8.0F, -1.25F});
    CHECK(guest().word1.min_lod == 32);
    CHECK(guest().word1.max_lod == 512);
    CHECK(guest().lod_bias() == -1.25F);
    call("GX2InitSamplerLOD", {kSampler}, {0.0F, 16.0F, -0.01F});
    CHECK(guest().word1.max_lod == 1023);       // 16 does not fit: held
    CHECK(guest().lod_bias() == -1.0F / 64.0F); // floored, not truncated to zero
    // Held to the fields; NaN reads zero.
    call("GX2InitSamplerLOD", {kSampler}, {-1.0F, 20.0F, 40.0F});
    CHECK(guest().word1.min_lod == 0);
    CHECK(guest().word1.max_lod == 1023);
    CHECK(guest().word1.lod_bias == 2047);
    const f32 nan = std::numeric_limits<f32>::quiet_NaN();
    call("GX2InitSamplerLOD", {kSampler}, {nan, -3.0F, -40.0F});
    CHECK(guest().word1.min_lod == 0);
    CHECK(guest().word1.max_lod == 0);
    CHECK(guest().lod_bias() == -32.0F);
    CHECK(guest().word0.clamp_z == latte::TexClamp::ClampBorder);

    // decaf's three.
    call("GX2InitSamplerFilterAdjust", {kSampler, 1, 5, 2});
    CHECK(guest().word2.high_precision_filter);
    CHECK(guest().word2.perf_mip == 5);
    CHECK(guest().word2.perf_z == 2);
    call("GX2InitSamplerLODAdjust", {kSampler, 1}, {1.5F});
    CHECK(guest().word2.aniso_bias == 48);
    CHECK(guest().word0.lod_uses_minor_axis);
    call("GX2InitSamplerLODAdjust", {kSampler, 0}, {2.0F});
    CHECK(guest().word2.aniso_bias == 63); // 64 would wrap to 0: held
    CHECK_FALSE(guest().word0.lod_uses_minor_axis);
    call("GX2InitSamplerRoundingMode", {kSampler, 1});
    CHECK(guest().word2.truncate_coord);
    CHECK(guest().word2.type);
    CHECK(guest().word2.perf_mip == 5);
}

TEST_CASE_METHOD(SamplerFixture,
                 "GX2Set…Sampler writes the three words to the stage's sampler",
                 "[hle][gx2][sampler]") {
    call("GX2InitSampler", {kSampler, kClampBorder, kFilterPoint});
    call("GX2InitSamplerLOD", {kSampler}, {1.0F, 4.0F, 0.5F});
    call("GX2InitSampler", {kOther, kClampMirror, kFilterLinear});
    call("GX2SetPixelSampler", {kSampler, 3});
    call("GX2SetVertexSampler", {kSampler, 0});
    call("GX2SetGeometrySampler", {kSampler, 17});
    call("GX2SetComputeSampler", {kOther, 2}); // the vertex stage's, as Cemu has it
    run_gpu();

    const latte::TexSampler title = guest();
    for (const u32 id : {smp::ps_base + 3, smp::vs_base, smp::gs_base + 17}) {
        INFO("sampler " << id);
        CHECK(registers(id).word0.raw == title.word0.raw);
        CHECK(registers(id).word1.raw == title.word1.raw);
        CHECK(registers(id).word2.raw == title.word2.raw);
    }
    CHECK(registers(smp::ps_base + 3).max_lod() == 4.0F);
    CHECK(registers(smp::vs_base + 2).word0.clamp_x == latte::TexClamp::Mirror);
    CHECK(registers(smp::ps_base).word0.raw == 0);

    // The pixel stage's unit 18 would be the vertex stage's 0: refused.
    call("GX2SetPixelSampler", {kOther, 18});
    run_gpu();
    CHECK(registers(smp::vs_base).word0.raw == title.word0.raw);
}

TEST_CASE_METHOD(SamplerFixture,
                 "a sampler's border colour is four config registers of its stage",
                 "[hle][gx2][sampler]") {
    call("GX2SetPixelSamplerBorderColor", {2}, {0.25F, 0.5F, 0.75F, 1.0F});
    call("GX2SetVertexSamplerBorderColor", {17}, {1.0F, 0.0F, 0.0F, 0.5F});
    call("GX2SetGeometrySamplerBorderColor", {0}, {0.0F, 1.0F, 0.0F, 1.0F});
    call("GX2SetGeometrySamplerBorderColor", {18}, {9.0F, 9.0F, 9.0F, 9.0F}); // refused
    run_gpu();

    const latte::RegisterFile& file = os.gpu.registers();
    CHECK(latte::border_color(file, smp::ps_border_base, 2) ==
          std::array<f32, 4>{0.25F, 0.5F, 0.75F, 1.0F});
    CHECK(latte::border_color(file, smp::vs_border_base, 17) ==
          std::array<f32, 4>{1.0F, 0.0F, 0.0F, 0.5F});
    CHECK(latte::border_color(file, smp::gs_border_base, 0) ==
          std::array<f32, 4>{0.0F, 1.0F, 0.0F, 1.0F});
    // Linux's R600 register list: the pixel stage's sampler 2 at 0xA420,
    // the geometry stage's 0 at 0xA800.
    CHECK(file.read(0xA420) == 0x3E800000); // 0.25
    CHECK(file.read(0xA804) == 0x3F800000); // green, 1.0
    CHECK(latte::border_color(file, smp::gs_border_base, 18) == std::array<f32, 4>{});
    CHECK(os.gpu.processor().stats().errors == 0);
}
