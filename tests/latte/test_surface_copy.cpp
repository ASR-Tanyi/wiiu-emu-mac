// Phase 8.13 (ADR-006) — the GPU's surface copy, what GX2's scan-out copy
// becomes for a surface the GPU holds.
//
// What is guarded here, on the null backend: the copy is a render pass of its
// own that overwrites its target (P20); an sRGB source takes the pipeline that
// encodes again, so bytes arrive as they were; the sizes reach the shader per
// copy, with `bind_bytes`, picture first and target second; and what it does
// not copy is refused before anything is encoded. That the shader keeps every
// byte and scales as the CPU copy does is a question for a GPU: the Metal
// tests ask it.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/surface_copy.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

struct CopyFixture {
    gpu::NullBackend backend;
    SurfaceCopier copier;

    CopyFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        std::string error;
        REQUIRE(backend.initialize(error));
        REQUIRE(copier.initialize(backend, error));
    }

    [[nodiscard]] gpu::TextureHandle
    texture(gpu::TextureFormat format, u32 width, u32 height, gpu::TextureUsage usage) {
        gpu::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.usage = usage;
        desc.label = "copy test";
        const gpu::TextureHandle handle = backend.create_texture(desc);
        REQUIRE(handle.valid());
        return handle;
    }

    /// The id of the pipeline made with this label, from the trace.
    [[nodiscard]] std::string pipeline(std::string_view label) const {
        for (const std::string& line : backend.events()) {
            if (line.starts_with("create_pipeline ") &&
                line.ends_with(std::string{"label="} + std::string{label})) {
                return line.substr(16, line.find(' ', 16) - 16);
            }
        }
        FAIL("no pipeline labelled " << label);
        return {};
    }
};

constexpr gpu::TextureUsage kSurface = gpu::TextureUsage::RenderTarget | gpu::TextureUsage::Sampled;

} // namespace

TEST_CASE("the copy's shader encodes again only for an sRGB source", "[latte][surface][copy]") {
    const std::string srgb = SurfaceCopier::shader_source(true);
    const std::string unorm = SurfaceCopier::shader_source(false);
    CHECK(srgb.find("constant bool source_is_srgb = true;") != std::string::npos);
    CHECK(unorm.find("constant bool source_is_srgb = false;") != std::string::npos);
    // The CPU copy's nearest pixel, in integers, and a read rather than a
    // filtered sample.
    CHECK(unorm.find("(pixel * sizes.source) / sizes.target") != std::string::npos);
    CHECK(unorm.find("source.read(from)") != std::string::npos);
    CHECK(unorm.find("vertex_main") != std::string::npos);
    CHECK(unorm.find("fragment_main") != std::string::npos);
}

TEST_CASE_METHOD(CopyFixture,
                 "a copy is a pass of its own over the whole target",
                 "[latte][surface][copy]") {
    // The GamePad's picture, 854 wide in a colour buffer padded to 864, into
    // a scan texture of its own size.
    const gpu::TextureHandle source =
        texture(gpu::TextureFormat::RGBA8UnormSrgb, 864, 480, kSurface);
    const gpu::TextureHandle target = texture(SurfaceCopier::target_format, 854, 480, kSurface);
    const std::string srgb_pipeline = pipeline("surface copy (sRGB)");
    const std::string unorm_pipeline = pipeline("surface copy");

    backend.clear_events();
    std::string error;
    REQUIRE(copier.copy(CopySource{source, gpu::TextureFormat::RGBA8UnormSrgb, 854, 480},
                        CopyTarget{target, 854, 480},
                        error));
    CHECK(backend.recorded("begin_render_pass 'surface copy' colors=1 [0:dont_care/store]"));
    CHECK(backend.recorded("set_pipeline " + srgb_pipeline));
    CHECK(backend.recorded("set_viewport 854x480 at (0,0)"));
    CHECK(backend.recorded("bind_texture fragment [0] = " + std::to_string(source.value())));
    CHECK(backend.recorded("bind_bytes fragment [0] 16 bytes"));
    CHECK(backend.recorded("draw vertices 3"));
    CHECK_FALSE(backend.in_render_pass());

    // Picture, then target, as the shader's CopySizes has them.
    std::array<u32, 4> sizes{};
    REQUIRE(backend.last_bytes().size() == sizeof(sizes));
    std::memcpy(sizes.data(), backend.last_bytes().data(), sizeof(sizes));
    CHECK(sizes == std::array<u32, 4>{854, 480, 854, 480});

    // A UNORM source takes the other pipeline.
    backend.clear_events();
    const gpu::TextureHandle plain = texture(gpu::TextureFormat::RGBA8Unorm, 64, 32, kSurface);
    REQUIRE(copier.copy(CopySource{plain, gpu::TextureFormat::RGBA8Unorm, 64, 32},
                        CopyTarget{target, 854, 480},
                        error));
    CHECK(backend.recorded("set_pipeline " + unorm_pipeline));
    std::memcpy(sizes.data(), backend.last_bytes().data(), sizeof(sizes));
    CHECK(sizes == std::array<u32, 4>{64, 32, 854, 480});
    CHECK(backend.rejected() == 0);
}

TEST_CASE_METHOD(CopyFixture,
                 "what the copy does not do is refused before anything is encoded",
                 "[latte][surface][copy]") {
    const gpu::TextureHandle source = texture(gpu::TextureFormat::BGRA8Unorm, 64, 64, kSurface);
    const gpu::TextureHandle target = texture(SurfaceCopier::target_format, 64, 64, kSurface);
    backend.clear_events();

    std::string error;
    CHECK_FALSE(copier.copy(CopySource{source, gpu::TextureFormat::BGRA8Unorm, 64, 64},
                            CopyTarget{target, 64, 64},
                            error));
    CHECK(error.find("is not done yet") != std::string::npos);

    CHECK_FALSE(copier.copy(CopySource{source, gpu::TextureFormat::RGBA8Unorm, 0, 64},
                            CopyTarget{target, 64, 64},
                            error));
    CHECK(error.find("copies nothing") != std::string::npos);

    SurfaceCopier idle;
    CHECK_FALSE(idle.copy(CopySource{source, gpu::TextureFormat::RGBA8Unorm, 64, 64},
                          CopyTarget{target, 64, 64},
                          error));
    CHECK(error.find("no pipelines") != std::string::npos);

    CHECK_FALSE(backend.recorded("begin_render_pass"));
    CHECK(backend.rejected() == 0);

    // Its pipelines go with it.
    copier.shutdown();
    CHECK(backend.live_count(gpu::ResourceKind::Pipeline) == 0);
}
