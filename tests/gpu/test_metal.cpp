// Phase 7.2 / 7.3 — the Metal backend, on the machine that has Metal.
//
// Built only on Apple hosts, and the point of it is that it talks to the real
// GPU. Everything else in the GPU tests runs against the null backend, which
// proves the interface's rules but cannot prove that a device was created,
// that the capability probe reports the truth, or that a clear actually
// cleared.
//
// It needs no window. That is deliberate: a test that opened one could not
// run over ssh or in CI, and the headless path it uses instead is the same
// one the compatibility runner (10.5) will boot titles through.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/backend.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::gpu;

namespace {

/// A live Metal backend, or a failed test saying why.
[[nodiscard]] std::unique_ptr<GpuBackend> live_backend() {
    wiiu::log::set_all_levels(wiiu::log::Level::Off);

    std::unique_ptr<GpuBackend> backend = create_metal_backend();
    REQUIRE(backend != nullptr);

    std::string error;
    if (!backend->initialize(error)) {
        // The capability shortfall is the most useful thing this file can
        // say, so it goes in the failure rather than being swallowed by a
        // bare REQUIRE.
        FAIL("Metal backend did not initialize: " << error);
    }
    return backend;
}

} // namespace

TEST_CASE("the Metal device comes up and reports what it can do", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();
    const Capabilities& capabilities = backend->capabilities();

    INFO(capabilities.describe());

    CHECK_FALSE(capabilities.device_name.empty());
    CHECK(capabilities.sufficient());

    // ADR-002's floor. If this fails the emulator cannot run here at all,
    // and `initialize` above would already have refused -- so this is really
    // a check that the probe agrees with itself.
    CHECK(capabilities.metal3);
    CHECK(capabilities.mesh_shaders);

    // ADR-003 binds everything through argument buffers, which needs tier 2.
    CHECK(capabilities.argument_buffer_tier == 2);

    // The premise of P22's zero-copy guest memory. True on all Apple
    // silicon; if it is ever false here, that plan needs rethinking.
    CHECK(capabilities.unified_memory);
}

TEST_CASE("resources can be created on the real device", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    BufferDesc buffer_desc;
    buffer_desc.size = 1024;
    buffer_desc.storage = Storage::Shared;
    buffer_desc.label = "test buffer";
    const BufferHandle buffer = backend->create_buffer(buffer_desc);
    REQUIRE(buffer.valid());

    // Shared storage means the CPU can see the bytes directly, which is the
    // whole of P22 in one assertion.
    const std::array<u8, 4> bytes{1, 2, 3, 4};
    REQUIRE(backend->write_buffer(buffer, 0, bytes));
    const auto* const contents = static_cast<const u8*>(backend->buffer_contents(buffer));
    REQUIRE(contents != nullptr);
    CHECK(contents[2] == 3);

    TextureDesc texture_desc;
    texture_desc.width = 64;
    texture_desc.height = 64;
    texture_desc.format = TextureFormat::BGRA8Unorm;
    texture_desc.usage = TextureUsage::Sampled | TextureUsage::RenderTarget;
    texture_desc.label = "test texture";
    CHECK(backend->create_texture(texture_desc).valid());

    SamplerDesc sampler;
    sampler.label = "test sampler";
    CHECK(backend->create_sampler(sampler).valid());

    backend->destroy(buffer);
    backend->shutdown();
}

TEST_CASE("a clear really clears, on the GPU", "[gpu][metal]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    constexpr u32 kSize = 8;
    TextureDesc desc;
    desc.width = kSize;
    desc.height = kSize;
    desc.format = TextureFormat::BGRA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared; // so it can be read back
    desc.label = "offscreen";
    const TextureHandle target = backend->create_texture(desc);
    REQUIRE(target.valid());

    RenderPassDesc pass;
    pass.label = "offscreen clear";
    pass.color_count = 1;
    pass.colors[0].texture = target;
    pass.colors[0].load = LoadAction::Clear;
    pass.colors[0].store = StoreAction::Store;
    pass.colors[0].clear = ClearColor{1.0F, 0.0F, 0.0F, 1.0F};

    REQUIRE(backend->begin_render_pass(pass));
    backend->end_render_pass();
    backend->submit();

    std::vector<u8> pixels(static_cast<std::size_t>(kSize) * kSize * 4, 0);
    REQUIRE(backend->read_texture(target, pixels));

    // Opaque red, in BGRA order. This is the assertion the whole Metal path
    // exists for: a device was made, a command buffer was encoded, a render
    // pass ran on the GPU, and the result came back.
    CHECK(pixels[0] == 0x00);
    CHECK(pixels[1] == 0x00);
    CHECK(pixels[2] == 0xFF);
    CHECK(pixels[3] == 0xFF);

    const std::size_t last = pixels.size() - 4;
    CHECK(pixels[last + 2] == 0xFF);
    CHECK(pixels[last + 3] == 0xFF);

    backend->shutdown();
}

TEST_CASE("an unset store action is refused by the real backend too", "[gpu][metal][p20]") {
    const std::unique_ptr<GpuBackend> backend = live_backend();

    TextureDesc desc;
    desc.width = 8;
    desc.height = 8;
    desc.format = TextureFormat::BGRA8Unorm;
    desc.usage = TextureUsage::RenderTarget;
    desc.storage = Storage::Shared;

    RenderPassDesc pass;
    pass.color_count = 1;
    pass.colors[0].texture = backend->create_texture(desc);
    pass.colors[0].load = LoadAction::Clear;
    // store left Unspecified

    // P20 is enforced by the descriptor, so it holds for every backend rather
    // than being something the Metal one remembers to check.
    CHECK_FALSE(backend->begin_render_pass(pass));

    backend->shutdown();
}
