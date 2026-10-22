// Phase 8.13 (ADR-006) — see surface_copy.hpp.

#include <wiiu/latte/surface_copy.hpp>

#include <fmt/format.h>

#include <array>
#include <cstring>
#include <span>

namespace wiiu::latte {

namespace {

// One triangle over the whole target, as the presenter draws (gpu/presenter);
// nothing is bound for its vertices. The fragment reads its source pixel by
// integer coordinates, so no sampler and no filter come into it.
constexpr std::string_view shader_template = R"msl(
#include <metal_stdlib>
using namespace metal;

constant bool source_is_srgb = {};

struct CopyVertex {{
    float4 position [[position]];
}};

/// The picture's size in the source, then the target's.
struct CopySizes {{
    uint2 source;
    uint2 target;
}};

vertex CopyVertex vertex_main(uint id [[vertex_id]]) {{
    const float2 corner = float2(float((id << 1) & 2u), float(id & 2u));
    CopyVertex out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}}

/// The standard curve, linear to sRGB: what an sRGB texture's decode undoes.
float3 srgb_encode(float3 value) {{
    const float3 c = saturate(value);
    const float3 low = c * 12.92;
    const float3 high = 1.055 * powr(c, float3(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= 0.0031308);
}}

fragment float4 fragment_main(CopyVertex in [[stage_in]],
                              texture2d<float, access::read> source [[texture(0)]],
                              constant CopySizes& sizes [[buffer(0)]]) {{
    // A pixel's centre is (x + 0.5, y + 0.5), so its integer part is (x, y);
    // and the CPU copy's nearest pixel is x * sw / tw, in integers.
    const uint2 pixel = uint2(in.position.xy);
    const uint2 from = (pixel * sizes.source) / sizes.target;
    float4 colour = source.read(from);
    if (source_is_srgb) {{
        colour.rgb = srgb_encode(colour.rgb);
    }}
    return colour;
}}
)msl";

/// Metal's limit on a 2D texture's side on Apple GPUs; the sizes multiply
/// in 32 bits in the shader, and 16384 squared is well inside them.
constexpr u32 max_side = 16384;

} // namespace

std::string SurfaceCopier::shader_source(bool srgb_source) {
    return fmt::format(shader_template, srgb_source ? "true" : "false");
}

bool SurfaceCopier::initialize(gpu::GpuBackend& backend, std::string& error) {
    shutdown();

    for (const bool srgb : {false, true}) {
        gpu::PipelineDesc desc;
        desc.label = srgb ? "surface copy (sRGB)" : "surface copy";
        desc.vertex_source = shader_source(srgb);
        desc.color_formats[0] = target_format;
        desc.color_count = 1;
        std::string compile_error;
        const gpu::PipelineHandle pipeline = backend.create_pipeline(desc, compile_error);
        if (!pipeline.valid()) {
            error = fmt::format("the {} pipeline did not build: {}", desc.label, compile_error);
            if (unorm_.valid()) {
                backend.destroy(unorm_);
                unorm_ = {};
            }
            return false;
        }
        (srgb ? srgb_ : unorm_) = pipeline;
    }
    backend_ = &backend;
    return true;
}

void SurfaceCopier::shutdown() {
    if (backend_ == nullptr) {
        return;
    }
    for (gpu::PipelineHandle* pipeline : {&unorm_, &srgb_}) {
        if (pipeline->valid()) {
            backend_->destroy(*pipeline);
            *pipeline = {};
        }
    }
    backend_ = nullptr;
}

bool SurfaceCopier::copy(const CopySource& source, const CopyTarget& target, std::string& error) {
    if (backend_ == nullptr) {
        error = "the surface copier has no pipelines";
        return false;
    }
    gpu::PipelineHandle pipeline;
    switch (source.format) {
        case gpu::TextureFormat::RGBA8Unorm:
            pipeline = unorm_;
            break;
        case gpu::TextureFormat::RGBA8UnormSrgb:
            pipeline = srgb_;
            break;
        default:
            error = fmt::format("a surface copy from {} is not done yet",
                                gpu::texture_format_name(source.format));
            return false;
    }
    if (source.width == 0 || source.height == 0 || target.width == 0 || target.height == 0 ||
        source.width > max_side || source.height > max_side || target.width > max_side ||
        target.height > max_side) {
        error = fmt::format("a surface copy of {}x{} into {}x{} copies nothing",
                            source.width,
                            source.height,
                            target.width,
                            target.height);
        return false;
    }

    gpu::RenderPassDesc pass;
    pass.label = "surface copy";
    pass.color_count = 1;
    pass.colors[0].texture = target.texture;
    // Every pixel is written, so what was there is never needed (P20).
    pass.colors[0].load = gpu::LoadAction::DontCare;
    pass.colors[0].store = gpu::StoreAction::Store;
    if (!backend_->begin_render_pass(pass)) {
        error = "the backend refused the surface copy's render pass";
        return false;
    }

    const std::array<u32, 4> sizes{source.width, source.height, target.width, target.height};
    std::array<u8, sizeof(sizes)> bytes{};
    std::memcpy(bytes.data(), sizes.data(), sizeof(sizes));

    backend_->set_pipeline(pipeline);
    backend_->set_viewport(gpu::Viewport{
        0.0F, 0.0F, static_cast<f32>(target.width), static_cast<f32>(target.height), 0.0F, 1.0F});
    backend_->set_scissor(gpu::Rect{0, 0, target.width, target.height});
    backend_->bind_texture(gpu::ShaderStage::Fragment, 0, source.texture);
    backend_->bind_bytes(gpu::ShaderStage::Fragment, 0, bytes);
    gpu::DrawCall call;
    call.vertex_count = 3;
    backend_->draw(call);
    backend_->end_render_pass();
    return true;
}

} // namespace wiiu::latte
