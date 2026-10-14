// Phase 7.1 — names, capability reporting, and descriptor validation.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/backend.hpp>

#include <fmt/format.h>

#include <unistd.h>
#include <vector>

namespace wiiu::gpu {

u64 host_page_size() noexcept {
    // Asked rather than assumed: 16 KiB on Apple silicon, 4 KiB on most
    // other hosts, and Metal rejects a wrapped range that is not whole pages.
    const long size = sysconf(_SC_PAGESIZE);
    return size > 0 ? static_cast<u64>(size) : 4096U;
}

std::string_view resource_kind_name(ResourceKind kind) noexcept {
    switch (kind) {
        case ResourceKind::Buffer:
            return "buffer";
        case ResourceKind::Texture:
            return "texture";
        case ResourceKind::Sampler:
            return "sampler";
        case ResourceKind::Pipeline:
            return "pipeline";
        case ResourceKind::Fence:
            return "fence";
    }
    return "?";
}

std::string_view shader_stage_name(ShaderStage stage) noexcept {
    switch (stage) {
        case ShaderStage::Vertex:
            return "vertex";
        case ShaderStage::Fragment:
            return "fragment";
        case ShaderStage::Object:
            return "object";
        case ShaderStage::Mesh:
            return "mesh";
        case ShaderStage::Compute:
            return "compute";
        case ShaderStage::Count:
            break;
    }
    return "?";
}

std::string_view texture_format_name(TextureFormat format) noexcept {
    switch (format) {
        case TextureFormat::Unknown:
            return "unknown";
        case TextureFormat::R8Unorm:
            return "r8_unorm";
        case TextureFormat::RG8Unorm:
            return "rg8_unorm";
        case TextureFormat::RGBA8Unorm:
            return "rgba8_unorm";
        case TextureFormat::RGBA8UnormSrgb:
            return "rgba8_unorm_srgb";
        case TextureFormat::BGRA8Unorm:
            return "bgra8_unorm";
        case TextureFormat::BGRA8UnormSrgb:
            return "bgra8_unorm_srgb";
        case TextureFormat::RGB10A2Unorm:
            return "rgb10a2_unorm";
        case TextureFormat::R16Float:
            return "r16_float";
        case TextureFormat::RGBA16Float:
            return "rgba16_float";
        case TextureFormat::RGBA32Float:
            return "rgba32_float";
        case TextureFormat::BC1Unorm:
            return "bc1_unorm";
        case TextureFormat::BC2Unorm:
            return "bc2_unorm";
        case TextureFormat::BC3Unorm:
            return "bc3_unorm";
        case TextureFormat::BC4Unorm:
            return "bc4_unorm";
        case TextureFormat::BC5Unorm:
            return "bc5_unorm";
        case TextureFormat::Depth32Float:
            return "depth32_float";
        case TextureFormat::Depth32FloatStencil8:
            return "depth32_float_stencil8";
        case TextureFormat::Stencil8:
            return "stencil8";
    }
    return "?";
}

bool format_has_depth(TextureFormat format) noexcept {
    return format == TextureFormat::Depth32Float || format == TextureFormat::Depth32FloatStencil8;
}

bool format_has_stencil(TextureFormat format) noexcept {
    return format == TextureFormat::Stencil8 || format == TextureFormat::Depth32FloatStencil8;
}

bool format_is_compressed(TextureFormat format) noexcept {
    switch (format) {
        case TextureFormat::BC1Unorm:
        case TextureFormat::BC2Unorm:
        case TextureFormat::BC3Unorm:
        case TextureFormat::BC4Unorm:
        case TextureFormat::BC5Unorm:
            return true;
        default:
            return false;
    }
}

u32 format_bytes_per_pixel(TextureFormat format) noexcept {
    switch (format) {
        case TextureFormat::R8Unorm:
        case TextureFormat::Stencil8:
            return 1;
        case TextureFormat::RG8Unorm:
        case TextureFormat::R16Float:
            return 2;
        case TextureFormat::RGBA8Unorm:
        case TextureFormat::RGBA8UnormSrgb:
        case TextureFormat::BGRA8Unorm:
        case TextureFormat::BGRA8UnormSrgb:
        case TextureFormat::RGB10A2Unorm:
        case TextureFormat::Depth32Float:
            return 4;
        case TextureFormat::RGBA16Float:
        case TextureFormat::Depth32FloatStencil8:
            return 8;
        case TextureFormat::RGBA32Float:
            return 16;
        case TextureFormat::BC1Unorm:
        case TextureFormat::BC2Unorm:
        case TextureFormat::BC3Unorm:
        case TextureFormat::BC4Unorm:
        case TextureFormat::BC5Unorm:
        case TextureFormat::Unknown:
            break;
    }
    // Compressed formats have no per-pixel size, and neither does Unknown.
    // Zero rather than a guess: a caller computing a row stride from this
    // should notice, not quietly allocate a quarter of what it needs.
    return 0;
}

std::string_view storage_name(Storage storage) noexcept {
    switch (storage) {
        case Storage::Private:
            return "private";
        case Storage::Shared:
            return "shared";
        case Storage::Memoryless:
            return "memoryless";
    }
    return "?";
}

std::string_view load_action_name(LoadAction action) noexcept {
    switch (action) {
        case LoadAction::Unspecified:
            return "unspecified";
        case LoadAction::DontCare:
            return "dont_care";
        case LoadAction::Load:
            return "load";
        case LoadAction::Clear:
            return "clear";
    }
    return "?";
}

std::string_view store_action_name(StoreAction action) noexcept {
    switch (action) {
        case StoreAction::Unspecified:
            return "unspecified";
        case StoreAction::DontCare:
            return "dont_care";
        case StoreAction::Store:
            return "store";
        case StoreAction::Resolve:
            return "resolve";
        case StoreAction::StoreAndResolve:
            return "store_and_resolve";
    }
    return "?";
}

std::string_view primitive_type_name(PrimitiveType type) noexcept {
    switch (type) {
        case PrimitiveType::Points:
            return "points";
        case PrimitiveType::Lines:
            return "lines";
        case PrimitiveType::LineStrip:
            return "line_strip";
        case PrimitiveType::Triangles:
            return "triangles";
        case PrimitiveType::TriangleStrip:
            return "triangle_strip";
        case PrimitiveType::Quads:
            return "quads";
        case PrimitiveType::Rects:
            return "rects";
    }
    return "?";
}

bool needs_primitive_emulation(PrimitiveType type) noexcept {
    // Quads become triangles in the translator; RECTS needs the object+mesh
    // path, because the fourth vertex does not exist until something derives
    // it (8.8).
    return type == PrimitiveType::Quads || type == PrimitiveType::Rects;
}

// --------------------------------------------------------------------------
// Capabilities
// --------------------------------------------------------------------------

std::string Capabilities::shortfall() const {
    std::vector<std::string> missing;

    if (!metal3) {
        missing.emplace_back("Metal 3 (macOS 13 or later)");
    }
    if (!mesh_shaders) {
        // ADR-002's reason for existing. There is no second geometry path, so
        // this is refusal rather than degradation.
        missing.emplace_back("mesh shaders (GPU family Apple7: M1, A14 or later)");
    }
    if (argument_buffer_tier < 2) {
        // ADR-003 binds everything this way; tier 1 cannot.
        missing.emplace_back("argument buffers tier 2");
    }

    if (missing.empty()) {
        return {};
    }

    std::string text = fmt::format("{} cannot run this emulator: it is missing ",
                                   device_name.empty() ? "this GPU" : device_name);
    for (std::size_t index = 0; index < missing.size(); ++index) {
        if (index != 0) {
            text += index + 1 == missing.size() ? " and " : ", ";
        }
        text += missing[index];
    }
    text += ".";
    return text;
}

std::string Capabilities::describe() const {
    return fmt::format("{} [{}]\n"
                       "  metal 3 ................ {}\n"
                       "  mesh shaders (Apple7) .. {}\n"
                       "  indirect mesh (Apple9) . {}\n"
                       "  programmable blending .. {}\n"
                       "  tile shaders ........... {}\n"
                       "  argument buffers ....... tier {}\n"
                       "  unified memory ......... {}\n"
                       "  max 2D texture ......... {}\n"
                       "  api validation ......... {}",
                       device_name.empty() ? "unknown device" : device_name,
                       api_version.empty() ? "unknown" : api_version,
                       metal3,
                       mesh_shaders,
                       indirect_mesh_draws,
                       programmable_blending,
                       tile_shaders,
                       argument_buffer_tier,
                       unified_memory,
                       max_texture_size_2d,
                       api_validation);
}

// --------------------------------------------------------------------------
// Descriptor validation
// --------------------------------------------------------------------------

namespace {

/// The **P20** check, in one place because it is the same for colour and
/// depth and because getting it wrong is invisible.
[[nodiscard]] std::string check_actions(std::string_view what, LoadAction load, StoreAction store) {
    if (load == LoadAction::Unspecified) {
        return fmt::format("{} has no load action. Choose one deliberately: on a tile-based "
                           "GPU, loading an attachment that did not need loading moves the "
                           "whole framebuffer through memory (P20)",
                           what);
    }
    if (store == StoreAction::Unspecified) {
        return fmt::format("{} has no store action. Choose one deliberately: storing an "
                           "attachment nothing will read is the same cost again (P20)",
                           what);
    }
    return {};
}

} // namespace

std::string RenderPassDesc::validate() const {
    if (color_count > max_color_attachments) {
        return fmt::format("{} colour attachments, but at most {} are allowed",
                           color_count,
                           max_color_attachments);
    }

    const bool has_depth = depth_stencil.texture.valid();
    if (color_count == 0 && !has_depth) {
        // A pass with nothing attached is almost always a translation bug.
        // The exception is stream-out, which runs with rasterization disabled
        // and no framebuffer at all (8.9) -- and that path does not come
        // through here.
        return "a render pass with no attachments at all";
    }

    for (u32 index = 0; index < color_count; ++index) {
        const ColorAttachment& color = colors[index];
        if (!color.texture.valid()) {
            return fmt::format("colour attachment {} has no texture", index);
        }
        if (std::string problem =
                check_actions(fmt::format("colour attachment {}", index), color.load, color.store);
            !problem.empty()) {
            return problem;
        }
        const bool resolving =
            color.store == StoreAction::Resolve || color.store == StoreAction::StoreAndResolve;
        if (resolving && !color.resolve_texture.valid()) {
            return fmt::format("colour attachment {} resolves, but has no resolve texture", index);
        }
    }

    if (has_depth) {
        if (std::string problem = check_actions(
                "the depth/stencil attachment", depth_stencil.load, depth_stencil.store);
            !problem.empty()) {
            return problem;
        }
    }

    return {};
}

std::string PipelineDesc::validate() const {
    if (vertex_source.empty()) {
        return "a pipeline with no vertex shader";
    }
    if (color_count > max_color_attachments) {
        return fmt::format("{} colour attachments, but at most {} are allowed",
                           color_count,
                           max_color_attachments);
    }
    for (u32 index = 0; index < color_count; ++index) {
        if (color_formats[index] == TextureFormat::Unknown) {
            return fmt::format("colour attachment {} has no format", index);
        }
    }
    if (needs_primitive_emulation(primitive)) {
        // Reaching the backend with one of these means the translator did not
        // do its job: quads are expanded to triangles and RECTS goes through
        // the object+mesh path (8.8). Neither is something a render pipeline
        // can be asked for directly.
        return fmt::format("primitive type '{}' cannot be drawn directly; it must be expanded "
                           "by the translator or go through the mesh path",
                           primitive_type_name(primitive));
    }
    return {};
}

} // namespace wiiu::gpu
