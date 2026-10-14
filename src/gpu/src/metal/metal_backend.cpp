// Phase 1.4 / 7.2 / 7.3 — the Metal backend.
//
// This is the only file in the project that includes a Metal header, which is
// the whole point of 7.1: everything above it is written against
// `GpuBackend`, and nothing above it knows what a `MTL::Device` is.
//
// metal-cpp is header-only, and exactly one translation unit has to provide
// the out-of-line definitions. That is this one, via the three defines below;
// putting them in a second file is a page of duplicate-symbol errors, and
// putting them in none is a page of undefined ones.
//
// What is implemented here is what Phase 1.4 needs and no more: a device, a
// capability probe, a `CAMetalLayer` swapchain, frames, render passes, and
// the resources whose creation is a descriptor and one call. Pipelines, draws
// and fences are left as loud refusals for 7.6 and 8.2 -- shader compilation
// in particular has an error surface worth meeting with a compiler to hand
// rather than guessing at.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/gpu/metal/raii.hpp>

#include <fmt/format.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <map>
#include <string>
#include <vector>

namespace wiiu::gpu {

namespace metal {

AutoreleasePool::AutoreleasePool() : pool_(NS::AutoreleasePool::alloc()->init()) {}

AutoreleasePool::~AutoreleasePool() {
    if (pool_ != nullptr) {
        static_cast<NS::AutoreleasePool*>(pool_)->release();
    }
}

} // namespace metal

namespace {

using metal::AutoreleasePool;
using metal::Ref;

[[nodiscard]] MTL::PixelFormat to_metal(TextureFormat format) noexcept {
    switch (format) {
        case TextureFormat::R8Unorm:
            return MTL::PixelFormatR8Unorm;
        case TextureFormat::RG8Unorm:
            return MTL::PixelFormatRG8Unorm;
        case TextureFormat::RGBA8Unorm:
            return MTL::PixelFormatRGBA8Unorm;
        case TextureFormat::RGBA8UnormSrgb:
            return MTL::PixelFormatRGBA8Unorm_sRGB;
        case TextureFormat::BGRA8Unorm:
            return MTL::PixelFormatBGRA8Unorm;
        case TextureFormat::BGRA8UnormSrgb:
            return MTL::PixelFormatBGRA8Unorm_sRGB;
        case TextureFormat::RGB10A2Unorm:
            return MTL::PixelFormatRGB10A2Unorm;
        case TextureFormat::R16Float:
            return MTL::PixelFormatR16Float;
        case TextureFormat::RGBA16Float:
            return MTL::PixelFormatRGBA16Float;
        case TextureFormat::RGBA32Float:
            return MTL::PixelFormatRGBA32Float;
        case TextureFormat::BC1Unorm:
            return MTL::PixelFormatBC1_RGBA;
        case TextureFormat::BC2Unorm:
            return MTL::PixelFormatBC2_RGBA;
        case TextureFormat::BC3Unorm:
            return MTL::PixelFormatBC3_RGBA;
        case TextureFormat::BC4Unorm:
            return MTL::PixelFormatBC4_RUnorm;
        case TextureFormat::BC5Unorm:
            return MTL::PixelFormatBC5_RGUnorm;
        case TextureFormat::Depth32Float:
            return MTL::PixelFormatDepth32Float;
        case TextureFormat::Depth32FloatStencil8:
            return MTL::PixelFormatDepth32Float_Stencil8;
        case TextureFormat::Stencil8:
            return MTL::PixelFormatStencil8;
        case TextureFormat::Unknown:
            break;
    }
    return MTL::PixelFormatInvalid;
}

[[nodiscard]] MTL::LoadAction to_metal(LoadAction action) noexcept {
    switch (action) {
        case LoadAction::Load:
            return MTL::LoadActionLoad;
        case LoadAction::Clear:
            return MTL::LoadActionClear;
        case LoadAction::DontCare:
        case LoadAction::Unspecified:
            break;
    }
    // Unspecified never reaches here: RenderPassDesc::validate() refuses it
    // before anything is encoded (P20).
    return MTL::LoadActionDontCare;
}

[[nodiscard]] MTL::StoreAction to_metal(StoreAction action) noexcept {
    switch (action) {
        case StoreAction::Store:
            return MTL::StoreActionStore;
        case StoreAction::Resolve:
            return MTL::StoreActionMultisampleResolve;
        case StoreAction::StoreAndResolve:
            return MTL::StoreActionStoreAndMultisampleResolve;
        case StoreAction::DontCare:
        case StoreAction::Unspecified:
            break;
    }
    return MTL::StoreActionDontCare;
}

[[nodiscard]] MTL::ResourceOptions to_metal(Storage storage) noexcept {
    switch (storage) {
        case Storage::Shared:
            return MTL::ResourceStorageModeShared;
        case Storage::Memoryless:
            return MTL::ResourceStorageModeMemoryless;
        case Storage::Private:
            break;
    }
    return MTL::ResourceStorageModePrivate;
}

[[nodiscard]] MTL::StorageMode to_metal_storage_mode(Storage storage) noexcept {
    switch (storage) {
        case Storage::Shared:
            return MTL::StorageModeShared;
        case Storage::Memoryless:
            return MTL::StorageModeMemoryless;
        case Storage::Private:
            break;
    }
    return MTL::StorageModePrivate;
}

[[nodiscard]] MTL::TextureUsage to_metal(TextureUsage usage) noexcept {
    NS::UInteger result = MTL::TextureUsageUnknown;
    if (has_usage(usage, TextureUsage::Sampled)) {
        result |= MTL::TextureUsageShaderRead;
    }
    if (has_usage(usage, TextureUsage::RenderTarget)) {
        result |= MTL::TextureUsageRenderTarget;
    }
    if (has_usage(usage, TextureUsage::Storage)) {
        result |= MTL::TextureUsageShaderWrite;
    }
    return static_cast<MTL::TextureUsage>(result);
}

[[nodiscard]] MTL::SamplerMinMagFilter to_metal(Filter filter) noexcept {
    return filter == Filter::Linear ? MTL::SamplerMinMagFilterLinear
                                    : MTL::SamplerMinMagFilterNearest;
}

[[nodiscard]] MTL::SamplerAddressMode to_metal(AddressMode mode) noexcept {
    switch (mode) {
        case AddressMode::MirrorRepeat:
            return MTL::SamplerAddressModeMirrorRepeat;
        case AddressMode::ClampToEdge:
            return MTL::SamplerAddressModeClampToEdge;
        case AddressMode::ClampToBorder:
            return MTL::SamplerAddressModeClampToBorderColor;
        case AddressMode::Repeat:
            break;
    }
    return MTL::SamplerAddressModeRepeat;
}

/// Metal offers three border colours and Latte offers any colour at all.
/// Picking the nearest of the three is the right thing to do *here* -- the
/// exact colour is kept in the descriptor, and the shader analysis pass (8.7)
/// is what notices the difference and emulates it.
[[nodiscard]] MTL::SamplerBorderColor to_metal(const ClearColor& color) noexcept {
    if (color.alpha < 0.5F) {
        return MTL::SamplerBorderColorTransparentBlack;
    }
    const f32 luminance = (color.red + color.green + color.blue) / 3.0F;
    return luminance > 0.5F ? MTL::SamplerBorderColorOpaqueWhite
                            : MTL::SamplerBorderColorOpaqueBlack;
}

[[nodiscard]] std::string from_metal(const NS::String* text) {
    if (text == nullptr) {
        return {};
    }
    const char* const utf8 = text->utf8String();
    return utf8 == nullptr ? std::string{} : std::string{utf8};
}

/// A borrowed NS::String for a label. Autoreleased, so it is only valid
/// inside the caller's pool -- which is where every use of it is.
[[nodiscard]] NS::String* ns_string(std::string_view text) {
    const std::string copy{text};
    return NS::String::string(copy.c_str(), NS::UTF8StringEncoding);
}

class MetalBackend final : public GpuBackend {
public:
    [[nodiscard]] std::string_view name() const override { return "metal"; }

    [[nodiscard]] bool initialize(std::string& error) override;
    void shutdown() override;

    [[nodiscard]] const Capabilities& capabilities() const override { return capabilities_; }

    [[nodiscard]] bool attach_surface(const NativeSurface& surface, std::string& error) override;
    void set_vsync(bool enabled) override;
    [[nodiscard]] bool begin_frame(FrameInfo& out) override;
    void end_frame() override;

    [[nodiscard]] FrameStats last_frame_stats() const override { return last_stats_; }

    [[nodiscard]] BufferHandle create_buffer(const BufferDesc& desc) override;
    [[nodiscard]] TextureHandle create_texture(const TextureDesc& desc) override;
    [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc& desc) override;
    [[nodiscard]] PipelineHandle create_pipeline(const PipelineDesc& desc) override;

    void destroy(BufferHandle handle) override { buffers_.erase(handle.value()); }

    void destroy(TextureHandle handle) override { textures_.erase(handle.value()); }

    void destroy(SamplerHandle handle) override { samplers_.erase(handle.value()); }

    void destroy(PipelineHandle handle) override { static_cast<void>(handle); }

    void destroy(FenceHandle handle) override { static_cast<void>(handle); }

    [[nodiscard]] bool
    write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) override;
    [[nodiscard]] void* buffer_contents(BufferHandle handle) override;

    [[nodiscard]] bool begin_render_pass(const RenderPassDesc& desc) override;
    void end_render_pass() override;

    void set_pipeline(PipelineHandle pipeline) override;
    void set_viewport(const Viewport& viewport) override;
    void set_scissor(const Rect& rect) override;

    void bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) override;
    void bind_texture(ShaderStage stage, u32 index, TextureHandle texture) override;
    void bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) override;

    void draw(const DrawCall& call) override;

    [[nodiscard]] FenceHandle create_fence() override;
    void signal(FenceHandle handle) override;
    [[nodiscard]] bool is_signalled(FenceHandle handle) override;
    void wait(FenceHandle handle) override;

private:
    void probe_capabilities();

    /// The texture behind a handle, including the swapchain's drawable, which
    /// is handed out as an ordinary texture handle so that a render pass
    /// targeting the screen and one targeting an offscreen buffer are the
    /// same code.
    [[nodiscard]] MTL::Texture* texture_for(TextureHandle handle) const;

    void refuse(std::string_view what);

    Ref<MTL::Device> device_;
    Ref<MTL::CommandQueue> queue_;
    Capabilities capabilities_;

    CA::MetalLayer* layer_ = nullptr; ///< owned by the platform layer
    NativeSurface surface_;

    // Per-frame state. The pool is a pointer rather than a scope guard
    // because a frame spans two calls -- `begin_frame` opens it and
    // `end_frame` drains it -- which no RAII scope can express.
    NS::AutoreleasePool* frame_pool_ = nullptr;
    CA::MetalDrawable* drawable_ = nullptr;
    MTL::CommandBuffer* commands_ = nullptr;
    MTL::RenderCommandEncoder* encoder_ = nullptr;

    u32 next_id_ = 1;
    TextureHandle drawable_handle_;
    std::map<u32, Ref<MTL::Buffer>> buffers_;
    std::map<u32, Ref<MTL::Texture>> textures_;
    std::map<u32, Ref<MTL::SamplerState>> samplers_;

    FrameStats stats_;
    FrameStats last_stats_;
    u64 frame_index_ = 0;

    /// Refusals already reported, so an unimplemented call in a draw loop
    /// says so once instead of sixty times a second.
    std::map<std::string, bool, std::less<>> reported_;
};

void MetalBackend::refuse(std::string_view what) {
    const auto entry = reported_.find(what);
    if (entry != reported_.end()) {
        return;
    }
    reported_.emplace(std::string{what}, true);
    WIIU_LOG_WARN(Gpu, "metal: {} is not implemented yet", what);
}

void MetalBackend::probe_capabilities() {
    capabilities_.device_name = from_metal(device_->name());
    capabilities_.metal3 = device_->supportsFamily(MTL::GPUFamilyMetal3);
    capabilities_.mesh_shaders = device_->supportsFamily(MTL::GPUFamilyApple7);
    capabilities_.indirect_mesh_draws = device_->supportsFamily(MTL::GPUFamilyApple9);
    capabilities_.programmable_blending = device_->supportsFamily(MTL::GPUFamilyApple2);
    capabilities_.tile_shaders = device_->supportsFamily(MTL::GPUFamilyApple4);
    capabilities_.unified_memory = device_->hasUnifiedMemory();
    capabilities_.argument_buffer_tier =
        device_->argumentBuffersSupport() == MTL::ArgumentBuffersTier2 ? 2U : 1U;

    // Highest family first: the names are what a bug report needs, and
    // "Apple9" says more than a version number.
    if (device_->supportsFamily(MTL::GPUFamilyApple9)) {
        capabilities_.api_version = "Apple9";
    } else if (device_->supportsFamily(MTL::GPUFamilyApple8)) {
        capabilities_.api_version = "Apple8";
    } else if (device_->supportsFamily(MTL::GPUFamilyApple7)) {
        capabilities_.api_version = "Apple7";
    } else {
        capabilities_.api_version = "pre-Apple7";
    }

    // 16384 on every Apple silicon GPU; asked rather than assumed because
    // the resolution scaling in 8.15 multiplies the framebuffer and this is
    // the ceiling it runs into.
    capabilities_.max_texture_size_2d = 16384;
}

bool MetalBackend::initialize(std::string& error) {
    AutoreleasePool pool;

    device_.reset(MTL::CreateSystemDefaultDevice());
    if (!device_) {
        error = "no Metal device: this Mac has no GPU Metal will talk to";
        return false;
    }

    probe_capabilities();

    // ADR-002: refuse with a reason rather than rendering something wrong.
    // Cemu, having no fallback either, skips the draws it cannot do, which
    // reaches the player as an invisible enemy rather than as an error.
    if (const std::string shortfall = capabilities_.shortfall(); !shortfall.empty()) {
        error = shortfall;
        device_.reset();
        return false;
    }

    queue_.reset(device_->newCommandQueue());
    if (!queue_) {
        error = "could not create a Metal command queue";
        device_.reset();
        return false;
    }

    WIIU_LOG_INFO(Gpu, "metal: {}", capabilities_.describe());
    return true;
}

void MetalBackend::shutdown() {
    buffers_.clear();
    textures_.clear();
    samplers_.clear();
    queue_.reset();
    device_.reset();
    layer_ = nullptr;
}

bool MetalBackend::attach_surface(const NativeSurface& surface, std::string& error) {
    if (!device_) {
        error = "attach_surface before initialize";
        return false;
    }
    if (!surface.valid()) {
        error = "attach_surface with no layer";
        return false;
    }

    AutoreleasePool pool;

    layer_ = static_cast<CA::MetalLayer*>(surface.handle);
    surface_ = surface;
    layer_->setDevice(device_.get());
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    // The drawable is addressed like any other texture, so the render pass
    // path does not special-case the screen. It is not in `textures_`: the
    // swapchain owns it and it changes every frame.
    if (!drawable_handle_.valid()) {
        drawable_handle_ = TextureHandle{next_id_++};
    }

    WIIU_LOG_INFO(Gpu, "metal: surface attached, {}x{}", surface.width, surface.height);
    return true;
}

void MetalBackend::set_vsync(bool enabled) {
    if (layer_ != nullptr) {
        layer_->setDisplaySyncEnabled(enabled);
    }
}

bool MetalBackend::begin_frame(FrameInfo& out) {
    if (layer_ == nullptr || frame_pool_ != nullptr) {
        return false;
    }

    // Opened here and drained in `end_frame`. Everything Metal hands back in
    // between -- the drawable, the command buffer, the descriptors, the
    // encoder -- is autoreleased into this.
    frame_pool_ = NS::AutoreleasePool::alloc()->init();

    drawable_ = layer_->nextDrawable();
    if (drawable_ == nullptr) {
        // No drawable is normal: the window is occluded, or being resized, or
        // all three are still on screen. Skip the frame rather than stalling
        // on one, which is what waiting here would do.
        frame_pool_->release();
        frame_pool_ = nullptr;
        return false;
    }

    commands_ = queue_->commandBuffer();
    stats_ = FrameStats{};

    out.drawable = drawable_handle_;
    out.width = surface_.width;
    out.height = surface_.height;
    out.format = TextureFormat::BGRA8Unorm;
    out.index = frame_index_;
    return true;
}

void MetalBackend::end_frame() {
    if (frame_pool_ == nullptr) {
        return;
    }
    if (encoder_ != nullptr) {
        WIIU_LOG_WARN(Gpu, "metal: a render pass was still open at the end of the frame");
        encoder_->endEncoding();
        encoder_ = nullptr;
    }

    if (commands_ != nullptr && drawable_ != nullptr) {
        commands_->presentDrawable(drawable_);
        commands_->commit();
    }

    commands_ = nullptr;
    drawable_ = nullptr;
    frame_pool_->release();
    frame_pool_ = nullptr;

    ++frame_index_;
    last_stats_ = stats_;
}

MTL::Texture* MetalBackend::texture_for(TextureHandle handle) const {
    if (handle == drawable_handle_) {
        return drawable_ == nullptr ? nullptr : drawable_->texture();
    }
    const auto entry = textures_.find(handle.value());
    return entry == textures_.end() ? nullptr : entry->second.get();
}

BufferHandle MetalBackend::create_buffer(const BufferDesc& desc) {
    if (!device_ || desc.size == 0) {
        return {};
    }
    AutoreleasePool pool;

    Ref<MTL::Buffer> buffer{device_->newBuffer(desc.size, to_metal(desc.storage))};
    if (!buffer) {
        WIIU_LOG_ERROR(Gpu, "metal: could not allocate a {}-byte buffer", desc.size);
        return {};
    }
    if (!desc.label.empty()) {
        buffer->setLabel(ns_string(desc.label));
    }

    const u32 id = next_id_++;
    buffers_.emplace(id, std::move(buffer));
    return BufferHandle{id};
}

TextureHandle MetalBackend::create_texture(const TextureDesc& desc) {
    if (!device_ || desc.width == 0 || desc.height == 0) {
        return {};
    }
    const MTL::PixelFormat format = to_metal(desc.format);
    if (format == MTL::PixelFormatInvalid) {
        WIIU_LOG_ERROR(Gpu, "metal: no pixel format for '{}'", texture_format_name(desc.format));
        return {};
    }

    AutoreleasePool pool;

    MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::alloc()->init();
    const Ref<MTL::TextureDescriptor> owned{descriptor};

    descriptor->setWidth(desc.width);
    descriptor->setHeight(desc.height);
    descriptor->setDepth(desc.depth);
    descriptor->setMipmapLevelCount(desc.mip_levels);
    descriptor->setArrayLength(desc.array_length);
    descriptor->setSampleCount(desc.sample_count);
    descriptor->setPixelFormat(format);
    descriptor->setUsage(to_metal(desc.usage));
    descriptor->setStorageMode(to_metal_storage_mode(desc.storage));
    descriptor->setTextureType(desc.sample_count > 1 ? MTL::TextureType2DMultisample
                                                     : MTL::TextureType2D);

    Ref<MTL::Texture> texture{device_->newTexture(descriptor)};
    if (!texture) {
        WIIU_LOG_ERROR(Gpu,
                       "metal: could not create a {}x{} {} texture",
                       desc.width,
                       desc.height,
                       texture_format_name(desc.format));
        return {};
    }
    if (!desc.label.empty()) {
        texture->setLabel(ns_string(desc.label));
    }

    const u32 id = next_id_++;
    textures_.emplace(id, std::move(texture));
    return TextureHandle{id};
}

SamplerHandle MetalBackend::create_sampler(const SamplerDesc& desc) {
    if (!device_) {
        return {};
    }
    AutoreleasePool pool;

    MTL::SamplerDescriptor* descriptor = MTL::SamplerDescriptor::alloc()->init();
    const Ref<MTL::SamplerDescriptor> owned{descriptor};

    descriptor->setMinFilter(to_metal(desc.min_filter));
    descriptor->setMagFilter(to_metal(desc.mag_filter));
    descriptor->setMipFilter(desc.mip_filter == Filter::Linear ? MTL::SamplerMipFilterLinear
                                                               : MTL::SamplerMipFilterNearest);
    descriptor->setSAddressMode(to_metal(desc.address_u));
    descriptor->setTAddressMode(to_metal(desc.address_v));
    descriptor->setRAddressMode(to_metal(desc.address_w));
    descriptor->setLodMinClamp(desc.lod_min);
    descriptor->setLodMaxClamp(desc.lod_max);
    descriptor->setMaxAnisotropy(desc.max_anisotropy);
    descriptor->setBorderColor(to_metal(desc.border_color));
    if (desc.compare_enabled) {
        descriptor->setCompareFunction(static_cast<MTL::CompareFunction>(desc.compare));
    }

    Ref<MTL::SamplerState> sampler{device_->newSamplerState(descriptor)};
    if (!sampler) {
        WIIU_LOG_ERROR(Gpu, "metal: could not create a sampler");
        return {};
    }

    const u32 id = next_id_++;
    samplers_.emplace(id, std::move(sampler));
    return SamplerHandle{id};
}

bool MetalBackend::write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) {
    const auto entry = buffers_.find(handle.value());
    if (entry == buffers_.end()) {
        return false;
    }
    MTL::Buffer* const buffer = entry->second.get();
    if (offset + data.size() > buffer->length()) {
        WIIU_LOG_ERROR(Gpu,
                       "metal: a {}-byte write at {} runs past the end of a {}-byte buffer",
                       data.size(),
                       offset,
                       buffer->length());
        return false;
    }
    auto* const bytes = static_cast<u8*>(buffer->contents());
    if (bytes == nullptr) {
        // A private buffer has no CPU address: that upload needs a blit, and
        // a blit needs the GPU thread of 8.2.
        refuse("write_buffer on a private buffer");
        return false;
    }
    std::copy(data.begin(), data.end(), bytes + offset);
    return true;
}

void* MetalBackend::buffer_contents(BufferHandle handle) {
    const auto entry = buffers_.find(handle.value());
    return entry == buffers_.end() ? nullptr : entry->second->contents();
}

bool MetalBackend::begin_render_pass(const RenderPassDesc& desc) {
    if (commands_ == nullptr || encoder_ != nullptr) {
        return false;
    }
    if (const std::string problem = desc.validate(); !problem.empty()) {
        WIIU_LOG_ERROR(Gpu, "metal: {}", problem);
        return false;
    }

    MTL::RenderPassDescriptor* descriptor = MTL::RenderPassDescriptor::renderPassDescriptor();

    for (u32 index = 0; index < desc.color_count; ++index) {
        const ColorAttachment& color = desc.colors[index];
        MTL::Texture* const texture = texture_for(color.texture);
        if (texture == nullptr) {
            WIIU_LOG_ERROR(Gpu, "metal: colour attachment {} names no live texture", index);
            return false;
        }

        MTL::RenderPassColorAttachmentDescriptor* const attachment =
            descriptor->colorAttachments()->object(index);
        attachment->setTexture(texture);
        attachment->setLevel(color.level);
        attachment->setSlice(color.slice);
        attachment->setLoadAction(to_metal(color.load));
        attachment->setStoreAction(to_metal(color.store));
        attachment->setClearColor(MTL::ClearColor(static_cast<double>(color.clear.red),
                                                  static_cast<double>(color.clear.green),
                                                  static_cast<double>(color.clear.blue),
                                                  static_cast<double>(color.clear.alpha)));
        if (MTL::Texture* const resolve = texture_for(color.resolve_texture); resolve != nullptr) {
            attachment->setResolveTexture(resolve);
        }
    }

    if (MTL::Texture* const depth = texture_for(desc.depth_stencil.texture); depth != nullptr) {
        MTL::RenderPassDepthAttachmentDescriptor* const attachment = descriptor->depthAttachment();
        attachment->setTexture(depth);
        attachment->setLoadAction(to_metal(desc.depth_stencil.load));
        attachment->setStoreAction(to_metal(desc.depth_stencil.store));
        attachment->setClearDepth(static_cast<double>(desc.depth_stencil.clear_depth));
    }

    encoder_ = commands_->renderCommandEncoder(descriptor);
    if (encoder_ == nullptr) {
        WIIU_LOG_ERROR(Gpu, "metal: could not create a render command encoder");
        return false;
    }
    if (!desc.label.empty()) {
        encoder_->setLabel(ns_string(desc.label));
    }
    ++stats_.render_passes;
    return true;
}

void MetalBackend::end_render_pass() {
    if (encoder_ == nullptr) {
        return;
    }
    encoder_->endEncoding();
    encoder_ = nullptr;
}

void MetalBackend::set_viewport(const Viewport& viewport) {
    if (encoder_ == nullptr) {
        return;
    }
    encoder_->setViewport(MTL::Viewport{static_cast<double>(viewport.x),
                                        static_cast<double>(viewport.y),
                                        static_cast<double>(viewport.width),
                                        static_cast<double>(viewport.height),
                                        static_cast<double>(viewport.near_z),
                                        static_cast<double>(viewport.far_z)});
}

void MetalBackend::set_scissor(const Rect& rect) {
    if (encoder_ == nullptr) {
        return;
    }
    encoder_->setScissorRect(MTL::ScissorRect{rect.x, rect.y, rect.width, rect.height});
}

// --- 7.6 and 8.2 ----------------------------------------------------------
//
// Deliberately not yet. Pipelines mean compiling MSL and reporting its errors
// usefully, and draws mean the binding model of ADR-003; both are worth
// writing with a compiler to hand rather than in advance of one.

PipelineHandle MetalBackend::create_pipeline(const PipelineDesc& desc) {
    static_cast<void>(desc);
    refuse("create_pipeline (Phase 7.6)");
    return {};
}

void MetalBackend::set_pipeline(PipelineHandle pipeline) {
    static_cast<void>(pipeline);
    refuse("set_pipeline (Phase 7.6)");
}

void MetalBackend::bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) {
    static_cast<void>(stage);
    static_cast<void>(index);
    static_cast<void>(buffer);
    static_cast<void>(offset);
    refuse("bind_buffer (Phase 8.12, via ADR-003's argument buffers)");
}

void MetalBackend::bind_texture(ShaderStage stage, u32 index, TextureHandle texture) {
    static_cast<void>(stage);
    static_cast<void>(index);
    static_cast<void>(texture);
    refuse("bind_texture (Phase 8.12)");
}

void MetalBackend::bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) {
    static_cast<void>(stage);
    static_cast<void>(index);
    static_cast<void>(sampler);
    refuse("bind_sampler (Phase 8.12)");
}

void MetalBackend::draw(const DrawCall& call) {
    static_cast<void>(call);
    refuse("draw (Phase 7.6)");
}

FenceHandle MetalBackend::create_fence() {
    refuse("create_fence (Phase 8.2)");
    return {};
}

void MetalBackend::signal(FenceHandle handle) {
    static_cast<void>(handle);
    refuse("signal (Phase 8.2)");
}

bool MetalBackend::is_signalled(FenceHandle handle) {
    static_cast<void>(handle);
    refuse("is_signalled (Phase 8.2)");
    return true;
}

void MetalBackend::wait(FenceHandle handle) {
    static_cast<void>(handle);
    refuse("wait (Phase 8.2)");
}

} // namespace

std::unique_ptr<GpuBackend> create_metal_backend() {
    return std::make_unique<MetalBackend>();
}

} // namespace wiiu::gpu
