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
// What is here: a device and its capability probe (7.3); a `CAMetalLayer`
// swapchain with at most `frames_in_flight` frames queued (7.2); resources
// suballocated from heaps, and host memory wrapped in place (7.4, P22);
// pipelines compiled from MSL with errors worth reading, direct binding, and
// draws (7.6); and GPU capture (7.8). Fences are still loud refusals: they
// arrive with the GPU thread in 8.2, which is the only thing that will wait
// on one.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/backend.hpp>
#include <wiiu/gpu/metal/raii.hpp>
#include <wiiu/gpu/shader_diagnostics.hpp>

#include <fmt/format.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <objc/message.h>
#include <objc/runtime.h>
#include <semaphore>
#include <set>
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

[[nodiscard]] MTL::PrimitiveType to_metal(PrimitiveType type) noexcept {
    switch (type) {
        case PrimitiveType::Points:
            return MTL::PrimitiveTypePoint;
        case PrimitiveType::Lines:
            return MTL::PrimitiveTypeLine;
        case PrimitiveType::LineStrip:
            return MTL::PrimitiveTypeLineStrip;
        case PrimitiveType::TriangleStrip:
            return MTL::PrimitiveTypeTriangleStrip;
        case PrimitiveType::Triangles:
        case PrimitiveType::Quads: // refused by PipelineDesc::validate
        case PrimitiveType::Rects: // likewise: the mesh path (8.8)
            break;
    }
    return MTL::PrimitiveTypeTriangle;
}

/// The names of every function in a library, for the error that says the one
/// asked for is not among them.
[[nodiscard]] std::string function_names(const MTL::Library* library) {
    std::string names;
    const NS::Array* const list = library->functionNames();
    const NS::UInteger count = list == nullptr ? 0 : list->count();
    for (NS::UInteger index = 0; index < count; ++index) {
        if (!names.empty()) {
            names += ", ";
        }
        names += from_metal(list->object<NS::String>(index));
    }
    return names.empty() ? std::string{"none"} : names;
}

/// Is this device wrapped by Metal's API or shader validation layer?
///
/// The layers wrap the driver's device in objects of their own -- capture
/// outermost when it is enabled, then `MTLDebugDevice` (API validation) or
/// `MTLGPUDebugDevice` (shader validation) -- and each answers `baseObject`
/// with the one it wraps. So the chain is walked by class name, which does
/// not depend on how any of them chooses to describe itself.
[[nodiscard]] bool wrapped_by_debug_layer(MTL::Device* device) {
    if (from_metal(device->description()).find("Debug") != std::string::npos) {
        return true;
    }
    const SEL base_object = sel_registerName("baseObject");
    auto* object = reinterpret_cast<id>(device);
    for (int depth = 0; object != nullptr && depth < 8; ++depth) {
        const char* const name = object_getClassName(object);
        if (name != nullptr && std::string_view{name}.find("Debug") != std::string_view::npos) {
            return true;
        }
        if (!class_respondsToSelector(object_getClass(object), base_object)) {
            break;
        }
        object = reinterpret_cast<id (*)(id, SEL)>(objc_msgSend)(object, base_object);
    }
    return false;
}

[[nodiscard]] constexpr u64 align_up(u64 value, u64 alignment) noexcept {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

/// Phase 7.4 — resources suballocated from heaps.
///
/// ADR-003 binds through argument buffers, and the GPU does not know what an
/// argument buffer points at: every resource a pass might touch has to be
/// made resident by hand. Doing that per resource is a bookkeeping problem
/// with a GPU fault for a failure mode. Doing it per *heap* -- a handful of
/// `useHeap` calls an encoder -- is not, and that is what this is for. The
/// calls themselves arrive with the first argument buffer (8.12); until then
/// every resource reaches a shader through a direct binding, which Metal
/// makes resident on its own.
///
/// Two decisions worth knowing:
///
///   * **Heaps are hazard-tracked.** Untracked is the faster default and the
///     one Metal picks for heaps, and it makes every read-after-write between
///     passes the emulator's job, with `MTLFence`s. That is a P-list item for
///     when a profile asks for it, not a default to discover through
///     corruption. Tracking is per heap, so it is coarse: a write anywhere in
///     a heap orders against reads anywhere in it.
///   * **Memoryless targets are kept out of heaps** -- they occupy no memory,
///     so there is nothing to suballocate -- and so is anything a heap
///     refuses. Those are counted as dedicated, so a heap quietly failing
///     shows up as a number rather than as nothing.
class HeapPool {
public:
    /// 64 MiB at a time: big enough that a frame's render targets share one,
    /// small enough not to reserve memory a title never asks for.
    static constexpr u64 chunk_size = 64ULL << 20;

    void reset() {
        heaps_.clear();
        dedicated_ = 0;
        failed_ = {};
    }

    /// A buffer, owned by the caller (+1). `dedicated` says whether it had
    /// to be allocated on its own.
    [[nodiscard]] MTL::Buffer*
    new_buffer(MTL::Device* device, u64 size, Storage storage, bool& dedicated) {
        const MTL::ResourceOptions options = to_metal(storage);
        dedicated = false;
        if (storage != Storage::Memoryless) {
            const MTL::SizeAndAlign need = device->heapBufferSizeAndAlign(size, options);
            if (MTL::Heap* const heap = heap_with_room(device, storage, need); heap != nullptr) {
                if (MTL::Buffer* const buffer = heap->newBuffer(size, options); buffer != nullptr) {
                    return buffer;
                }
            }
        }
        MTL::Buffer* const buffer = device->newBuffer(size, options);
        if (buffer != nullptr) {
            dedicated = true;
            ++dedicated_;
        }
        return buffer;
    }

    /// A texture, owned by the caller (+1). `descriptor` already carries the
    /// storage mode.
    [[nodiscard]] MTL::Texture* new_texture(MTL::Device* device,
                                            MTL::TextureDescriptor* descriptor,
                                            Storage storage,
                                            bool& dedicated) {
        dedicated = false;
        if (storage != Storage::Memoryless) {
            const MTL::SizeAndAlign need = device->heapTextureSizeAndAlign(descriptor);
            if (MTL::Heap* const heap = heap_with_room(device, storage, need); heap != nullptr) {
                if (MTL::Texture* const texture = heap->newTexture(descriptor);
                    texture != nullptr) {
                    return texture;
                }
            }
        }
        MTL::Texture* const texture = device->newTexture(descriptor);
        if (texture != nullptr) {
            dedicated = true;
            ++dedicated_;
        }
        return texture;
    }

    void add_stats(MemoryStats& stats) const {
        stats.heaps = heaps_.size();
        for (const Heap& entry : heaps_) {
            stats.heap_bytes += entry.heap->size();
            stats.heap_used_bytes += entry.heap->usedSize();
        }
        stats.dedicated = dedicated_;
    }

    void forget_dedicated() noexcept {
        if (dedicated_ != 0) {
            --dedicated_;
        }
    }

private:
    struct Heap {
        Ref<MTL::Heap> heap;
        Storage storage = Storage::Private;
    };

    [[nodiscard]] MTL::Heap*
    heap_with_room(MTL::Device* device, Storage storage, const MTL::SizeAndAlign& need) {
        for (Heap& entry : heaps_) {
            if (entry.storage == storage && entry.heap->maxAvailableSize(need.align) >= need.size) {
                return entry.heap.get();
            }
        }
        bool& failed = failed_[static_cast<std::size_t>(storage)];
        if (failed) {
            return nullptr;
        }

        const Ref<MTL::HeapDescriptor> descriptor{MTL::HeapDescriptor::alloc()->init()};
        descriptor->setType(MTL::HeapTypeAutomatic);
        descriptor->setStorageMode(to_metal_storage_mode(storage));
        descriptor->setHazardTrackingMode(MTL::HazardTrackingModeTracked);
        descriptor->setSize(std::max(chunk_size, align_up(need.size, need.align)));

        Ref<MTL::Heap> heap{device->newHeap(descriptor.get())};
        if (!heap) {
            // Said once: after this every resource is dedicated, which works,
            // and the memory stats say so.
            failed = true;
            WIIU_LOG_WARN(Gpu,
                          "metal: could not create a {} heap; resources will be allocated "
                          "one by one",
                          storage_name(storage));
            return nullptr;
        }
        heap->setLabel(ns_string(fmt::format("{} heap {}", storage_name(storage), heaps_.size())));
        MTL::Heap* const raw = heap.get();
        heaps_.push_back(Heap{std::move(heap), storage});
        return raw;
    }

    std::vector<Heap> heaps_;
    u64 dedicated_ = 0;

    /// Per storage mode: a heap of that kind could not be made, so stop
    /// asking. Private and shared heaps fail independently.
    std::array<bool, 3> failed_{};
};

class MetalBackend final : public GpuBackend {
public:
    MetalBackend() = default;

    ~MetalBackend() override { shutdown(); }

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    [[nodiscard]] std::string_view name() const override { return "metal"; }

    [[nodiscard]] bool initialize(std::string& error) override;
    void shutdown() override;

    [[nodiscard]] const Capabilities& capabilities() const override { return capabilities_; }

    [[nodiscard]] bool attach_surface(const NativeSurface& surface, std::string& error) override;
    void set_vsync(bool enabled) override;
    [[nodiscard]] bool begin_frame(FrameInfo& out) override;
    void end_frame() override;
    void submit() override;
    void wait_idle() override;

    [[nodiscard]] FrameStats last_frame_stats() const override { return last_stats_; }

    [[nodiscard]] BufferHandle create_buffer(const BufferDesc& desc) override;
    [[nodiscard]] TextureHandle create_texture(const TextureDesc& desc) override;
    [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc& desc) override;
    [[nodiscard]] PipelineHandle create_pipeline(const PipelineDesc& desc,
                                                 std::string& error) override;

    void destroy(BufferHandle handle) override {
        buffers_.erase(handle.value());
        if (imported_.erase(handle.value()) != 0) {
            return;
        }
        if (dedicated_.erase(handle.value()) != 0) {
            heaps_.forget_dedicated();
        }
    }

    void destroy(TextureHandle handle) override {
        textures_.erase(handle.value());
        texture_formats_.erase(handle.value());
        if (dedicated_.erase(handle.value()) != 0) {
            heaps_.forget_dedicated();
        }
    }

    void destroy(SamplerHandle handle) override { samplers_.erase(handle.value()); }

    void destroy(PipelineHandle handle) override {
        const auto entry = pipelines_.find(handle.value());
        if (entry == pipelines_.end()) {
            return;
        }
        // The encoder keeps the state object alive for the pass; the pointer
        // to our record of it is the thing that must not dangle.
        if (pass_pipeline_ == &entry->second) {
            pass_pipeline_ = nullptr;
        }
        pipelines_.erase(entry);
    }

    void destroy(FenceHandle handle) override { static_cast<void>(handle); }

    [[nodiscard]] bool
    write_buffer(BufferHandle handle, u64 offset, std::span<const u8> data) override;
    [[nodiscard]] void* buffer_contents(BufferHandle handle) override;
    [[nodiscard]] bool read_texture(TextureHandle handle, std::span<u8> out) override;
    [[nodiscard]] bool write_texture(TextureHandle handle,
                                     const Rect& region,
                                     std::span<const u8> data,
                                     u32 bytes_per_row) override;
    [[nodiscard]] BufferHandle
    import_host_memory(void* pointer, u64 size, std::string_view label) override;
    [[nodiscard]] MemoryStats memory_stats() const override;

    [[nodiscard]] u64 command_errors() const override { return command_errors_->load(); }

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

    [[nodiscard]] bool begin_capture(std::string_view path, std::string& error) override;
    void end_capture() override;

private:
    void probe_capabilities();

    /// The texture behind a handle, including the swapchain's drawable, which
    /// is handed out as an ordinary texture handle so that a render pass
    /// targeting the screen and one targeting an offscreen buffer are the
    /// same code.
    [[nodiscard]] MTL::Texture* texture_for(TextureHandle handle) const;

    void refuse(std::string_view what);

    /// An encoding call made with no pass open. Logged once per call site,
    /// like `refuse`, because it is a frontend bug rather than a GPU one.
    [[nodiscard]] bool require_pass(std::string_view what);

    /// Open a command buffer if there is not one already.
    ///
    /// Encoding does not require a frame: the headless runner (10.5) renders
    /// with no swapchain at all. The buffer is retained rather than left
    /// autoreleased, because it has to outlive whichever pool happened to be
    /// open when it was made.
    [[nodiscard]] bool ensure_commands();

    /// Commit the open command buffer, optionally presenting first.
    void flush(bool present);

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
    Ref<MTL::CommandBuffer> commands_;
    Ref<MTL::CommandBuffer> submitted_; ///< the last one committed, for wait_idle
    /// Retained, rather than left to whichever autorelease pool was open
    /// when it was made: outside a frame (a headless render) there is none,
    /// and an autoreleased encoder there simply leaks.
    Ref<MTL::RenderCommandEncoder> encoder_;

    /// The size of the open pass's attachments, which a scissor may not
    /// reach past.
    u32 pass_width_ = 0;
    u32 pass_height_ = 0;

    /// Command buffers the GPU reported as failed: a fault, a timeout, a
    /// shader-validation error. Counted from the completion handler, on a
    /// Metal thread, hence shared and atomic.
    std::shared_ptr<std::atomic<u64>> command_errors_ = std::make_shared<std::atomic<u64>>(0);

    /// Frames between `begin_frame` and the GPU finishing them (7.2). Taken
    /// in `begin_frame`, given back by the command buffer's completion
    /// handler -- which runs on a Metal thread, and can run after this
    /// backend has started to go away, so the handler holds its own
    /// reference rather than a pointer to a member.
    using FrameSlots = std::counting_semaphore<frames_in_flight>;
    std::shared_ptr<FrameSlots> frame_slots_ = std::make_shared<FrameSlots>(frames_in_flight);

    /// Does the open command buffer belong to a frame, and so carry a slot
    /// that its completion has to give back?
    bool commands_hold_slot_ = false;

    struct Pipeline {
        Ref<MTL::RenderPipelineState> state;
        MTL::PrimitiveType primitive = MTL::PrimitiveTypeTriangle;
    };

    std::map<u32, Pipeline> pipelines_;

    /// The pipeline set in the open pass. A new encoder starts with none.
    const Pipeline* pass_pipeline_ = nullptr;

    HeapPool heaps_;
    std::set<u32> dedicated_;     ///< handles that did not come from a heap
    std::map<u32, u64> imported_; ///< handles wrapping host memory, and their sizes
    bool capturing_ = false;

    u32 next_id_ = 1;
    TextureHandle drawable_handle_;
    std::map<u32, Ref<MTL::Buffer>> buffers_;
    std::map<u32, Ref<MTL::Texture>> textures_;

    /// The format each texture was created with. Metal knows it too, but as
    /// a `MTLPixelFormat`, and the one question asked of it here -- how many
    /// bytes a pixel takes -- is answered by our own enum without a reverse
    /// mapping that would need updating twice.
    std::map<u32, TextureFormat> texture_formats_;
    std::map<u32, Ref<MTL::SamplerState>> samplers_;

    FrameStats stats_;
    FrameStats last_stats_;
    u64 frame_index_ = 0;

    /// Refusals already reported, so an unimplemented call in a draw loop
    /// says so once instead of sixty times a second.
    std::map<std::string, bool, std::less<>> reported_;
};

bool MetalBackend::require_pass(std::string_view what) {
    if (encoder_) {
        return true;
    }
    refuse(fmt::format("{} outside a render pass", what));
    return false;
}

void MetalBackend::refuse(std::string_view what) {
    const auto entry = reported_.find(what);
    if (entry != reported_.end()) {
        return;
    }
    reported_.emplace(std::string{what}, true);
    WIIU_LOG_WARN(Gpu, "metal: refused: {}", what);
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

    // The debug layer wraps the device in an `MTLDebugDevice` (or, with
    // shader validation, an `MTLGPUDebugDevice`), and says so in its
    // description. Asked of the object rather than of the environment,
    // because what matters is whether it took effect.
    capabilities_.api_validation = wrapped_by_debug_layer(device_.get());
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
    if (!device_) {
        return;
    }
    if (capturing_) {
        end_capture();
    }
    if (frame_pool_ != nullptr) {
        // A frame left open would keep its slot, and draining the slots
        // below would then wait for it forever.
        end_frame();
    }
    wait_idle();

    // Every frame's completion has given its slot back once all of them can
    // be taken. Nothing after this touches a command buffer.
    for (u32 slot = 0; slot < frames_in_flight; ++slot) {
        frame_slots_->acquire();
    }
    frame_slots_->release(frames_in_flight);

    commands_.reset();
    pass_pipeline_ = nullptr;
    pipelines_.clear();
    buffers_.clear();
    textures_.clear();
    texture_formats_.clear();
    samplers_.clear();
    dedicated_.clear();
    imported_.clear();
    heaps_.reset();
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
        texture_formats_.emplace(drawable_handle_.value(), TextureFormat::BGRA8Unorm);
    }

    WIIU_LOG_INFO(Gpu, "metal: surface attached, {}x{}", surface.width, surface.height);
    return true;
}

void MetalBackend::set_vsync(bool enabled) {
    if (layer_ != nullptr) {
        layer_->setDisplaySyncEnabled(enabled);
    }
}

bool MetalBackend::ensure_commands() {
    if (commands_) {
        return true;
    }
    if (!queue_) {
        return false;
    }
    const AutoreleasePool pool;
    commands_ = Ref<MTL::CommandBuffer>::retained(queue_->commandBuffer());
    return static_cast<bool>(commands_);
}

void MetalBackend::flush(bool present) {
    if (encoder_) {
        // An encoder left open is a hang on Metal rather than a wrong pixel,
        // so it is closed and complained about rather than ignored.
        WIIU_LOG_WARN(Gpu, "metal: a render pass was still open at submission");
        encoder_->endEncoding();
        encoder_.reset();
        pass_pipeline_ = nullptr;
    }
    if (!commands_) {
        return;
    }

    if (present && drawable_ != nullptr) {
        commands_->presentDrawable(drawable_);
    }
    // Runs on a Metal thread when the GPU is done with this buffer. It holds
    // its own references to what it touches, because the backend may be on
    // its way out by then.
    //
    // A frame's slot comes back here -- when the GPU is done with the frame,
    // not when the CPU is done describing it; that difference is what bounds
    // the frames in flight (7.2). And a failed buffer is counted, because
    // nothing else would notice: a GPU fault or a shader-validation error
    // leaves the CPU side looking exactly as if it had worked.
    commands_->addCompletedHandler([slots = commands_hold_slot_ ? frame_slots_ : nullptr,
                                    errors = command_errors_](MTL::CommandBuffer* buffer) {
        if (buffer->status() == MTL::CommandBufferStatusError) {
            errors->fetch_add(1);
        }
        if (slots) {
            slots->release();
        }
    });
    commands_hold_slot_ = false;
    commands_->commit();

    // Kept so `wait_idle` has something to wait on. Retained already, by
    // `ensure_commands`.
    submitted_ = commands_;
    commands_.reset();
}

void MetalBackend::submit() {
    flush(false);
}

void MetalBackend::wait_idle() {
    if (commands_) {
        flush(false);
    }
    if (submitted_) {
        submitted_->waitUntilCompleted();
        if (submitted_->status() == MTL::CommandBufferStatusError) {
            // Said here, on the thread that waited, rather than from the
            // completion handler on a Metal thread.
            const AutoreleasePool pool;
            const NS::Error* const failure = submitted_->error();
            WIIU_LOG_ERROR(Gpu,
                           "metal: a command buffer failed on the GPU: {}",
                           failure == nullptr ? std::string{"no reason given"}
                                              : from_metal(failure->localizedDescription()));
        }
        submitted_.reset();
    }
}

bool MetalBackend::begin_frame(FrameInfo& out) {
    if (layer_ == nullptr || frame_pool_ != nullptr) {
        return false;
    }

    // Work submitted outside a frame (a headless render, a readback) goes in
    // its own command buffer: the one this frame opens has to carry the slot.
    if (commands_) {
        flush(false);
    }

    // The pacing wait: at most `frames_in_flight` frames between here and
    // the GPU finishing them (7.2). Without it the CPU runs ahead until
    // `nextDrawable` blocks, and anything a frame writes from the CPU -- a
    // texture upload, a uniform -- can be overwritten while the GPU still
    // reads it. `nextDrawable` blocking counts as pacing too.
    const auto started = std::chrono::steady_clock::now();
    frame_slots_->acquire();

    // Opened here and drained in `end_frame`. Everything Metal hands back in
    // between -- the drawable, the descriptors, the encoder -- is
    // autoreleased into this.
    frame_pool_ = NS::AutoreleasePool::alloc()->init();

    drawable_ = layer_->nextDrawable();
    if (drawable_ == nullptr) {
        // No drawable is normal: the window is occluded, or being resized, or
        // all of them are still on screen. Skip the frame rather than
        // stalling on one, which is what waiting here would do.
        frame_pool_->release();
        frame_pool_ = nullptr;
        frame_slots_->release();
        return false;
    }

    if (!ensure_commands()) {
        frame_pool_->release();
        frame_pool_ = nullptr;
        drawable_ = nullptr;
        frame_slots_->release();
        return false;
    }
    commands_hold_slot_ = true;

    stats_ = FrameStats{};
    stats_.pacing_wait_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
    // The drawable's own size, not the surface's: during a resize the two
    // disagree for a frame, and a scissor computed from the wrong one runs
    // past the target, which API validation refuses.
    MTL::Texture* const target = drawable_->texture();
    out.drawable = drawable_handle_;
    out.width = static_cast<u32>(target->width());
    out.height = static_cast<u32>(target->height());
    out.format = TextureFormat::BGRA8Unorm;
    out.index = frame_index_;
    return true;
}

void MetalBackend::end_frame() {
    if (frame_pool_ == nullptr) {
        return;
    }

    flush(true);

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
    if (desc.storage == Storage::Memoryless) {
        // Only render targets can live in tile memory alone.
        WIIU_LOG_ERROR(Gpu, "metal: a buffer cannot be memoryless");
        return {};
    }
    AutoreleasePool pool;

    bool dedicated = false;
    Ref<MTL::Buffer> buffer{heaps_.new_buffer(device_.get(), desc.size, desc.storage, dedicated)};
    if (!buffer) {
        WIIU_LOG_ERROR(Gpu, "metal: could not allocate a {}-byte buffer", desc.size);
        return {};
    }
    if (!desc.label.empty()) {
        buffer->setLabel(ns_string(desc.label));
    }

    const u32 id = next_id_++;
    buffers_.emplace(id, std::move(buffer));
    if (dedicated) {
        dedicated_.insert(id);
    }
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

    bool dedicated = false;
    Ref<MTL::Texture> texture{
        heaps_.new_texture(device_.get(), descriptor, desc.storage, dedicated)};
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
    texture_formats_.emplace(id, desc.format);
    if (dedicated) {
        dedicated_.insert(id);
    }
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

bool MetalBackend::write_texture(TextureHandle handle,
                                 const Rect& region,
                                 std::span<const u8> data,
                                 u32 bytes_per_row) {
    MTL::Texture* const texture = texture_for(handle);
    if (texture == nullptr || handle == drawable_handle_) {
        return false;
    }
    if (texture->storageMode() != MTL::StorageModeShared) {
        refuse("write_texture on a non-shared texture (a blit, which belongs to 8.2)");
        return false;
    }

    const auto format = texture_formats_.find(handle.value());
    const u32 pixel = format == texture_formats_.end() ? 0 : format_bytes_per_pixel(format->second);
    const std::size_t row = static_cast<std::size_t>(region.width) * pixel;
    const std::size_t needed =
        region.height == 0 ? 0
                           : (static_cast<std::size_t>(bytes_per_row) * (region.height - 1)) + row;
    if (pixel == 0 || region.width == 0 || region.height == 0 ||
        region.x + region.width > texture->width() ||
        region.y + region.height > texture->height() || bytes_per_row < row ||
        data.size() < needed) {
        WIIU_LOG_ERROR(Gpu,
                       "metal: write_texture of {}x{} at ({},{}) from {} bytes, {} to a row, does "
                       "not fit a {}x{} texture",
                       region.width,
                       region.height,
                       region.x,
                       region.y,
                       data.size(),
                       bytes_per_row,
                       texture->width(),
                       texture->height());
        return false;
    }

    texture->replaceRegion(MTL::Region::Make2D(region.x, region.y, region.width, region.height),
                           0,
                           data.data(),
                           bytes_per_row);
    return true;
}

BufferHandle MetalBackend::import_host_memory(void* pointer, u64 size, std::string_view label) {
    if (!device_) {
        return {};
    }
    const u64 page = host_page_size();
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || size == 0 || (address % page) != 0 || (size % page) != 0) {
        WIIU_LOG_ERROR(Gpu,
                       "metal: cannot wrap {} bytes at {:#x}: Metal wants whole {}-byte pages",
                       size,
                       address,
                       page);
        return {};
    }

    AutoreleasePool pool;
    // The no-copy form, with no deallocator: the memory is the caller's and
    // outlives the buffer. (The three-argument `newBuffer` with a pointer
    // *copies*, which would be exactly the staging this exists to avoid.)
    Ref<MTL::Buffer> buffer{
        device_->newBuffer(pointer, size, MTL::ResourceStorageModeShared, nullptr)};
    if (!buffer) {
        WIIU_LOG_ERROR(Gpu, "metal: could not wrap {} bytes at {:#x}", size, address);
        return {};
    }
    if (!label.empty()) {
        buffer->setLabel(ns_string(label));
    }

    const u32 id = next_id_++;
    buffers_.emplace(id, std::move(buffer));
    imported_.emplace(id, size);
    return BufferHandle{id};
}

MemoryStats MetalBackend::memory_stats() const {
    MemoryStats stats;
    heaps_.add_stats(stats);
    stats.imported = imported_.size();
    for (const auto& [id, size] : imported_) {
        static_cast<void>(id);
        stats.imported_bytes += size;
    }
    return stats;
}

bool MetalBackend::read_texture(TextureHandle handle, std::span<u8> out) {
    MTL::Texture* const texture = texture_for(handle);
    if (texture == nullptr) {
        return false;
    }
    if (texture->storageMode() != MTL::StorageModeShared) {
        // A private texture has no CPU address. Copying one back needs a blit
        // into a staging buffer, which belongs with the GPU thread (8.2);
        // saying so beats returning a buffer of zeroes.
        refuse("read_texture on a non-shared texture");
        return false;
    }

    const auto format = texture_formats_.find(handle.value());
    const u32 pixel = format == texture_formats_.end() ? 0 : format_bytes_per_pixel(format->second);
    if (pixel == 0) {
        refuse("read_texture of a compressed texture");
        return false;
    }

    const auto width = static_cast<u32>(texture->width());
    const auto height = static_cast<u32>(texture->height());
    const std::size_t row = static_cast<std::size_t>(width) * pixel;
    if (out.size() < row * height) {
        WIIU_LOG_ERROR(Gpu,
                       "metal: {} bytes of room for a {}x{} readback needing {}",
                       out.size(),
                       width,
                       height,
                       row * height);
        return false;
    }

    // Whatever was drawn has to have happened before it can be read.
    wait_idle();

    const AutoreleasePool pool;
    texture->getBytes(out.data(), row, MTL::Region::Make2D(0, 0, width, height), 0);
    return true;
}

bool MetalBackend::begin_render_pass(const RenderPassDesc& desc) {
    if (encoder_) {
        return false;
    }
    if (!ensure_commands()) {
        return false;
    }
    if (const std::string problem = desc.validate(); !problem.empty()) {
        WIIU_LOG_ERROR(Gpu, "metal: {}", problem);
        return false;
    }

    // A pool of its own: outside a frame there is no other, and the
    // descriptor and the encoder both arrive autoreleased.
    const AutoreleasePool pool;
    MTL::RenderPassDescriptor* descriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    pass_width_ = 0;
    pass_height_ = 0;
    const auto note_size = [this](const MTL::Texture* texture, u32 level) {
        const auto width = static_cast<u32>(std::max<NS::UInteger>(texture->width() >> level, 1));
        const auto height = static_cast<u32>(std::max<NS::UInteger>(texture->height() >> level, 1));
        pass_width_ = pass_width_ == 0 ? width : std::min(pass_width_, width);
        pass_height_ = pass_height_ == 0 ? height : std::min(pass_height_, height);
    };

    for (u32 index = 0; index < desc.color_count; ++index) {
        const ColorAttachment& color = desc.colors[index];
        MTL::Texture* const texture = texture_for(color.texture);
        if (texture == nullptr) {
            WIIU_LOG_ERROR(Gpu, "metal: colour attachment {} names no live texture", index);
            return false;
        }

        MTL::RenderPassColorAttachmentDescriptor* const attachment =
            descriptor->colorAttachments()->object(index);
        note_size(texture, color.level);
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
        note_size(depth, 0);
        const auto format = texture_formats_.find(desc.depth_stencil.texture.value());
        const TextureFormat kind =
            format == texture_formats_.end() ? TextureFormat::Unknown : format->second;

        // A combined format goes on both attachments, a stencil-only one on
        // the stencil attachment alone: Metal checks the pass against the
        // pipeline's depth and stencil formats separately.
        if (format_has_depth(kind)) {
            MTL::RenderPassDepthAttachmentDescriptor* const attachment =
                descriptor->depthAttachment();
            attachment->setTexture(depth);
            attachment->setLoadAction(to_metal(desc.depth_stencil.load));
            attachment->setStoreAction(to_metal(desc.depth_stencil.store));
            attachment->setClearDepth(static_cast<double>(desc.depth_stencil.clear_depth));
        }
        if (format_has_stencil(kind)) {
            MTL::RenderPassStencilAttachmentDescriptor* const attachment =
                descriptor->stencilAttachment();
            attachment->setTexture(depth);
            attachment->setLoadAction(to_metal(desc.depth_stencil.load));
            attachment->setStoreAction(to_metal(desc.depth_stencil.store));
            attachment->setClearStencil(desc.depth_stencil.clear_stencil);
        }
    }

    encoder_ =
        Ref<MTL::RenderCommandEncoder>::retained(commands_->renderCommandEncoder(descriptor));
    if (!encoder_) {
        WIIU_LOG_ERROR(Gpu, "metal: could not create a render command encoder");
        return false;
    }
    if (!desc.label.empty()) {
        encoder_->setLabel(ns_string(desc.label));
    }
    pass_pipeline_ = nullptr;
    ++stats_.render_passes;
    return true;
}

void MetalBackend::end_render_pass() {
    if (!encoder_) {
        return;
    }
    encoder_->endEncoding();
    encoder_.reset();
    pass_pipeline_ = nullptr;
}

void MetalBackend::set_viewport(const Viewport& viewport) {
    if (!require_pass("set_viewport")) {
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
    if (!require_pass("set_scissor")) {
        return;
    }
    // Metal refuses a scissor that reaches past the attachments, and API
    // validation asserts on one. Clamped rather than refused: a scissor
    // computed from a window size one frame stale should cut, not fail.
    const NS::UInteger x = std::min(rect.x, pass_width_);
    const NS::UInteger y = std::min(rect.y, pass_height_);
    const NS::UInteger width = std::min<NS::UInteger>(rect.width, pass_width_ - x);
    const NS::UInteger height = std::min<NS::UInteger>(rect.height, pass_height_ - y);
    if (width == 0 || height == 0) {
        refuse("set_scissor with a rectangle entirely outside the attachments");
        return;
    }
    encoder_->setScissorRect(MTL::ScissorRect{x, y, width, height});
}

// --- 7.6 ------------------------------------------------------------------

PipelineHandle MetalBackend::create_pipeline(const PipelineDesc& desc, std::string& error) {
    error.clear();
    if (!device_) {
        error = "create_pipeline before initialize";
        return {};
    }
    if (std::string problem = desc.validate(); !problem.empty()) {
        error = std::move(problem);
        WIIU_LOG_ERROR(Gpu, "metal: pipeline '{}': {}", desc.label, error);
        return {};
    }

    AutoreleasePool pool;

    const Ref<MTL::CompileOptions> options{MTL::CompileOptions::alloc()->init()};
    // MSL 3.0 is the floor ADR-002 already sets (Metal 3, mesh shaders), and
    // pinning it keeps a newer SDK from changing what the same source means.
    options->setLanguageVersion(MTL::LanguageVersion3_0);
    // Fast math reassociates, contracts and assumes there are no NaNs. Latte
    // shaders are written for a GPU that did none of that, and the
    // decompiler (8.6) has to be able to say which multiply it meant.
    options->setFastMathEnabled(false);

    const auto compile = [&](const std::string& source, std::string_view which) {
        NS::Error* failure = nullptr;
        Ref<MTL::Library> library{device_->newLibrary(ns_string(source), options.get(), &failure)};
        const std::string messages =
            failure == nullptr ? std::string{} : from_metal(failure->localizedDescription());
        if (!library) {
            error = fmt::format("the {} shader did not compile:\n{}",
                                which,
                                annotate_shader_errors(source, messages));
        } else if (!messages.empty()) {
            // Compiled, with warnings. Worth seeing while the decompiler is
            // young; not worth failing for.
            WIIU_LOG_DEBUG(Gpu,
                           "metal: {} shader of '{}' compiled with warnings:\n{}",
                           which,
                           desc.label,
                           annotate_shader_errors(source, messages));
        }
        return library;
    };

    const Ref<MTL::Library> vertex_library = compile(desc.vertex_source, "vertex");
    if (!vertex_library) {
        WIIU_LOG_ERROR(Gpu, "metal: pipeline '{}': {}", desc.label, error);
        return {};
    }
    // One source for both stages is the common case; compiling it twice
    // would be twice the wait for the same library.
    Ref<MTL::Library> fragment_library = vertex_library;
    if (!desc.fragment_source.empty() && desc.fragment_source != desc.vertex_source) {
        fragment_library = compile(desc.fragment_source, "fragment");
        if (!fragment_library) {
            WIIU_LOG_ERROR(Gpu, "metal: pipeline '{}': {}", desc.label, error);
            return {};
        }
    }

    const Ref<MTL::Function> vertex{vertex_library->newFunction(ns_string(desc.vertex_entry))};
    if (!vertex) {
        error = fmt::format("no function '{}' in the vertex source (it has: {})",
                            desc.vertex_entry,
                            function_names(vertex_library.get()));
        WIIU_LOG_ERROR(Gpu, "metal: pipeline '{}': {}", desc.label, error);
        return {};
    }
    const Ref<MTL::Function> fragment{
        fragment_library->newFunction(ns_string(desc.fragment_entry))};
    if (!fragment) {
        error = fmt::format("no function '{}' in the fragment source (it has: {})",
                            desc.fragment_entry,
                            function_names(fragment_library.get()));
        WIIU_LOG_ERROR(Gpu, "metal: pipeline '{}': {}", desc.label, error);
        return {};
    }

    const Ref<MTL::RenderPipelineDescriptor> descriptor{
        MTL::RenderPipelineDescriptor::alloc()->init()};
    if (!desc.label.empty()) {
        descriptor->setLabel(ns_string(desc.label));
    }
    descriptor->setVertexFunction(vertex.get());
    descriptor->setFragmentFunction(fragment.get());
    for (u32 index = 0; index < desc.color_count; ++index) {
        descriptor->colorAttachments()->object(index)->setPixelFormat(
            to_metal(desc.color_formats[index]));
    }
    if (format_has_depth(desc.depth_format)) {
        descriptor->setDepthAttachmentPixelFormat(to_metal(desc.depth_format));
    }
    if (format_has_stencil(desc.depth_format)) {
        descriptor->setStencilAttachmentPixelFormat(to_metal(desc.depth_format));
    }
    descriptor->setRasterSampleCount(desc.sample_count);

    NS::Error* failure = nullptr;
    Ref<MTL::RenderPipelineState> state{
        device_->newRenderPipelineState(descriptor.get(), &failure)};
    if (!state) {
        error = fmt::format("the pipeline did not link: {}",
                            failure == nullptr ? std::string{"no reason given"}
                                               : from_metal(failure->localizedDescription()));
        WIIU_LOG_ERROR(Gpu, "metal: pipeline '{}': {}", desc.label, error);
        return {};
    }

    const u32 id = next_id_++;
    pipelines_.emplace(id, Pipeline{std::move(state), to_metal(desc.primitive)});
    return PipelineHandle{id};
}

void MetalBackend::set_pipeline(PipelineHandle pipeline) {
    if (!require_pass("set_pipeline")) {
        return;
    }
    const auto entry = pipelines_.find(pipeline.value());
    if (entry == pipelines_.end()) {
        refuse("set_pipeline with a handle that names no pipeline");
        return;
    }
    encoder_->setRenderPipelineState(entry->second.state.get());
    pass_pipeline_ = &entry->second;
}

// Direct binding, until ADR-003's argument buffers arrive with the MSL
// emitter (8.6, 8.12): index n is [[buffer(n)]], [[texture(n)]] or
// [[sampler(n)]] in the stage it is bound for. The interface already speaks
// in stages and indices, so the change will be behind these three functions.

void MetalBackend::bind_buffer(ShaderStage stage, u32 index, BufferHandle buffer, u64 offset) {
    if (!require_pass("bind_buffer")) {
        return;
    }
    const auto entry = buffers_.find(buffer.value());
    if (entry == buffers_.end() || index >= max_bound_buffers) {
        refuse("bind_buffer with an unknown handle or an index past the table");
        return;
    }
    MTL::Buffer* const resource = entry->second.get();
    switch (stage) {
        case ShaderStage::Vertex:
            encoder_->setVertexBuffer(resource, offset, index);
            return;
        case ShaderStage::Fragment:
            encoder_->setFragmentBuffer(resource, offset, index);
            return;
        case ShaderStage::Object:
            encoder_->setObjectBuffer(resource, offset, index);
            return;
        case ShaderStage::Mesh:
            encoder_->setMeshBuffer(resource, offset, index);
            return;
        case ShaderStage::Compute:
        case ShaderStage::Count:
            break;
    }
    refuse("bind_buffer to a compute stage inside a render pass");
}

void MetalBackend::bind_texture(ShaderStage stage, u32 index, TextureHandle texture) {
    if (!require_pass("bind_texture")) {
        return;
    }
    MTL::Texture* const resource = texture_for(texture);
    if (resource == nullptr || index >= max_bound_textures) {
        refuse("bind_texture with an unknown handle or an index past the table");
        return;
    }
    switch (stage) {
        case ShaderStage::Vertex:
            encoder_->setVertexTexture(resource, index);
            return;
        case ShaderStage::Fragment:
            encoder_->setFragmentTexture(resource, index);
            return;
        case ShaderStage::Object:
            encoder_->setObjectTexture(resource, index);
            return;
        case ShaderStage::Mesh:
            encoder_->setMeshTexture(resource, index);
            return;
        case ShaderStage::Compute:
        case ShaderStage::Count:
            break;
    }
    refuse("bind_texture to a compute stage inside a render pass");
}

void MetalBackend::bind_sampler(ShaderStage stage, u32 index, SamplerHandle sampler) {
    if (!require_pass("bind_sampler")) {
        return;
    }
    const auto entry = samplers_.find(sampler.value());
    if (entry == samplers_.end() || index >= max_bound_samplers) {
        refuse("bind_sampler with an unknown handle or an index past the table");
        return;
    }
    MTL::SamplerState* const resource = entry->second.get();
    switch (stage) {
        case ShaderStage::Vertex:
            encoder_->setVertexSamplerState(resource, index);
            return;
        case ShaderStage::Fragment:
            encoder_->setFragmentSamplerState(resource, index);
            return;
        case ShaderStage::Object:
            encoder_->setObjectSamplerState(resource, index);
            return;
        case ShaderStage::Mesh:
            encoder_->setMeshSamplerState(resource, index);
            return;
        case ShaderStage::Compute:
        case ShaderStage::Count:
            break;
    }
    refuse("bind_sampler to a compute stage inside a render pass");
}

void MetalBackend::draw(const DrawCall& call) {
    if (!require_pass("draw")) {
        return;
    }
    if (pass_pipeline_ == nullptr) {
        // An encoder with no pipeline state is a validation failure at best
        // and a GPU fault at worst; the null backend refuses the same thing.
        refuse("draw with no pipeline set in this pass");
        return;
    }

    if (!call.index_buffer.valid()) {
        if (call.vertex_count == 0) {
            return;
        }
        encoder_->drawPrimitives(pass_pipeline_->primitive,
                                 call.first_vertex,
                                 call.vertex_count,
                                 call.instance_count,
                                 call.first_instance);
        ++stats_.draws;
        return;
    }

    const auto entry = buffers_.find(call.index_buffer.value());
    const u64 index_size = call.index_is_32bit ? 4 : 2;
    if (entry == buffers_.end() || (call.index_offset % index_size) != 0) {
        refuse("an indexed draw with an unknown index buffer or a misaligned offset");
        return;
    }
    if (call.index_count == 0) {
        return;
    }
    encoder_->drawIndexedPrimitives(pass_pipeline_->primitive,
                                    call.index_count,
                                    call.index_is_32bit ? MTL::IndexTypeUInt32
                                                        : MTL::IndexTypeUInt16,
                                    entry->second.get(),
                                    call.index_offset,
                                    call.instance_count,
                                    static_cast<NS::Integer>(call.first_vertex),
                                    call.first_instance);
    ++stats_.draws;
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

// --- 7.8 ------------------------------------------------------------------

bool MetalBackend::begin_capture(std::string_view path, std::string& error) {
    if (!device_) {
        error = "begin_capture before initialize";
        return false;
    }
    if (capturing_) {
        error = "a capture is already running";
        return false;
    }

    AutoreleasePool pool;
    MTL::CaptureManager* const manager = MTL::CaptureManager::sharedCaptureManager();
    if (!manager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument)) {
        // Outside Xcode, capture is off unless the process asked for it
        // before Metal started -- and asking later does nothing.
        error = "this process cannot write GPU captures: set MTL_CAPTURE_ENABLED=1 in the "
                "environment before starting it";
        return false;
    }

    // Metal refuses to overwrite a trace, and a `.gputrace` is a directory,
    // so an old one would make every second capture of a session fail.
    const std::filesystem::path target{path};
    if (target.extension() != ".gputrace") {
        error = fmt::format("'{}' should end in .gputrace -- that is what Xcode opens", path);
        return false;
    }
    std::error_code ignored;
    std::filesystem::remove_all(target, ignored);

    const Ref<MTL::CaptureDescriptor> descriptor{MTL::CaptureDescriptor::alloc()->init()};
    descriptor->setCaptureObject(device_.get());
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(NS::URL::fileURLWithPath(ns_string(path)));

    NS::Error* failure = nullptr;
    if (!manager->startCapture(descriptor.get(), &failure)) {
        error = fmt::format("the capture did not start: {}",
                            failure == nullptr ? std::string{"no reason given"}
                                               : from_metal(failure->localizedDescription()));
        return false;
    }
    capturing_ = true;
    WIIU_LOG_INFO(Gpu, "metal: capturing to {}", path);
    return true;
}

void MetalBackend::end_capture() {
    if (!capturing_) {
        return;
    }
    // Whatever is still being recorded belongs in the capture.
    wait_idle();
    MTL::CaptureManager::sharedCaptureManager()->stopCapture();
    capturing_ = false;
    WIIU_LOG_INFO(Gpu, "metal: capture written");
}

} // namespace

std::unique_ptr<GpuBackend> create_metal_backend() {
    return std::make_unique<MetalBackend>();
}

} // namespace wiiu::gpu
