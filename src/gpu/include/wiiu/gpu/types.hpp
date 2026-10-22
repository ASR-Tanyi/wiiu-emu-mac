// Phase 7.1 — the vocabulary the GPU interface is written in.
//
// Deliberately not Metal's vocabulary, and deliberately not Latte's. Metal
// types must not cross the backend boundary (that is the whole point of 7.1),
// and Latte's encodings belong in the Phase 8 translator, not in an interface
// a second backend might one day implement.
//
// Everything here is a plain value. Resources are opaque integer handles
// rather than pointers, which costs nothing and buys two things Phase 11 and
// Phase 8 both want: a handle can be written to a save state or an API trace
// and read back, and a stale one is a lookup failure rather than a crash.
#pragma once

#include <wiiu/common/types.hpp>

#include <array>
#include <compare>
#include <string_view>

namespace wiiu::gpu {

enum class ResourceKind : u8 {
    Buffer,
    Texture,
    Sampler,
    Pipeline,
    Fence,
};

/// An opaque reference to something a backend owns.
///
/// Zero is never valid, so a default-constructed handle names nothing and a
/// zeroed structure does not accidentally refer to the first resource
/// created.
template <ResourceKind Kind>
class Handle {
public:
    static constexpr ResourceKind kind = Kind;

    constexpr Handle() noexcept = default;

    explicit constexpr Handle(u32 value) noexcept : value_(value) {}

    [[nodiscard]] constexpr u32 value() const noexcept { return value_; }

    [[nodiscard]] constexpr bool valid() const noexcept { return value_ != 0; }

    explicit constexpr operator bool() const noexcept { return valid(); }

    [[nodiscard]] constexpr bool operator==(const Handle&) const noexcept = default;

private:
    u32 value_ = 0;
};

using BufferHandle = Handle<ResourceKind::Buffer>;
using TextureHandle = Handle<ResourceKind::Texture>;
using SamplerHandle = Handle<ResourceKind::Sampler>;
using PipelineHandle = Handle<ResourceKind::Pipeline>;
using FenceHandle = Handle<ResourceKind::Fence>;

[[nodiscard]] std::string_view resource_kind_name(ResourceKind kind) noexcept;

/// The programmable stages this emulator has to produce.
///
/// `Object` and `Mesh` are the pair that stands in for Latte's geometry
/// shader, and for its RECTS primitive, which is not an exotic case (8.8).
enum class ShaderStage : u8 {
    Vertex,
    Fragment,
    Object,
    Mesh,
    Compute,
    Count,
};

inline constexpr std::size_t shader_stage_count = static_cast<std::size_t>(ShaderStage::Count);

[[nodiscard]] std::string_view shader_stage_name(ShaderStage stage) noexcept;

/// Pixel formats, as the emulator needs them rather than as Latte encodes
/// them. Phase 8.5 maps the hardware's much longer list onto these, and adds
/// to it when a title turns up wanting something that is not here.
enum class TextureFormat : u8 {
    Unknown = 0,
    R8Unorm,
    RG8Unorm,
    RGBA8Unorm,
    RGBA8UnormSrgb,
    BGRA8Unorm,
    BGRA8UnormSrgb,
    RGB10A2Unorm,
    R16Float,
    RGBA16Float,
    RGBA32Float,
    BC1Unorm,
    BC2Unorm,
    BC3Unorm,
    BC4Unorm,
    BC5Unorm,
    Depth32Float,
    Depth32FloatStencil8,
    Stencil8,
};

[[nodiscard]] std::string_view texture_format_name(TextureFormat format) noexcept;
[[nodiscard]] bool format_has_depth(TextureFormat format) noexcept;
[[nodiscard]] bool format_has_stencil(TextureFormat format) noexcept;
[[nodiscard]] bool format_is_compressed(TextureFormat format) noexcept;

/// Bytes per pixel, or zero for a compressed format — which has bytes per
/// *block* and has to be asked a different question.
[[nodiscard]] u32 format_bytes_per_pixel(TextureFormat format) noexcept;

/// Where a resource lives.
///
/// `Shared` is the interesting one on this hardware: CPU and GPU address the
/// same bytes, which is what makes the zero-copy guest memory of **P22**
/// possible at all. `Memoryless` is the other (**P21**): a depth or MSAA
/// target that is never read back never needs to exist in memory, only in
/// tile storage.
enum class Storage : u8 {
    Private,
    Shared,
    Memoryless,
};

[[nodiscard]] std::string_view storage_name(Storage storage) noexcept;

enum class TextureUsage : u8 {
    None = 0,
    Sampled = 1U << 0,
    RenderTarget = 1U << 1,
    Storage = 1U << 2,
};

[[nodiscard]] constexpr TextureUsage operator|(TextureUsage left, TextureUsage right) noexcept {
    return static_cast<TextureUsage>(static_cast<u8>(left) | static_cast<u8>(right));
}

[[nodiscard]] constexpr bool has_usage(TextureUsage set, TextureUsage wanted) noexcept {
    return (static_cast<u8>(set) & static_cast<u8>(wanted)) != 0;
}

/// What happens to an attachment at the start of a render pass.
///
/// **There is no default.** `Unspecified` is the zero value and every backend
/// refuses it, which is the point: Apple GPUs are tile-based deferred, and an
/// attachment that loads and stores when it did not need to moves the whole
/// framebuffer through memory twice per pass (**P20**, worth 10–30% of GPU
/// time). The wrong choice is still *correct* — the image is right, it is
/// merely slower — so it would never arrive as a bug report. Making the
/// omission impossible is cheaper than remembering.
enum class LoadAction : u8 {
    Unspecified = 0,
    DontCare,
    Load,
    Clear,
};

enum class StoreAction : u8 {
    Unspecified = 0,
    DontCare,
    Store,
    Resolve,
    StoreAndResolve,
};

[[nodiscard]] std::string_view load_action_name(LoadAction action) noexcept;
[[nodiscard]] std::string_view store_action_name(StoreAction action) noexcept;

enum class Filter : u8 {
    Nearest,
    Linear,
};

enum class AddressMode : u8 {
    Repeat,
    MirrorRepeat,
    ClampToEdge,
    ClampToBorder,
};

enum class CompareFunction : u8 {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

/// Primitive topology.
///
/// `Rects` is Latte's RECTS: three vertices describing a parallelogram, with
/// the fourth derived. Metal has no such thing, which is why the object+mesh
/// path (8.8) is needed for ordinary titles and not only for the ones that
/// use real geometry shaders. `Quads` is the other Latte-ism, expanded to
/// triangles by the translator rather than by the backend.
enum class PrimitiveType : u8 {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    Quads,
    Rects,
};

[[nodiscard]] std::string_view primitive_type_name(PrimitiveType type) noexcept;

/// True for the topologies no Metal pipeline can draw directly, so the
/// translator knows it has to go through the mesh path or rewrite indices.
[[nodiscard]] bool needs_primitive_emulation(PrimitiveType type) noexcept;

struct ClearColor {
    f32 red = 0.0F;
    f32 green = 0.0F;
    f32 blue = 0.0F;
    f32 alpha = 1.0F;

    [[nodiscard]] bool operator==(const ClearColor&) const noexcept = default;
};

/// A blending factor: Metal's set, in Metal's order. Latte has these and
/// two more, D3D9's "both" factors, which the translator has to express in
/// these (8.11).
enum class BlendFactor : u8 {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationColor,
    OneMinusDestinationColor,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceAlphaSaturated,
    BlendColor,
    OneMinusBlendColor,
    BlendAlpha,
    OneMinusBlendAlpha,
    Source1Color,
    OneMinusSource1Color,
    Source1Alpha,
    OneMinusSource1Alpha,
};

enum class BlendOperation : u8 {
    Add,
    Subtract,        ///< source less destination
    ReverseSubtract, ///< destination less source
    Min,
    Max,
};

/// Which channels a colour attachment writes, a bit each.
namespace color_write {
inline constexpr u8 red = 1U << 0;
inline constexpr u8 green = 1U << 1;
inline constexpr u8 blue = 1U << 2;
inline constexpr u8 alpha = 1U << 3;
inline constexpr u8 all = red | green | blue | alpha;
} // namespace color_write

/// How one colour attachment is written: blended or not, and which
/// channels. Part of a pipeline, as Metal has it.
struct BlendState {
    bool enabled = false;
    BlendFactor source_rgb = BlendFactor::One;
    BlendFactor destination_rgb = BlendFactor::Zero;
    BlendOperation rgb = BlendOperation::Add;
    BlendFactor source_alpha = BlendFactor::One;
    BlendFactor destination_alpha = BlendFactor::Zero;
    BlendOperation alpha = BlendOperation::Add;
    u8 write_mask = color_write::all;

    auto operator<=>(const BlendState&) const = default;
};

/// Which faces a pass's draws skip.
enum class CullMode : u8 {
    None,
    Front,
    Back,
};

/// Which winding, in window coordinates, is a front face.
enum class Winding : u8 {
    Clockwise,
    CounterClockwise,
};

struct Viewport {
    f32 x = 0.0F;
    f32 y = 0.0F;
    f32 width = 0.0F;
    f32 height = 0.0F;

    /// Metal's depth range is always [0,1]; Latte's is configurable and often
    /// [-1,1]. The translator folds the difference into the projection rather
    /// than asking the backend for something Metal cannot do (Phase 7 table).
    f32 near_z = 0.0F;
    f32 far_z = 1.0F;

    [[nodiscard]] bool operator==(const Viewport&) const noexcept = default;
};

struct Rect {
    u32 x = 0;
    u32 y = 0;
    u32 width = 0;
    u32 height = 0;

    [[nodiscard]] bool operator==(const Rect&) const noexcept = default;
};

/// What a frame cost, and the one number **P4** asks for.
struct FrameStats {
    u64 draws = 0;
    u64 render_passes = 0;

    /// How long the CPU spent blocked on the GPU this frame, in microseconds.
    ///
    /// Should read near zero: the guest blocks where it asked to and nowhere
    /// else, and every other wait is a bug worth up to 2× on sync-heavy
    /// titles. A counter rather than a log line, because the question is what
    /// fraction of a frame it was.
    u64 cpu_stall_us = 0;

    /// How long `begin_frame` waited for a frame slot, in microseconds.
    ///
    /// Kept apart from `cpu_stall_us` on purpose. This one is pacing: with
    /// `frames_in_flight` frames queued the CPU *should* wait here, and with
    /// vsync on it is most of an idle frame. A stall is the CPU waiting on the
    /// GPU where the guest did not ask it to, which is never fine.
    u64 pacing_wait_us = 0;
};

/// Where the backend's memory is, for the debug UI (10.1) and for the tests
/// that check resources really are coming out of heaps (7.4).
struct MemoryStats {
    /// Heaps resources are suballocated from (**P22**), the bytes they
    /// reserve, and how many of those are in use.
    u64 heaps = 0;
    u64 heap_bytes = 0;
    u64 heap_used_bytes = 0;

    /// Resources that had to be allocated on their own: memoryless targets,
    /// which no heap can hold, and anything a heap refused.
    u64 dedicated = 0;

    /// Host ranges wrapped in place rather than copied (`import_host_memory`).
    u64 imported = 0;
    u64 imported_bytes = 0;
};

inline constexpr u32 max_color_attachments = 8;

/// Binding indices per stage, until ADR-003's argument buffers lift them.
///
/// These are Metal's direct-binding limits, and they are the limits *now*
/// because binding is direct now: an index reaches the shader as
/// `[[buffer(n)]]`, `[[texture(n)]]` or `[[sampler(n)]]` in the stage it was
/// bound for. ADR-003 moves the guest's resources into argument buffers with
/// the MSL emitter (8.6, 8.12); the indices stay, the ceilings go.
inline constexpr u32 max_bound_buffers = 31;

/// The most `bind_bytes` takes at once: Metal's limit for `set*Bytes`, past
/// which it wants a buffer.
inline constexpr u32 max_bound_bytes = 4096;
inline constexpr u32 max_bound_textures = 128;
inline constexpr u32 max_bound_samplers = 16;

/// How many frames may be in flight before the CPU waits for the GPU.
/// Triple buffering: one being displayed, one queued, one being built.
inline constexpr u32 frames_in_flight = 3;

} // namespace wiiu::gpu
