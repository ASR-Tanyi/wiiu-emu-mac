// Phase 8.6 — the emitter.
//
// What is guarded: the MSL decaf's example triangle becomes, word for word
// (that it compiles and draws is test_metal.cpp's, on the Mac); the vertex
// shader's outputs matching the pixel shader's inputs by semantic, those it
// does not write included; each attribute index and component select; each
// interpolation's qualifier; OpenGL's clip-space depth brought to Metal's
// when asked; a sample, through the texture's selects and then its own,
// with each texture and sampler bound once at its unit's slot; ALU groups,
// word for word -- every source read before any result is written, a
// reduction computed once and written where each unit's write mask says,
// each opcode, source modifier and output modifier, and only the helpers a
// shader uses; uniforms, the registers and each block at its slot; and the
// refusal of what is not emitted yet.

#include <wiiu/latte/msl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <variant>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

constexpr ir::Swizzle kXyzw{Sel::X, Sel::Y, Sel::Z, Sel::W};

/// Four floats, big-endian, from attribute buffer `buffer` into `gpr`.
[[nodiscard]] ir::VertexAttribute float4_attribute(u32 semantic, u32 gpr, u32 buffer) {
    ir::VertexAttribute a;
    a.semantic = semantic;
    a.gpr = gpr;
    a.buffer = buffer;
    a.bytes = 16;
    a.data_format = 0x23;
    a.num_format = NumFormat::Scaled;
    a.endian = Endian::Swap8In32;
    a.components = kXyzw;
    return a;
}

/// decaf's example vertex shader as the IR has it: colour into R1,
/// position into R2, the position and parameter 0 exported from them.
[[nodiscard]] ir::Shader example_vertex() {
    ir::Shader vs;
    vs.stage = ir::Stage::Vertex;
    vs.attributes = {float4_attribute(0, 1, 0), float4_attribute(1, 2, 1)};
    vs.body = {ir::FetchAttributes{},
               ir::ExportPosition{.gpr = 2, .swizzle = kXyzw},
               ir::ExportParameter{.index = 0, .semantic = 0, .gpr = 1, .swizzle = kXyzw}};
    return vs;
}

/// And its pixel shader: the colour in, to target 0.
[[nodiscard]] ir::Shader example_pixel() {
    ir::Shader ps;
    ps.stage = ir::Stage::Pixel;
    ps.inputs = {ir::PixelInput{.gpr = 0,
                                .source = ir::InputSource::Parameter,
                                .semantic = 0,
                                .parameter = 0,
                                .default_value = 1,
                                .interpolation = ir::Interpolation::Perspective,
                                .centroid = false,
                                .sample = false}};
    ps.body = {ir::ExportColor{.index = 0, .target = 0, .gpr = 0, .swizzle = kXyzw}};
    return ps;
}

[[nodiscard]] std::string vertex(const ir::Shader& vs, const ir::Shader* ps = nullptr) {
    std::string out;
    std::string error;
    INFO(error);
    REQUIRE(msl::emit_vertex(vs, ps, out, error));
    return out;
}

[[nodiscard]] std::string pixel(const ir::Shader& ps) {
    std::string out;
    std::string error;
    INFO(error);
    REQUIRE(msl::emit_pixel(ps, out, error));
    return out;
}

/// The source from `marker` on: the part a test cares about, without the
/// shared helpers above it.
[[nodiscard]] std::string from(const std::string& source, std::string_view marker) {
    const std::size_t at = source.find(marker);
    REQUIRE(at != std::string::npos);
    return source.substr(at);
}

} // namespace

TEST_CASE("decaf's example triangle, as MSL", "[latte][shader][msl]") {
    const ir::Shader ps = example_pixel();
    const std::string vs = vertex(example_vertex(), &ps);
    CHECK(vs.starts_with("#include <metal_stdlib>\nusing namespace metal;\n"));
    CHECK(vs.find("static uint4 fetch32(") != std::string::npos);
    CHECK(from(vs, "struct VertexOut") ==
          "struct VertexOut {\n"
          "    float4 position [[position]];\n"
          "    float4 sem0 [[user(sem0)]];\n"
          "};\n"
          "\n"
          "vertex VertexOut vertex_main(uint vid [[vertex_id]],\n"
          "                    uint iid [[instance_id]],\n"
          "                    constant FetchConstants& fetch [[buffer(0)]],\n"
          "                    device const uchar* attrib0 [[buffer(1)]],\n"
          "                    device const uchar* attrib1 [[buffer(2)]]) {\n"
          "    float4 R1 = float4(0.0);\n"
          "    float4 R2 = float4(0.0);\n"
          "    VertexOut out = {};\n"
          "    {\n"
          "        const float4 v = as_type<float4>(fetch32(attrib0, fetch.size[0], "
          "fetch.stride[0], vid, 0u, 4u, true));\n"
          "        R1.x = v.x;\n"
          "        R1.y = v.y;\n"
          "        R1.z = v.z;\n"
          "        R1.w = v.w;\n"
          "    }\n"
          "    {\n"
          "        const float4 v = as_type<float4>(fetch32(attrib1, fetch.size[1], "
          "fetch.stride[1], vid, 0u, 4u, true));\n"
          "        R2.x = v.x;\n"
          "        R2.y = v.y;\n"
          "        R2.z = v.z;\n"
          "        R2.w = v.w;\n"
          "    }\n"
          "    out.position.x = R2.x;\n"
          "    out.position.y = R2.y;\n"
          "    out.position.z = R2.z;\n"
          "    out.position.w = R2.w;\n"
          "    out.sem0.x = R1.x;\n"
          "    out.sem0.y = R1.y;\n"
          "    out.sem0.z = R1.z;\n"
          "    out.sem0.w = R1.w;\n"
          "    return out;\n"
          "}\n");

    CHECK(pixel(ps) == "#include <metal_stdlib>\n"
                       "using namespace metal;\n"
                       "\n"
                       "struct PixelIn {\n"
                       "    float4 sem0 [[user(sem0)]] [[center_perspective]];\n"
                       "};\n"
                       "\n"
                       "struct PixelOut {\n"
                       "    float4 color0 [[color(0)]];\n"
                       "};\n"
                       "\n"
                       "fragment PixelOut fragment_main(PixelIn in [[stage_in]]) {\n"
                       "    float4 R0 = float4(0.0);\n"
                       "    R0 = in.sem0;\n"
                       "    PixelOut out = {};\n"
                       "    out.color0.x = R0.x;\n"
                       "    out.color0.y = R0.y;\n"
                       "    out.color0.z = R0.z;\n"
                       "    out.color0.w = R0.w;\n"
                       "    return out;\n"
                       "}\n");
}

TEST_CASE("the MSL fetch constants are the C++ ones", "[latte][shader][msl]") {
    // Sixteen strides, sixteen sizes, two step rates and two words of
    // padding, as uints: 144 bytes either side.
    const std::string vs = vertex(example_vertex());
    CHECK(from(vs, "struct FetchConstants")
              .starts_with("struct FetchConstants {\n"
                           "    uint stride[16];\n"
                           "    uint size[16];\n"
                           "    uint step_rate[2];\n"
                           "    uint pad[2];\n"
                           "};\n"));
    CHECK(offsetof(msl::FetchConstants, size) == 64);
    CHECK(offsetof(msl::FetchConstants, step_rate) == 128);
}

TEST_CASE("a semantic the pixel shader reads is output, if only as zero", "[latte][shader][msl]") {
    ir::Shader ps = example_pixel();
    ps.inputs.push_back(ir::PixelInput{.gpr = 1,
                                       .source = ir::InputSource::Parameter,
                                       .semantic = 9,
                                       .parameter = 3,
                                       .default_value = 0,
                                       .interpolation = ir::Interpolation::Flat,
                                       .centroid = false,
                                       .sample = false});
    const std::string vs = vertex(example_vertex(), &ps);
    CHECK(vs.find("    float4 sem9 [[user(sem9)]];\n") != std::string::npos);
    CHECK(vs.find("out.sem9.") == std::string::npos); // declared, never written
}

TEST_CASE("OpenGL's clip-space depth is brought to Metal's as the position is returned",
          "[latte][shader][msl]") {
    const ir::Shader ps = example_pixel();
    CHECK(vertex(example_vertex(), &ps).find("out.position.w) * 0.5") == std::string::npos);

    std::string out;
    std::string error;
    REQUIRE(msl::emit_vertex(
        example_vertex(), &ps, out, error, msl::VertexOptions{.depth_from_minus_one = true}));
    CHECK(out.ends_with("    out.position.w = R2.w;\n"
                        "    out.sem0.x = R1.x;\n"
                        "    out.sem0.y = R1.y;\n"
                        "    out.sem0.z = R1.z;\n"
                        "    out.sem0.w = R1.w;\n"
                        "    out.position.z = (out.position.z + out.position.w) * 0.5;\n"
                        "    return out;\n"
                        "}\n"));
}

namespace {

/// The example pixel shader, sampling texture 3 with sampler 5 at its
/// input's XY into R1 -- an RGB texture, so alpha reads one -- and
/// exporting R1.
[[nodiscard]] ir::Shader sampling_pixel() {
    ir::Shader ps = example_pixel();
    ps.body = {ir::SampleTexture{.texture = 3,
                                 .sampler = 5,
                                 .src_gpr = 0,
                                 .coordinates = {Sel::X, Sel::Y},
                                 .texture_select = {Sel::X, Sel::Y, Sel::Z, Sel::One},
                                 .dst_gpr = 1,
                                 .swizzle = kXyzw},
               ir::ExportColor{.index = 0, .target = 0, .gpr = 1, .swizzle = kXyzw}};
    return ps;
}

} // namespace

TEST_CASE("a sample reads its unit's texture with its sampler", "[latte][shader][msl]") {
    CHECK(from(pixel(sampling_pixel()), "fragment") ==
          "fragment PixelOut fragment_main(PixelIn in [[stage_in]],\n"
          "                    texture2d<float> tex3 [[texture(3)]],\n"
          "                    sampler smp5 [[sampler(5)]]) {\n"
          "    float4 R0 = float4(0.0);\n"
          "    float4 R1 = float4(0.0);\n"
          "    R0 = in.sem0;\n"
          "    PixelOut out = {};\n"
          "    {\n"
          "        const float4 t = tex3.sample(smp5, float2(R0.x, R0.y));\n"
          "        const float4 v = float4(t.x, t.y, t.z, 1.0);\n"
          "        R1.x = v.x;\n"
          "        R1.y = v.y;\n"
          "        R1.z = v.z;\n"
          "        R1.w = v.w;\n"
          "    }\n"
          "    out.color0.x = R1.x;\n"
          "    out.color0.y = R1.y;\n"
          "    out.color0.z = R1.z;\n"
          "    out.color0.w = R1.w;\n"
          "    return out;\n"
          "}\n");

    // Its own selects, a coordinate from a constant, and a texture
    // channel masked off -- which reads zero.
    ir::Shader swizzled = sampling_pixel();
    auto& sample = std::get<ir::SampleTexture>(swizzled.body[0]);
    sample.coordinates = {Sel::W, Sel::Zero};
    sample.texture_select = {Sel::W, Sel::Mask, Sel::One, Sel::X};
    sample.swizzle = {Sel::Y, Sel::Mask, Sel::One, Sel::X};
    const std::string text = pixel(swizzled);
    CHECK(text.find("tex3.sample(smp5, float2(R0.w, 0.0))") != std::string::npos);
    CHECK(text.find("const float4 v = float4(t.w, 0.0, 1.0, t.x);") != std::string::npos);
    CHECK(text.find("        R1.x = v.y;\n"
                    "        R1.z = 1.0;\n"
                    "        R1.w = v.x;\n"
                    "    }\n") != std::string::npos);

    // Two samples of one texture with two samplers: the texture bound once.
    ir::Shader twice = sampling_pixel();
    ir::SampleTexture second = std::get<ir::SampleTexture>(twice.body[0]);
    second.sampler = 6;
    second.dst_gpr = 2;
    twice.body.insert(twice.body.begin() + 1, second);
    const std::string both = pixel(twice);
    CHECK(both.find("texture2d<float> tex3 [[texture(3)]],\n"
                    "                    sampler smp5 [[sampler(5)]],\n"
                    "                    sampler smp6 [[sampler(6)]]) {") != std::string::npos);
}

TEST_CASE("an attribute's index and components", "[latte][shader][msl]") {
    ir::Shader vs = example_vertex();
    vs.attributes[0].index = ir::AttributeIndex::InstanceRate1;
    vs.attributes[0].data_format = 0x1E; // 32_32 float
    vs.attributes[0].offset = 12;
    vs.attributes[0].endian = Endian::None;
    vs.attributes[0].components = {Sel::Y, Sel::X, Sel::Zero, Sel::One};
    vs.attributes[1].index = ir::AttributeIndex::Instance;
    vs.attributes[1].components = {Sel::X, Sel::Mask, Sel::Mask, Sel::W};
    const std::string text = vertex(vs);
    CHECK(
        text.find("fetch32(attrib0, fetch.size[0], fetch.stride[0], iid / max(fetch.step_rate[1], "
                  "1u), 12u, 2u, false)") != std::string::npos);
    CHECK(text.find("        R1.x = v.y;\n        R1.y = v.x;\n        R1.z = 0.0;\n        R1.w = "
                    "1.0;\n") != std::string::npos);
    CHECK(text.find(", iid, 0u, 4u, true)") != std::string::npos);
    CHECK(text.find("        R2.x = v.x;\n        R2.w = v.w;\n") != std::string::npos);
}

TEST_CASE("each interpolation, the position and a default", "[latte][shader][msl]") {
    ir::Shader ps;
    ps.stage = ir::Stage::Pixel;
    const auto input =
        [](u32 gpr, u32 semantic, ir::Interpolation interpolation, bool centroid, bool sample) {
            return ir::PixelInput{.gpr = gpr,
                                  .source = ir::InputSource::Parameter,
                                  .semantic = semantic,
                                  .parameter = gpr,
                                  .default_value = 0,
                                  .interpolation = interpolation,
                                  .centroid = centroid,
                                  .sample = sample};
        };
    ps.inputs = {input(0, 1, ir::Interpolation::Flat, false, false),
                 input(1, 2, ir::Interpolation::Linear, true, false),
                 input(2, 3, ir::Interpolation::Linear, false, true),
                 input(3, 4, ir::Interpolation::Perspective, true, false),
                 input(4, 5, ir::Interpolation::Perspective, false, true)};
    ps.inputs.push_back(ir::PixelInput{.gpr = 5, .source = ir::InputSource::Position});
    ps.inputs.push_back(
        ir::PixelInput{.gpr = 6, .source = ir::InputSource::Default, .default_value = 2});
    const std::string text = pixel(ps);
    CHECK(text.find("sem1 [[user(sem1)]] [[flat]];") != std::string::npos);
    CHECK(text.find("sem2 [[user(sem2)]] [[centroid_no_perspective]];") != std::string::npos);
    CHECK(text.find("sem3 [[user(sem3)]] [[sample_no_perspective]];") != std::string::npos);
    CHECK(text.find("sem4 [[user(sem4)]] [[centroid_perspective]];") != std::string::npos);
    CHECK(text.find("sem5 [[user(sem5)]] [[sample_perspective]];") != std::string::npos);
    CHECK(text.find("    float4 position [[position]];\n") != std::string::npos);
    CHECK(text.find("    R5 = in.position;\n") != std::string::npos);
    CHECK(text.find("    R6 = float4(1.0, 1.0, 1.0, 0.0);\n") != std::string::npos);
    // Nothing exported: a fragment function that returns nothing.
    CHECK(text.find("fragment void fragment_main(PixelIn in [[stage_in]])") != std::string::npos);
}

TEST_CASE("what is not emitted yet is refused", "[latte][shader][msl]") {
    std::string out;
    std::string error;

    ir::Shader bytes = example_vertex();
    bytes.attributes[0].data_format = 0x1A; // 8_8_8_8
    CHECK_FALSE(msl::emit_vertex(bytes, nullptr, out, error));
    CHECK(error.find("32-bit float") != std::string::npos);

    ir::Shader swapped = example_vertex();
    swapped.attributes[1].endian = Endian::Swap8In16;
    CHECK_FALSE(msl::emit_vertex(swapped, nullptr, out, error));
    CHECK(error.find("byte swap 1") != std::string::npos);

    ir::Shader twice = example_pixel();
    twice.inputs.push_back(twice.inputs[0]);
    twice.inputs[1].gpr = 1;
    twice.inputs[1].interpolation = ir::Interpolation::Flat;
    CHECK_FALSE(msl::emit_pixel(twice, out, error));
    CHECK(error.find("two interpolations") != std::string::npos);

    // The same semantic twice the same way is one input, read into both.
    twice.inputs[1].interpolation = ir::Interpolation::Perspective;
    REQUIRE(msl::emit_pixel(twice, out, error));
    CHECK(out.find("R1 = in.sem0;") != std::string::npos);

    CHECK_FALSE(msl::emit_vertex(example_pixel(), nullptr, out, error));
    CHECK_FALSE(msl::emit_pixel(example_vertex(), out, error));

    // Samplers past Metal's sixteen, and sampling in a vertex shader.
    ir::Shader seventeen = sampling_pixel();
    std::get<ir::SampleTexture>(seventeen.body[0]).sampler = 16;
    CHECK_FALSE(msl::emit_pixel(seventeen, out, error));
    CHECK(error.find("sampler 16 is past the 16") != std::string::npos);
    ir::Shader sampling_vertex = example_vertex();
    sampling_vertex.body.push_back(sampling_pixel().body[0]);
    CHECK_FALSE(msl::emit_vertex(sampling_vertex, nullptr, out, error));
    CHECK(error.find("vertex shader is not emitted yet") != std::string::npos);
}

// --------------------------------------------------------------------------
// ALU groups
// --------------------------------------------------------------------------

namespace {

[[nodiscard]] ir::AluSource gpr_source(u32 gpr, u32 chan) {
    return {.kind = ir::SourceKind::Register, .gpr = gpr, .chan = chan};
}

[[nodiscard]] ir::AluSource value(u32 bits) {
    return {.kind = ir::SourceKind::Value, .bits = bits};
}

[[nodiscard]] ir::AluSource pv(u32 chan) {
    return {.kind = ir::SourceKind::PreviousVector, .chan = chan};
}

[[nodiscard]] ir::AluSource previous_scalar() {
    return {.kind = ir::SourceKind::PreviousScalar};
}

[[nodiscard]] ir::AluOperation operation(AluUnit unit,
                                         ir::AluOpcode opcode,
                                         u32 dst_gpr,
                                         u32 dst_chan,
                                         std::array<ir::AluSource, 3> sources,
                                         bool write = true) {
    ir::AluOperation op;
    op.unit = unit;
    op.opcode = opcode;
    op.sources = sources;
    op.write = write;
    op.dst_gpr = dst_gpr;
    op.dst_chan = dst_chan;
    return op;
}

/// The three groups test_shader_ir.cpp decodes and test_metal.cpp draws:
/// R0 in, R1 out.
[[nodiscard]] ir::Shader alu_pixel() {
    using enum ir::AluOpcode;
    ir::Shader ps = example_pixel();
    ir::AluGroup first;
    first.operations = {
        operation(AluUnit::X, Mul, 2, 0, {gpr_source(0, 1), value(0x3E800000)}),
        operation(AluUnit::Y, Mov, 2, 1, {gpr_source(0, 1)}, false),
        operation(AluUnit::T, Mov, 0, 1, {value(0)}),
    };
    ir::AluGroup second;
    second.operations = {
        operation(AluUnit::X, Add, 1, 0, {pv(0), pv(0)}),
        operation(AluUnit::Y, Mov, 1, 1, {gpr_source(0, 1)}),
        operation(AluUnit::Z, Mul, 1, 2, {previous_scalar(), pv(1)}),
        operation(AluUnit::W, MulAdd, 1, 3, {pv(1), value(0x40000000), previous_scalar()}),
    };
    second.operations[0].modifier = ir::OutputModifier::Times2;
    second.operations[3].clamp = true;
    ir::AluGroup third;
    third.operations = {
        operation(AluUnit::X, Dot4, 1, 0, {gpr_source(1, 0), value(0x3F800000)}, false),
        operation(AluUnit::Y, Dot4, 1, 1, {gpr_source(1, 1), value(0x3F800000)}),
        operation(AluUnit::Z, Dot4, 1, 2, {gpr_source(1, 2), value(0x3F800000)}, false),
        operation(AluUnit::W, Dot4, 1, 3, {gpr_source(1, 3), value(0)}, false),
    };
    ps.body = {
        first, second, third, ir::ExportColor{.index = 0, .target = 0, .gpr = 1, .swizzle = kXyzw}};
    return ps;
}

} // namespace

TEST_CASE("an ALU group reads every source before it writes", "[latte][shader][msl][alu]") {
    const std::string text = pixel(alu_pixel());
    // The helper it uses, and not the ones it does not.
    CHECK(text.find("static float mul_dx9(float a, float b) {\n"
                    "    return (a == 0.0f || b == 0.0f) ? 0.0f : a * b;\n"
                    "}\n") != std::string::npos);
    CHECK(text.find("max_r700") == std::string::npos);
    // Each group: its results, then its writes, then PV and PS. So the T
    // unit's R0.y = 0 is not seen by the X and Y units that read R0.y in
    // the same group; the reduction is computed once and written only where
    // a unit's write mask says.
    CHECK(from(text, "fragment") == "fragment PixelOut fragment_main(PixelIn in [[stage_in]]) {\n"
                                    "    float4 R0 = float4(0.0);\n"
                                    "    float4 R1 = float4(0.0);\n"
                                    "    float4 R2 = float4(0.0);\n"
                                    "    float4 PV = float4(0.0);\n"
                                    "    float PS = 0.0;\n"
                                    "    R0 = in.sem0;\n"
                                    "    PixelOut out = {};\n"
                                    "    {\n"
                                    "        const float alu_x = mul_dx9(R0.y, "
                                    "as_type<float>(0x3E800000u));\n"
                                    "        const float alu_y = R0.y;\n"
                                    "        const float alu_t = 0.0f;\n"
                                    "        R2.x = alu_x;\n"
                                    "        R0.y = alu_t;\n"
                                    "        PV.x = alu_x;\n"
                                    "        PV.y = alu_y;\n"
                                    "        PS = alu_t;\n"
                                    "    }\n"
                                    "    {\n"
                                    "        const float alu_x = (PV.x + PV.x) * 2.0f;\n"
                                    "        const float alu_y = R0.y;\n"
                                    "        const float alu_z = mul_dx9(PS, PV.y);\n"
                                    "        const float alu_w = clamp(mul_dx9(PV.y, "
                                    "as_type<float>(0x40000000u)) + PS, 0.0f, 1.0f);\n"
                                    "        R1.x = alu_x;\n"
                                    "        R1.y = alu_y;\n"
                                    "        R1.z = alu_z;\n"
                                    "        R1.w = alu_w;\n"
                                    "        PV.x = alu_x;\n"
                                    "        PV.y = alu_y;\n"
                                    "        PV.z = alu_z;\n"
                                    "        PV.w = alu_w;\n"
                                    "    }\n"
                                    "    {\n"
                                    "        const float reduction = mul_dx9(R1.x, 1.0f) + "
                                    "mul_dx9(R1.y, 1.0f) + mul_dx9(R1.z, 1.0f) + mul_dx9(R1.w, "
                                    "0.0f);\n"
                                    "        const float alu_x = reduction;\n"
                                    "        const float alu_y = reduction;\n"
                                    "        const float alu_z = reduction;\n"
                                    "        const float alu_w = reduction;\n"
                                    "        R1.y = alu_y;\n"
                                    "        PV.x = alu_x;\n"
                                    "        PV.y = alu_y;\n"
                                    "        PV.z = alu_z;\n"
                                    "        PV.w = alu_w;\n"
                                    "    }\n"
                                    "    out.color0.x = R1.x;\n"
                                    "    out.color0.y = R1.y;\n"
                                    "    out.color0.z = R1.z;\n"
                                    "    out.color0.w = R1.w;\n"
                                    "    return out;\n"
                                    "}\n");

    // A shader with no ALU has neither the helpers nor PV and PS.
    const std::string plain = pixel(example_pixel());
    CHECK(plain.find("mul_dx9") == std::string::npos);
    CHECK(plain.find("PV") == std::string::npos);
}

TEST_CASE("each ALU opcode, source modifier and output modifier, as MSL",
          "[latte][shader][msl][alu]") {
    using enum ir::AluOpcode;
    ir::AluSource negated = gpr_source(0, 2);
    negated.negate = true;
    ir::AluSource both = gpr_source(0, 3);
    both.negate = true;
    both.absolute = true;
    ir::AluGroup group;
    group.operations = {
        operation(AluUnit::X, MulIeee, 1, 0, {negated, value(0x3F000000)}),
        operation(AluUnit::Y, Max, 1, 1, {both, value(0xC0400000)}),
        operation(AluUnit::Z, Min, 1, 2, {gpr_source(0, 0), gpr_source(0, 1)}),
        operation(AluUnit::W, MulAddIeee, 1, 3, {gpr_source(0, 0), gpr_source(0, 1), negated}),
        operation(AluUnit::T, MulAdd, 2, 0, {gpr_source(0, 0), gpr_source(0, 1), gpr_source(0, 2)}),
    };
    group.operations[1].modifier = ir::OutputModifier::Times4;
    group.operations[2].modifier = ir::OutputModifier::Half;
    group.operations[4].clamp = true;
    group.operations[4].modifier = ir::OutputModifier::Times2;
    ir::AluGroup dot;
    dot.operations = {
        operation(AluUnit::X, Dot4Ieee, 3, 0, {gpr_source(1, 0), gpr_source(2, 0)}),
        operation(AluUnit::Y, Dot4Ieee, 3, 1, {gpr_source(1, 1), gpr_source(2, 0)}),
        operation(AluUnit::Z, Dot4Ieee, 3, 2, {gpr_source(1, 2), gpr_source(2, 0)}, false),
        operation(AluUnit::W, Dot4Ieee, 3, 3, {gpr_source(1, 3), gpr_source(2, 0)}, false),
    };
    ir::Shader ps = example_pixel();
    ps.body = {group, dot, ir::ExportColor{.index = 0, .target = 0, .gpr = 3, .swizzle = kXyzw}};
    const std::string text = pixel(ps);
    CHECK(text.find("        const float alu_x = -R0.z * 0.5f;\n"
                    "        const float alu_y = (max_r700(-abs(R0.w), "
                    "as_type<float>(0xC0400000u))) * 4.0f;\n"
                    "        const float alu_z = (min_r700(R0.x, R0.y)) * 0.5f;\n"
                    "        const float alu_w = R0.x * R0.y + -R0.z;\n"
                    "        const float alu_t = clamp((mul_dx9(R0.x, R0.y) + R0.z) * 2.0f, 0.0f, "
                    "1.0f);\n"
                    "        R1.x = alu_x;\n"
                    "        R1.y = alu_y;\n"
                    "        R1.z = alu_z;\n"
                    "        R1.w = alu_w;\n"
                    "        R2.x = alu_t;\n") != std::string::npos);
    // DOT4_IEEE's multiplies are IEEE's; the result goes to every unit that
    // writes.
    CHECK(text.find("        const float reduction = R1.x * R2.x + R1.y * R2.x + R1.z * R2.x + "
                    "R1.w * R2.x;\n") != std::string::npos);
    CHECK(text.find("        R3.x = alu_x;\n"
                    "        R3.y = alu_y;\n"
                    "        PV.x = alu_x;\n") != std::string::npos);

    // In a vertex shader too, with the helpers and PV beside the fetch.
    ir::Shader vs = example_vertex();
    ir::AluGroup scale;
    scale.operations = {operation(AluUnit::X, Mul, 2, 0, {gpr_source(2, 0), value(0x3F000000)})};
    vs.body.insert(vs.body.begin() + 1, scale);
    const std::string vertex_text = vertex(vs);
    // Only the helpers it uses.
    CHECK(vertex_text.find("static float mul_dx9(float a, float b)") != std::string::npos);
    CHECK(vertex_text.find("max_r700") == std::string::npos);
    CHECK(text.find("static float max_r700(float a, float b) {\n"
                    "    return a >= b ? a : b;\n"
                    "}\n") != std::string::npos);
    CHECK(text.find("static float min_r700(float a, float b) {\n"
                    "    return a < b ? a : b;\n"
                    "}\n") != std::string::npos);
    CHECK(vertex_text.find("    float4 PV = float4(0.0);\n"
                           "    float PS = 0.0;\n"
                           "    VertexOut out = {};\n") != std::string::npos);
    CHECK(vertex_text.find("        const float alu_x = mul_dx9(R2.x, 0.5f);\n"
                           "        R2.x = alu_x;\n") != std::string::npos);
}

TEST_CASE("uniforms are read where they are bound", "[latte][shader][msl][alu][uniforms]") {
    using enum ir::AluOpcode;
    const auto block = [](u32 number, u32 index, u32 chan) {
        return ir::AluSource{
            .kind = ir::SourceKind::UniformBlock, .chan = chan, .block = number, .index = index};
    };
    const auto file = [](u32 index, u32 chan) {
        return ir::AluSource{.kind = ir::SourceKind::UniformRegister, .chan = chan, .index = index};
    };
    ir::AluGroup group;
    group.operations = {
        operation(AluUnit::X, Mul, 1, 0, {block(3, 52, 1), file(7, 2)}),
        operation(AluUnit::Y, Add, 1, 1, {block(0, 0, 3), block(3, 1, 0)}),
    };
    group.operations[1].sources[1].negate = true;

    // The pixel shader: the uniform registers at 17, block n at 18 + n,
    // after its inputs and before its textures.
    ir::Shader ps = example_pixel();
    ps.body = {group, ir::ExportColor{.index = 0, .target = 0, .gpr = 1, .swizzle = kXyzw}};
    const std::string text = pixel(ps);
    CHECK(text.find("fragment PixelOut fragment_main(PixelIn in [[stage_in]],\n"
                    "                    constant float4* uniforms [[buffer(17)]],\n"
                    "                    constant float4* ub0 [[buffer(18)]],\n"
                    "                    constant float4* ub3 [[buffer(21)]]) {\n") !=
          std::string::npos);
    CHECK(text.find("        const float alu_x = mul_dx9(ub3[52].y, uniforms[7].z);\n"
                    "        const float alu_y = ub0[0].w + -ub3[1].x;\n") != std::string::npos);

    // The vertex shader: the same slots, after its attribute buffers.
    ir::Shader vs = example_vertex();
    vs.body.insert(vs.body.begin() + 1, group);
    const std::string vertex_text = vertex(vs);
    CHECK(vertex_text.find(
              "                    device const uchar* attrib1 [[buffer(2)]],\n"
              "                    constant float4* uniforms [[buffer(17)]],\n"
              "                    constant float4* ub0 [[buffer(18)]],\n"
              "                    constant float4* ub3 [[buffer(21)]]) {\n") != std::string::npos);

    // A shader with no uniforms has no such parameters.
    CHECK(pixel(alu_pixel()).find("uniforms") == std::string::npos);

    // Block 13 on is past Metal's thirty-one buffer slots.
    group.operations[0].sources[0].block = 13;
    ps.body[0] = group;
    std::string out;
    std::string error;
    CHECK_FALSE(msl::emit_pixel(ps, out, error));
    CHECK(error.find("uniform block 13 is past the 13 Metal has buffer slots for") !=
          std::string::npos);
}
