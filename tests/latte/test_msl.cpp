// Phase 8.6 — the emitter.
//
// What is guarded: the MSL decaf's example triangle becomes, word for word
// (that it compiles and draws is test_metal.cpp's, on the Mac); the vertex
// shader's outputs matching the pixel shader's inputs by semantic, those it
// does not write included; each attribute index and component select; each
// interpolation's qualifier; and the refusal of what is not emitted yet.

#include <wiiu/latte/msl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

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
}
