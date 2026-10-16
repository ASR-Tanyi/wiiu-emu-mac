// Phase 8.1 — the fetch shader GX2 writes.
//
// What is guarded: the program's size, against the formula decaf and Cemu
// share (and decaf's for the tessellated sizes); its words, worked out by
// hand from the field positions decaf and Cemu agree on; sixteen fetches to
// a clause; the instance divisors and their step rates; the attribute
// format tables; and the calls themselves -- the program written
// little-endian into the title's buffer, the structure filled in, and
// binding it reaching the registers.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/state.hpp>
#include <wiiu/hle/coreinit.hpp>
#include <wiiu/hle/gx2.hpp>
#include <wiiu/hle/gx2_fetch_shader.hpp>
#include <wiiu/hle/gx2_shader.hpp>
#include <wiiu/hle/registry.hpp>
#include <wiiu/latte/control_flow.hpp>
#include <wiiu/latte/microcode.hpp>
#include <wiiu/latte/shader_program.hpp>
#include <wiiu/latte/shader_state.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace wiiu;
using namespace wiiu::hle;
using gx2::FetchShaderType;
using gx2::TessellationMode;
namespace reg = latte::reg;

namespace {

// GX2AttribFormat values (decaf's names).
constexpr u32 kUnorm8888 = 0x00A;
constexpr u32 kUint88 = 0x104;
constexpr u32 kSnorm1010102 = 0x20B;
constexpr u32 kFloat32 = 0x806;
constexpr u32 kFloat323232 = 0x811;

// GX2's component selects: XYZW, and XYZ with W one.
constexpr u32 kSelXyzw = 0x00010203;
constexpr u32 kSelXyz1 = 0x00010205;

[[nodiscard]] gx2::AttribStream
attrib(u32 location, u32 buffer, u32 offset, u32 format, u32 mask = kSelXyzw, u32 swap = 3) {
    gx2::AttribStream out{};
    out.location = location;
    out.buffer = buffer;
    out.offset = offset;
    out.format = format;
    out.type = static_cast<u32>(gx2::AttribIndexType::PerVertex);
    out.mask = mask;
    out.endian_swap = swap;
    return out;
}

[[nodiscard]] gx2::AttribStream per_instance(u32 location, u32 divisor) {
    gx2::AttribStream out = attrib(location, 2, 0, kFloat32);
    out.type = static_cast<u32>(gx2::AttribIndexType::PerInstance);
    out.alu_divisor = divisor;
    return out;
}

[[nodiscard]] latte::VtxWord0
fetch_word0(const gx2::FetchShaderProgram& program, u32 head, u32 fetch) {
    return latte::VtxWord0{.raw = program.words[(head / 4) + (4 * fetch)]};
}

} // namespace

TEST_CASE("the fetch shader's size", "[hle][gx2][fetch]") {
    constexpr auto none = FetchShaderType::NoTessellation;
    constexpr auto discrete = TessellationMode::Discrete;
    // Untessellated: a VTX_TC per sixteen fetches and the RETURN, eight
    // bytes each, padded to sixteen; then sixteen bytes a fetch.
    CHECK(gx2::fetch_shader_size(0, none, discrete) == 16);
    CHECK(gx2::fetch_shader_size(1, none, discrete) == 16 + 16);
    CHECK(gx2::fetch_shader_size(2, none, discrete) == 16 + 32);
    CHECK(gx2::fetch_shader_size(16, none, discrete) == 16 + 256);
    CHECK(gx2::fetch_shader_size(17, none, discrete) == 32 + 272); // 3 CF: 24, padded
    CHECK(gx2::fetch_shader_size(33, none, discrete) == 32 + 528); // 4 CF: 32
    // The mode means nothing without tessellation.
    CHECK(gx2::fetch_shader_size(5, none, TessellationMode::Adaptive) == 16 + 80);

    // decaf's tessellated sizes. Lines, four attributes, discrete: two
    // fetches for each but two attributes, 4 + 4 ALU instructions, and a
    // VTX_TC, two more and the RETURN: 32 + 64 bytes, then 64.
    CHECK(gx2::fetch_shader_size(4, FetchShaderType::LineTessellation, discrete) == 96 + 64);
    // Triangles, five, adaptive: nine fetches, 57 + 4 ALU, four CF: 32 + 488
    // padded to 528, then 144.
    CHECK(gx2::fetch_shader_size(
              5, FetchShaderType::TriangleTessellation, TessellationMode::Adaptive) == 528 + 144);
    // Fewer than the two attributes tessellation takes: no fetches.
    CHECK(gx2::fetch_shader_size(1, FetchShaderType::QuadTessellation, discrete) == 96);
    // Garbage counts get a size no buffer has.
    CHECK(gx2::fetch_shader_size(0x10000, none, discrete) == 0xFFFFFFFF);
}

TEST_CASE("the fetch shader's words", "[hle][gx2][fetch]") {
    const std::array attribs{attrib(0, 0, 0, kFloat323232, kSelXyz1),
                             attrib(1, 1, 12, kUnorm8888, kSelXyzw, 0)};
    const gx2::FetchShaderProgram program = gx2::generate_fetch_shader(attribs);
    REQUIRE(program.words.size() == 48 / 4);

    // One VTX_TC of two fetches starting at byte 16, then RETURN with a barrier.
    CHECK(program.words[0] == 16 / 8);
    CHECK(program.words[1] == ((2U - 1) << 10 | 0x03U << 23));
    CHECK(program.words[2] == 0);
    CHECK(program.words[3] == 0x8A000000); // Cemu's note of the word
    // Position: a semantic fetch from attribute buffer 0 (resource 0xA0), the
    // vertex index in R0.x, twelve bytes; semantic 0, XYZ1, FMT_32_32_32_FLOAT
    // (0x30) read through the "scaled" flag the float formats carry; 8-in-32.
    CHECK(program.words[4] == (0x1U | 0xA0U << 8 | 11U << 26));
    CHECK(program.words[5] == (1U << 12 | 2U << 15 | 5U << 18 | 0x30U << 22 | 2U << 28));
    CHECK(program.words[6] == (0U | 2U << 16 | 1U << 19));
    CHECK(program.words[7] == 0);
    // Colour: buffer 1, offset 12, FMT_8_8_8_8 (0x1A) normalised, no swap
    // asked for.
    CHECK(program.words[8] == (0x1U | 0xA1U << 8 | 3U << 26));
    CHECK(program.words[9] == (1U | 1U << 12 | 2U << 15 | 3U << 18 | 0x1AU << 22));
    CHECK(program.words[10] == (12U | 1U << 19));
    CHECK(program.divisor_count == 0);
}

TEST_CASE("the fetch shader's formats and swaps", "[hle][gx2][fetch]") {
    wiiu::log::set_all_levels(wiiu::log::Level::Off); // the unknown format is logged
    // The data formats: Cemu's table, decaf's switch but for 10_10_10_2.
    CHECK(gx2::attrib_data_format(0x00) == 0x01U);          // 8
    CHECK(gx2::attrib_data_format(kUint88) == 0x07U);       // 8_8
    CHECK(gx2::attrib_data_format(kFloat32) == 0x0EU);      // 32_FLOAT
    CHECK(gx2::attrib_data_format(0x009) == 0x16U);         // 10_11_11_FLOAT
    CHECK(gx2::attrib_data_format(kSnorm1010102) == 0x19U); // FMT_2_10_10_10
    CHECK(gx2::attrib_data_format(0x813) == 0x23U);         // 32_32_32_32_FLOAT
    CHECK_FALSE(gx2::attrib_data_format(0x014).has_value());

    CHECK(gx2::attrib_format_bits(0x001) == 8);
    CHECK(gx2::attrib_format_bits(kUint88) == 16);
    CHECK(gx2::attrib_format_bits(kSnorm1010102) == 32);
    CHECK(gx2::attrib_format_bits(0x80F) == 64); // 16_16_16_16_FLOAT
    CHECK(gx2::attrib_format_bits(kFloat323232) == 96);
    CHECK(gx2::attrib_format_bits(0x014) == 0);

    CHECK(gx2::attrib_default_endian(kUnorm8888) == latte::Endian::None);
    CHECK(gx2::attrib_default_endian(0x003) == latte::Endian::Swap8In16); // 16_FLOAT
    CHECK(gx2::attrib_default_endian(kSnorm1010102) == latte::Endian::Swap8In32);

    // Integer, signed and scaled reach the fetch; a swap the stream asks
    // for overrides the format's own.
    const std::array attribs{attrib(0, 0, 0, kUint88, kSelXyzw, 1),
                             attrib(1, 0, 4, kSnorm1010102),
                             attrib(2, 0, 8, 0x014)};
    const gx2::FetchShaderProgram program = gx2::generate_fetch_shader(attribs);
    const auto word1 = [&](u32 fetch) {
        return latte::VtxWord1{.raw = program.words[4 + (4 * fetch) + 1]};
    };
    const auto word2 = [&](u32 fetch) {
        return latte::VtxWord2{.raw = program.words[4 + (4 * fetch) + 2]};
    };
    CHECK(word1(0).num_format_all == latte::NumFormat::Int);
    CHECK_FALSE(word1(0).format_comp_signed);
    CHECK(word2(0).endian_swap == latte::Endian::Swap8In16);
    CHECK(word1(1).num_format_all == latte::NumFormat::Norm);
    CHECK(word1(1).format_comp_signed);
    CHECK(word1(1).data_format == 0x19U);
    CHECK(word2(1).endian_swap == latte::Endian::Swap8In32);
    // A type GX2 does not have: FMT_INVALID, and a mega-fetch of nothing.
    CHECK(word1(2).data_format == 0U);
    CHECK(latte::VtxWord0{.raw = program.words[4 + 8]}.mega_fetch_count == 0U);
}

TEST_CASE("sixteen fetches to a clause", "[hle][gx2][fetch]") {
    std::vector<gx2::AttribStream> attribs;
    for (u32 i = 0; i < 17; ++i) {
        attribs.push_back(attrib(i, 0, 4 * i, kFloat32));
    }
    const gx2::FetchShaderProgram program = gx2::generate_fetch_shader(attribs);
    REQUIRE(program.words.size() == (32 + (17 * 16)) / 4);

    const latte::CfWord1 first{.raw = program.words[1]};
    CHECK(program.words[0] == 32 / 8);
    CHECK(first.cf_inst == latte::CfInst::VtxTc);
    CHECK(first.count == 7U);   // fifteen, low three bits
    CHECK(first.count_3 == 1U); // and the fourth
    CHECK_FALSE(first.barrier);

    const latte::CfWord1 second{.raw = program.words[3]};
    CHECK(program.words[2] == (32 + (16 * 16)) / 8);
    CHECK(second.count == 0U);
    CHECK(second.count_3 == 0U);

    const latte::CfWord1 end{.raw = program.words[5]};
    CHECK(end.cf_inst == latte::CfInst::Return);
    CHECK(end.barrier);
    CHECK(program.words[6] == 0); // padding
    CHECK(program.words[7] == 0);

    // The seventeenth fetch is the last.
    const latte::VtxWord1 last{.raw = program.words[(32 / 4) + (16 * 4) + 1]};
    CHECK(last.semantic_id == 16U);

    // And the decompiler's first stage reads it back: two clauses, inside
    // the program, then the RETURN.
    latte::ControlFlow flow;
    std::string error;
    REQUIRE(
        latte::decode_control_flow(program.words, latte::ProgramShape::FetchShader, flow, error));
    REQUIRE(flow.clauses.size() == 2);
    CHECK(flow.clauses[0].count == 16);
    CHECK(flow.clauses[1].offset_bytes() == 32 + (16 * 16));
    CHECK(flow.clauses[1].offset_bytes() + flow.clauses[1].size_bytes() ==
          program.words.size() * 4);
    CHECK(latte::disassemble(flow) == "00 VTX_TC: ADDR(4) CNT(16) NO_BARRIER\n"
                                      "01 VTX_TC: ADDR(36) CNT(1) NO_BARRIER\n"
                                      "02 RETURN\n");

    // The next stage reads the fetches: seventeen, each into its semantic.
    latte::ShaderProgram decoded;
    REQUIRE(latte::decode_program(program.words, latte::ProgramShape::FetchShader, decoded, error));
    REQUIRE(decoded.clauses.size() == 2);
    std::vector<latte::VertexFetch> fetches = decoded.clauses[0].vertex_fetches;
    fetches.insert(fetches.end(),
                   decoded.clauses[1].vertex_fetches.begin(),
                   decoded.clauses[1].vertex_fetches.end());
    REQUIRE(fetches.size() == 17);
    for (u32 i = 0; i < 17; ++i) {
        CHECK(fetches[i].is_semantic());
        CHECK(fetches[i].fetch1().semantic_id == i);
        CHECK(fetches[i].fetch2().offset == 4 * i);
    }
}

TEST_CASE("instance divisors become step rates", "[hle][gx2][fetch]") {
    const std::array attribs{per_instance(0, 1),
                             per_instance(1, 4),
                             per_instance(2, 4),
                             per_instance(3, 7),
                             per_instance(4, 9),
                             attrib(5, 0, 0, kFloat32)};
    const gx2::FetchShaderProgram program = gx2::generate_fetch_shader(attribs);
    constexpr u32 head = 16;
    CHECK(fetch_word0(program, head, 0).src_sel_x == latte::Sel::W); // the instance index
    CHECK(fetch_word0(program, head, 1).src_sel_x == latte::Sel::Y); // over step rate 0
    CHECK(fetch_word0(program, head, 2).src_sel_x == latte::Sel::Y); // shared
    CHECK(fetch_word0(program, head, 3).src_sel_x == latte::Sel::Z); // over step rate 1
    CHECK(fetch_word0(program, head, 4).src_sel_x == latte::Sel::X); // no third: Cemu's
    CHECK(fetch_word0(program, head, 0).fetch_type == latte::VtxFetchType::InstanceData);
    CHECK(fetch_word0(program, head, 5).fetch_type == latte::VtxFetchType::VertexData);
    CHECK(fetch_word0(program, head, 5).src_sel_x == latte::Sel::X);
    CHECK(program.divisor_count == 2);
    CHECK(program.divisors[0] == 4);
    CHECK(program.divisors[1] == 7);
}

// --------------------------------------------------------------------------
// Through GX2
// --------------------------------------------------------------------------

namespace {

constexpr u32 kMain = 0x10001000;
constexpr ppcaddr_t kShader = 0x10004000;
constexpr ppcaddr_t kAttribs = 0x10005000;
constexpr ppcaddr_t kProgram = 0x10010000;

struct FetchFixture {
    memory::Memory memory;
    cpu::CoreState state;
    Registry registry;
    coreinit::Context os;

    FetchFixture() {
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

    u32 call(std::string_view name, std::vector<u32> arguments) {
        Function* const function = registry.find("gx2", name);
        REQUIRE(function != nullptr);
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            state.gpr[3 + i] = arguments[i];
        }
        Call context{.state = state, .memory = memory};
        function->handler(context);
        return state.gpr[3];
    }

    void run_gpu() {
        call("GX2Flush", {});
        state.timebase += os.gpu.timeline().latency();
        static_cast<void>(os.scheduler.reschedule(state));
    }

    [[nodiscard]] u32 reg(u32 address) const { return os.gpu.registers().read(address); }

    [[nodiscard]] u32 physical(ppcaddr_t address) const {
        const memory::Region* const region = memory.region_of(address);
        REQUIRE(region != nullptr);
        physaddr_t out = 0;
        REQUIRE(memory::to_physical(*region, address, out));
        return out;
    }

    void write_attribs(std::span<const gx2::AttribStream> attribs) {
        for (std::size_t i = 0; i < attribs.size(); ++i) {
            REQUIRE(gx2::write_guest(memory, kAttribs + static_cast<u32>(0x20 * i), attribs[i]));
        }
    }
};

} // namespace

TEST_CASE_METHOD(FetchFixture,
                 "GX2InitFetchShaderEx writes the program and the structure",
                 "[hle][gx2][fetch]") {
    const std::array attribs{attrib(0, 0, 0, kFloat323232, kSelXyz1), per_instance(1, 3)};
    write_attribs(attribs);
    // Whatever was in the structure before, it is cleared (Cemu).
    gx2::FetchShader stale{};
    stale.divisor_count = 0xFFFFFFFF;
    stale.divisors[1] = 0x1234;
    stale.sq_pgm_resources_fs = 0xFF;
    REQUIRE(gx2::write_guest(memory, kShader, stale));

    CHECK(call("GX2CalcFetchShaderSizeEx", {2, 0, 0}) == 48);
    call("GX2InitFetchShaderEx", {kShader, kProgram, 2, kAttribs, 0, 0});

    gx2::FetchShader shader{};
    REQUIRE(gx2::read_guest(memory, kShader, shader));
    CHECK(shader.type == 0U);
    CHECK(shader.sq_pgm_resources_fs == 0U);
    CHECK(shader.size == 48U);
    CHECK(shader.data == kProgram);
    CHECK(shader.attrib_count == 2U);
    CHECK(shader.divisor_count == 1U);
    CHECK(shader.divisors[0] == 3U);
    CHECK(shader.divisors[1] == 0U);

    // Little-endian, as the GPU reads it: the generated words, byte for byte.
    const gx2::FetchShaderProgram expected = gx2::generate_fetch_shader(attribs);
    std::array<u8, 48> bytes{};
    REQUIRE(memory.read_bytes(kProgram, bytes));
    CHECK(bytes[0] == 2);
    CHECK(bytes[1] == 0);
    CHECK(bytes[4] == 0x00);
    CHECK(bytes[5] == 0x04);
    CHECK(bytes[6] == 0x80);
    CHECK(bytes[7] == 0x01);
    CHECK(bytes[15] == 0x8A);
    for (std::size_t i = 0; i < expected.words.size(); ++i) {
        const u32 word = static_cast<u32>(bytes[4 * i]) |
                         (static_cast<u32>(bytes[(4 * i) + 1]) << 8) |
                         (static_cast<u32>(bytes[(4 * i) + 2]) << 16) |
                         (static_cast<u32>(bytes[(4 * i) + 3]) << 24);
        CHECK(word == expected.words[i]);
    }

    // Binding it: the program, and the divisor as step rate 0.
    call("GX2SetFetchShader", {kShader});
    run_gpu();
    CHECK(reg(reg::sq_pgm_start_fs) == physical(kProgram) >> 8);
    CHECK(reg(reg::sq_pgm_start_fs + 4) == 48 >> 3);
    CHECK(reg(reg::sq_pgm_resources_fs) == 0);
    CHECK(reg(reg::vgt_instance_step_rate_0) == 3);
    CHECK(os.gpu.processor().stats().errors == 0);

    CHECK(call("GX2GetAttribFormatBits", {kFloat323232}) == 96);
    CHECK(call("GX2GetAttribFormatBits", {0x1F}) == 0);
}

TEST_CASE_METHOD(FetchFixture,
                 "a tessellated fetch shader is written untessellated",
                 "[hle][gx2][fetch]") {
    const std::array attribs{
        attrib(0, 0, 0, kFloat32), attrib(1, 0, 4, kFloat32), attrib(2, 0, 8, kFloat32)};
    write_attribs(attribs);
    const u32 size = call("GX2CalcFetchShaderSizeEx", {3, 2, 0});
    CHECK(size == gx2::fetch_shader_size(
                      3, FetchShaderType::TriangleTessellation, TessellationMode::Discrete));
    call("GX2InitFetchShaderEx", {kShader, kProgram, 3, kAttribs, 2, 0});
    gx2::FetchShader shader{};
    REQUIRE(gx2::read_guest(memory, kShader, shader));
    CHECK(shader.type == 2U);
    CHECK(shader.size == 16U + 48U); // the untessellated program, within the size given
    CHECK(shader.size <= size);
    CHECK(shader.attrib_count == 3U);

    // A type GX2 does not have is sized untessellated.
    CHECK(call("GX2CalcFetchShaderSizeEx", {3, 9, 0}) == 16 + 48);
}

TEST_CASE_METHOD(FetchFixture,
                 "GX2InitFetchShaderEx refuses what it cannot read or write",
                 "[hle][gx2][fetch]") {
    REQUIRE(gx2::write_guest(memory, kShader, gx2::FetchShader{}));
    // A count no title has.
    call("GX2InitFetchShaderEx", {kShader, kProgram, 0x10000, kAttribs, 0, 0});
    // Attributes where there is no memory.
    call("GX2InitFetchShaderEx", {kShader, kProgram, 1, 0x00000100, 0, 0});
    gx2::FetchShader shader{};
    REQUIRE(gx2::read_guest(memory, kShader, shader));
    CHECK(shader.size == 0U);
    CHECK(shader.data == 0U);
}

TEST_CASE("the fetch shader matches ones that ship with games", "[hle][gx2][fetch]") {
    // Cemu's notes on fetch shaders shipped with games (Breath of the Wild),
    // in FetchShader.cpp: the start of three programs. The second is a whole
    // one-attribute shader -- the control flow and its one fetch -- and the
    // first and third begin with the first of two and of four. Each is
    // matched by GX2InitFetchShaderEx's program for one attribute of the
    // format and components the fetch reads: 16_16_16_16 float into XYZW,
    // 32_32 float into XY01, 32_32_32 float into XYZ1.
    struct Shipped {
        u32 format;
        u32 mask;
        std::array<u32, 3> fetch;
    };

    const std::array shipped{
        Shipped{0x80F, kSelXyzw, {0x1C00A001, 0x280D1000, 0x00090000}},
        Shipped{0x80D, 0x00010405, {0x1C00A001, 0x27961000, 0x000A0000}},
        Shipped{0x811, kSelXyz1, {0x2C00A001, 0x2C151000, 0x000A0000}},
    };
    for (const Shipped& game : shipped) {
        const std::array attribs{attrib(0, 0, 0, game.format, game.mask)};
        const gx2::FetchShaderProgram program = gx2::generate_fetch_shader(attribs);
        REQUIRE(program.words.size() == 8);
        CHECK(program.words[4] == game.fetch[0]);
        CHECK(program.words[5] == game.fetch[1]);
        CHECK(program.words[6] == game.fetch[2]);
    }

    // And the whole of the one-attribute program, control flow included.
    const std::array attribs{attrib(0, 0, 0, 0x80D, 0x00010405)};
    const gx2::FetchShaderProgram program = gx2::generate_fetch_shader(attribs);
    const std::array<u32, 7> botw{
        0x00000002, 0x01800000, 0x00000000, 0x8A000000, 0x1C00A001, 0x27961000, 0x000A0000};
    for (std::size_t i = 0; i < botw.size(); ++i) {
        CHECK(program.words[i] == botw[i]);
    }
}
