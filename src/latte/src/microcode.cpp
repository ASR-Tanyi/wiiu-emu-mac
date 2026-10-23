// Phase 8.6 — see microcode.hpp: the ALU instruction table.
//
// decaf's (latte_instructions_def.inl), entry for entry: the numbers, names,
// source counts and flags. A scratch harness held every number of both
// forms to decaf's own lookups.

#include <wiiu/latte/microcode.hpp>

#include <array>

namespace wiiu::latte {

namespace {

namespace f = alu_flag;

struct Entry {
    u32 inst = 0;
    AluOp op;
};

constexpr std::array op2_table{
    Entry{0x00, AluOp{"ADD", 2, f::vector | f::transcendental}},
    Entry{0x01, AluOp{"MUL", 2, f::vector | f::transcendental}},
    Entry{0x02, AluOp{"MUL_IEEE", 2, f::vector | f::transcendental}},
    Entry{0x03, AluOp{"MAX", 2, f::vector | f::transcendental}},
    Entry{0x04, AluOp{"MIN", 2, f::vector | f::transcendental}},
    Entry{0x05, AluOp{"MAX_DX10", 2, f::vector | f::transcendental}},
    Entry{0x06, AluOp{"MIN_DX10", 2, f::vector | f::transcendental}},
    Entry{0x07, AluOp{"FREXP_64", 1, f::vector | f::reduction}},
    Entry{0x08, AluOp{"SETE", 2, f::vector | f::transcendental}},
    Entry{0x09, AluOp{"SETGT", 2, f::vector | f::transcendental}},
    Entry{0x0a, AluOp{"SETGE", 2, f::vector | f::transcendental}},
    Entry{0x0b, AluOp{"SETNE", 2, f::vector | f::transcendental}},
    Entry{0x0c, AluOp{"SETE_DX10", 2, f::vector | f::transcendental | f::int_out}},
    Entry{0x0d, AluOp{"SETGT_DX10", 2, f::vector | f::transcendental | f::int_out}},
    Entry{0x0e, AluOp{"SETGE_DX10", 2, f::vector | f::transcendental | f::int_out}},
    Entry{0x0f, AluOp{"SETNE_DX10", 2, f::vector | f::transcendental | f::int_out}},
    Entry{0x10, AluOp{"FRACT", 1, f::vector | f::transcendental}},
    Entry{0x11, AluOp{"TRUNC", 1, f::vector | f::transcendental}},
    Entry{0x12, AluOp{"CEIL", 1, f::vector | f::transcendental}},
    Entry{0x13, AluOp{"RNDNE", 1, f::vector | f::transcendental}},
    Entry{0x14, AluOp{"FLOOR", 1, f::vector | f::transcendental}},
    Entry{0x15, AluOp{"MOVA", 1, f::vector | f::int_out}},
    Entry{0x16, AluOp{"MOVA_FLOOR", 1, f::vector | f::int_out}},
    Entry{0x17, AluOp{"ADD_64", 2, f::vector | f::transcendental}},
    Entry{0x18, AluOp{"MOVA_INT", 1, f::vector | f::int_in | f::int_out}},
    Entry{0x19, AluOp{"MOV", 1, f::vector | f::transcendental}},
    Entry{0x1a, AluOp{"NOP", 0, f::vector | f::transcendental}},
    Entry{0x1b, AluOp{"MUL_64", 2, f::vector | f::reduction}},
    Entry{0x1c, AluOp{"FLT64_TO_FLT32", 1, f::vector | f::reduction}},
    Entry{0x1d, AluOp{"FLT32_TO_FLT64", 1, f::vector | f::reduction}},
    Entry{0x1e,
          AluOp{"PRED_SETGT_UINT",
                2,
                f::vector | f::transcendental | f::pred_set | f::uint_in | f::uint_out}},
    Entry{0x1f,
          AluOp{"PRED_SETGE_UINT",
                2,
                f::vector | f::transcendental | f::pred_set | f::uint_in | f::uint_out}},
    Entry{0x20, AluOp{"PRED_SETE", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x21, AluOp{"PRED_SETGT", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x22, AluOp{"PRED_SETGE", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x23, AluOp{"PRED_SETNE", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x24, AluOp{"PRED_SET_INV", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x25, AluOp{"PRED_SET_POP", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x26, AluOp{"PRED_SET_CLR", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x27, AluOp{"PRED_SET_RESTORE", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x28, AluOp{"PRED_SETE_PUSH", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x29, AluOp{"PRED_SETGT_PUSH", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x2a, AluOp{"PRED_SETGE_PUSH", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x2b, AluOp{"PRED_SETNE_PUSH", 2, f::vector | f::transcendental | f::pred_set}},
    Entry{0x2c, AluOp{"KILLE", 2, f::vector | f::transcendental}},
    Entry{0x2d, AluOp{"KILLGT", 2, f::vector | f::transcendental}},
    Entry{0x2e, AluOp{"KILLGE", 2, f::vector | f::transcendental}},
    Entry{0x2f, AluOp{"KILLNE", 2, f::vector | f::transcendental}},
    Entry{0x30, AluOp{"AND_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x31, AluOp{"OR_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x32, AluOp{"XOR_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x33, AluOp{"NOT_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x34, AluOp{"ADD_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x35, AluOp{"SUB_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x36, AluOp{"MAX_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x37, AluOp{"MIN_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x38, AluOp{"MAX_UINT", 2, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x39, AluOp{"MIN_UINT", 2, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x3a, AluOp{"SETE_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x3b, AluOp{"SETGT_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x3c, AluOp{"SETGE_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x3d, AluOp{"SETNE_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x3e, AluOp{"SETGT_UINT", 2, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x3f, AluOp{"SETGE_UINT", 2, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x40, AluOp{"KILLGT_UINT", 2, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x41, AluOp{"KILLGE_UINT", 2, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x42,
          AluOp{"PRED_SETE_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x43,
          AluOp{"PRED_SETGT_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x44,
          AluOp{"PRED_SETGE_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x45,
          AluOp{"PRED_SETNE_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x46, AluOp{"KILLE_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x47, AluOp{"KILLGT_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x48, AluOp{"KILLGE_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x49, AluOp{"KILLNE_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x4a,
          AluOp{"PRED_SETE_PUSH_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x4b,
          AluOp{"PRED_SETGT_PUSH_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x4c,
          AluOp{"PRED_SETGE_PUSH_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x4d,
          AluOp{"PRED_SETNE_PUSH_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x4e,
          AluOp{"PRED_SETLT_PUSH_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x4f,
          AluOp{"PRED_SETLE_PUSH_INT",
                2,
                f::vector | f::transcendental | f::pred_set | f::int_in | f::int_out}},
    Entry{0x50, AluOp{"DOT4", 2, f::vector | f::reduction}},
    Entry{0x51, AluOp{"DOT4_IEEE", 2, f::vector | f::reduction}},
    Entry{0x52, AluOp{"CUBE", 2, f::vector | f::reduction}},
    Entry{0x53, AluOp{"MAX4", 2, f::vector | f::reduction}},
    Entry{0x54, AluOp{"GROUP_BARRIER", 2, f::vector | f::transcendental}},
    Entry{0x55, AluOp{"GROUP_SEQ_BEGIN", 2, f::vector | f::transcendental}},
    Entry{0x56, AluOp{"GROUP_SEQ_END", 2, f::vector | f::transcendental}},
    Entry{0x57, AluOp{"SET_MODE", 0, f::vector | f::transcendental}},
    Entry{0x58, AluOp{"SET_CF_IDX0", 0, f::vector | f::transcendental}},
    Entry{0x59, AluOp{"SET_CF_IDX1", 0, f::vector | f::transcendental}},
    Entry{0x5a, AluOp{"SET_LDS_SIZE", 2, f::vector | f::transcendental}},
    Entry{0x5b, AluOp{"MUL_INT24", 2, f::vector | f::transcendental}},
    Entry{0x5c, AluOp{"MULHI_INT24", 2, f::vector | f::transcendental}},
    Entry{0x60, AluOp{"MOVA_GPR_INT", 2, f::vector | f::int_in | f::int_out}},
    Entry{0x61, AluOp{"EXP_IEEE", 1, f::transcendental}},
    Entry{0x62, AluOp{"LOG_CLAMPED", 1, f::transcendental}},
    Entry{0x63, AluOp{"LOG_IEEE", 1, f::transcendental}},
    Entry{0x64, AluOp{"RECIP_CLAMPED", 1, f::transcendental}},
    Entry{0x65, AluOp{"RECIP_FF", 1, f::transcendental}},
    Entry{0x66, AluOp{"RECIP_IEEE", 1, f::transcendental}},
    Entry{0x67, AluOp{"RECIPSQRT_CLAMPED", 1, f::transcendental}},
    Entry{0x68, AluOp{"RECIPSQRT_FF", 1, f::transcendental}},
    Entry{0x69, AluOp{"RECIPSQRT_IEEE", 1, f::transcendental}},
    Entry{0x6a, AluOp{"SQRT_IEEE", 1, f::transcendental}},
    Entry{0x6b, AluOp{"FLT_TO_INT", 1, f::transcendental | f::int_out}},
    Entry{0x6c, AluOp{"INT_TO_FLT", 1, f::transcendental | f::int_in}},
    Entry{0x6d, AluOp{"UINT_TO_FLT", 1, f::transcendental | f::uint_in}},
    Entry{0x6e, AluOp{"SIN", 1, f::transcendental}},
    Entry{0x6f, AluOp{"COS", 1, f::transcendental}},
    Entry{0x70, AluOp{"ASHR_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x71, AluOp{"LSHR_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x72, AluOp{"LSHL_INT", 2, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x73, AluOp{"MULLO_INT", 2, f::transcendental | f::int_in | f::int_out}},
    Entry{0x74, AluOp{"MULHI_INT", 2, f::transcendental | f::int_in | f::int_out}},
    Entry{0x75, AluOp{"MULLO_UINT", 2, f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x76, AluOp{"MULHI_UINT", 2, f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x77, AluOp{"RECIP_INT", 1, f::transcendental | f::int_in | f::int_out}},
    Entry{0x78, AluOp{"RECIP_UINT", 1, f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x79, AluOp{"FLT_TO_UINT", 1, f::transcendental | f::uint_out}},
    Entry{0x7a, AluOp{"LDEXP_64", 2, f::vector | f::reduction}},
    Entry{0x7b, AluOp{"FRACT_64", 1, f::vector | f::reduction}},
    Entry{0x7c, AluOp{"PRED_SETGT_64", 2, f::vector | f::reduction | f::pred_set}},
    Entry{0x7d, AluOp{"PRED_SETE_64", 2, f::vector | f::reduction | f::pred_set}},
    Entry{0x7e, AluOp{"PRED_SETGE_64", 2, f::vector | f::reduction | f::pred_set}},
};

constexpr std::array op3_table{
    Entry{0x04, AluOp{"BFE_UINT", 3, f::vector | f::transcendental | f::uint_in | f::uint_out}},
    Entry{0x05, AluOp{"BFE_INT", 3, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x06, AluOp{"BFI_INT", 3, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x07, AluOp{"FMA", 3, f::vector | f::transcendental}},
    Entry{0x08, AluOp{"MULADD_64", 3, f::vector | f::reduction}},
    Entry{0x09, AluOp{"MULADD_64_M2", 3, f::vector | f::reduction}},
    Entry{0x0a, AluOp{"MULADD_64_M4", 3, f::vector | f::reduction}},
    Entry{0x0b, AluOp{"MULADD_64_D2", 3, f::vector | f::reduction}},
    Entry{0x0c, AluOp{"MUL_LIT", 3, f::transcendental}},
    Entry{0x0d, AluOp{"MUL_LIT_M2", 3, f::transcendental}},
    Entry{0x0e, AluOp{"MUL_LIT_M4", 3, f::transcendental}},
    Entry{0x0f, AluOp{"MUL_LIT_D2", 3, f::transcendental}},
    Entry{0x10, AluOp{"MULADD", 3, f::vector | f::transcendental}},
    Entry{0x11, AluOp{"MULADD_M2", 3, f::vector | f::transcendental}},
    Entry{0x12, AluOp{"MULADD_M4", 3, f::vector | f::transcendental}},
    Entry{0x13, AluOp{"MULADD_D2", 3, f::vector | f::transcendental}},
    Entry{0x14, AluOp{"MULADD_IEEE", 3, f::vector | f::transcendental}},
    Entry{0x15, AluOp{"MULADD_IEEE_M2", 3, f::vector | f::transcendental}},
    Entry{0x16, AluOp{"MULADD_IEEE_M4", 3, f::vector | f::transcendental}},
    Entry{0x17, AluOp{"MULADD_IEEE_D2", 3, f::vector | f::transcendental}},
    Entry{0x18, AluOp{"CNDE", 3, f::vector | f::transcendental}},
    Entry{0x19, AluOp{"CNDGT", 3, f::vector | f::transcendental}},
    Entry{0x1a, AluOp{"CNDGE", 3, f::vector | f::transcendental}},
    Entry{0x1c, AluOp{"CNDE_INT", 3, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x1d, AluOp{"CNDGT_INT", 3, f::vector | f::transcendental | f::int_in | f::int_out}},
    Entry{0x1e, AluOp{"CNDGE_INT", 3, f::vector | f::transcendental | f::int_in | f::int_out}},
};

template <std::size_t N>
[[nodiscard]] std::optional<AluOp> find(const std::array<Entry, N>& table, u32 inst) noexcept {
    for (const Entry& entry : table) {
        if (entry.inst == inst) {
            return entry.op;
        }
    }
    return std::nullopt;
}

} // namespace

std::optional<AluOp> alu_op2(u32 inst) noexcept {
    return find(op2_table, inst);
}

std::optional<AluOp> alu_op3(u32 inst) noexcept {
    return find(op3_table, inst);
}

} // namespace wiiu::latte
