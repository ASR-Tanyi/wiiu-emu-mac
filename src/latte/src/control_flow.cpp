// Phase 8.6 — see control_flow.hpp.

#include <wiiu/latte/control_flow.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <iterator>

namespace wiiu::latte {

namespace {

// --- names -------------------------------------------------------------------

constexpr std::array<std::string_view, 0x1E> normal_names{
    "NOP",
    "TEX",
    "VTX",
    "VTX_TC",
    "LOOP_START",
    "LOOP_END",
    "LOOP_START_DX10",
    "LOOP_START_NO_AL",
    "LOOP_CONTINUE",
    "LOOP_BREAK",
    "JUMP",
    "PUSH",
    "PUSH_ELSE",
    "ELSE",
    "POP",
    "POP_JUMP",
    "POP_PUSH",
    "POP_PUSH_ELSE",
    "CALL",
    "CALL_FS",
    "RETURN",
    "EMIT_VERTEX",
    "EMIT_CUT_VERTEX",
    "CUT_VERTEX",
    "KILL",
    "END_PROGRAM",
    "WAIT_ACK",
    "TEX_ACK",
    "VTX_ACK",
    "VTX_TC_ACK",
};

constexpr std::array<std::string_view, 8> alu_names{
    "ALU",
    "ALU_PUSH_BEFORE",
    "ALU_POP_AFTER",
    "ALU_POP2_AFTER",
    "ALU_EXT",
    "ALU_CONTINUE",
    "ALU_BREAK",
    "ALU_ELSE_AFTER",
};

[[nodiscard]] u32 raw_cf_inst(const CfInstruction& inst) noexcept {
    return (inst.word1 >> 23) & 0x7FU;
}

[[nodiscard]] u32 raw_alu_inst(const CfInstruction& inst) noexcept {
    return (inst.word1 >> 26) & 0xFU;
}

[[nodiscard]] std::string_view export_name(u32 inst) noexcept {
    switch (static_cast<ExportInst>(inst)) {
        case ExportInst::MemStream0:
            return "MEM_STREAM0";
        case ExportInst::MemStream1:
            return "MEM_STREAM1";
        case ExportInst::MemStream2:
            return "MEM_STREAM2";
        case ExportInst::MemStream3:
            return "MEM_STREAM3";
        case ExportInst::MemScratch:
            return "MEM_SCRATCH";
        case ExportInst::MemReduction:
            return "MEM_REDUCTION";
        case ExportInst::MemRing:
            return "MEM_RING";
        case ExportInst::Export:
            return "EXP";
        case ExportInst::ExportDone:
            return "EXP_DONE";
        case ExportInst::MemExport:
            return "MEM_EXPORT";
    }
    return {};
}

/// Whether R700 has the instruction: nothing past VTX_TC_ACK in the common
/// form, the ten exports, and every ALU clause but the reserved 0xC.
[[nodiscard]] bool known(const CfInstruction& inst) noexcept {
    switch (inst.type()) {
        case CfInstType::Normal:
            return raw_cf_inst(inst) < normal_names.size();
        case CfInstType::Export:
            return !export_name(raw_cf_inst(inst)).empty();
        case CfInstType::Alu:
        case CfInstType::AluExtended:
            return raw_alu_inst(inst) != static_cast<u32>(AluCfInst::AluExtended);
    }
    return false;
}

/// COUNT and COUNT_3, as the number of fetches.
[[nodiscard]] u32 fetch_count(const CfInstruction& inst) noexcept {
    const CfWord1 word1 = inst.normal1();
    return (word1.count | (word1.count_3 << 3)) + 1;
}

/// The common form's instructions whose ADDR is another control-flow
/// instruction.
[[nodiscard]] bool jumps(const CfInstruction& inst) noexcept {
    if (inst.type() != CfInstType::Normal) {
        return false;
    }
    switch (inst.normal1().cf_inst) {
        case CfInst::LoopStart:
        case CfInst::LoopEnd:
        case CfInst::LoopStartDx10:
        case CfInst::LoopStartNoAl:
        case CfInst::LoopContinue:
        case CfInst::LoopBreak:
        case CfInst::Jump:
        case CfInst::Else:
        case CfInst::PopJump:
        case CfInst::Call:
            return true;
        default:
            return false;
    }
}

// --- the walk ----------------------------------------------------------------

struct Walker {
    std::span<const u32> program;
    ControlFlow& out;
    std::string& error;
    std::vector<u32> calls; ///< subroutine addresses, in the order first called

    [[nodiscard]] u32 instructions() const noexcept { return static_cast<u32>(program.size() / 2); }

    [[nodiscard]] u32 bytes() const noexcept { return static_cast<u32>(program.size() * 4); }

    /// The clause `inst` runs, if any, checked to be inside the program.
    [[nodiscard]] bool clause(const CfInstruction& inst) {
        Clause found{.cf_index = inst.index};
        if (inst.is_alu()) {
            found.kind = ClauseKind::Alu;
            found.address = inst.alu0().addr;
            found.count = inst.alu1().count + 1;
        } else if (inst.is(CfInst::Tex)) {
            found.kind = ClauseKind::Tex;
            found.address = inst.word0;
            found.count = fetch_count(inst);
        } else if (inst.is(CfInst::Vtx) || inst.is(CfInst::VtxTc)) {
            found.kind = ClauseKind::Vtx;
            found.address = inst.word0;
            found.count = fetch_count(inst);
        } else {
            return true;
        }
        const u64 end = (static_cast<u64>(found.address) * 8) + found.size_bytes();
        if (end > bytes()) {
            error = fmt::format("instruction {} ({}) runs a clause of {} bytes at {:#x}, past the "
                                "end of the {}-byte program",
                                inst.index,
                                cf_inst_name(inst),
                                found.size_bytes(),
                                static_cast<u64>(found.address) * 8,
                                bytes());
            return false;
        }
        out.clauses.push_back(found);
        return true;
    }

    /// Walk from `start` to the end: END_OF_PROGRAM, or for a subroutine its
    /// RETURN.
    [[nodiscard]] bool walk(u32 start, bool until_return, std::vector<CfInstruction>& list) {
        for (u32 index = start;; ++index) {
            if (index >= instructions()) {
                error = fmt::format("{} at {} runs off the end of the {}-byte program without {}",
                                    until_return ? "a subroutine" : "the program",
                                    start,
                                    bytes(),
                                    until_return ? "a RETURN" : "END_OF_PROGRAM");
                return false;
            }
            const std::size_t at = std::size_t{2} * index;
            const CfInstruction inst{
                .index = index, .word0 = program[at], .word1 = program[at + 1]};
            if (!known(inst)) {
                error = fmt::format("instruction {} ({:#010x} {:#010x}) is not an R700 "
                                    "control-flow instruction",
                                    index,
                                    inst.word0,
                                    inst.word1);
                return false;
            }
            if (jumps(inst) && inst.word0 >= instructions()) {
                error = fmt::format("instruction {} ({}) jumps to {}, past the end of the program",
                                    index,
                                    cf_inst_name(inst),
                                    inst.word0);
                return false;
            }
            if (!clause(inst)) {
                return false;
            }
            if (inst.is(CfInst::Call) &&
                std::find(calls.begin(), calls.end(), inst.word0) == calls.end()) {
                calls.push_back(inst.word0);
            }
            list.push_back(inst);

            if (until_return) {
                if (inst.is(CfInst::Return)) {
                    return true;
                }
                if (inst.ends_program()) {
                    error = fmt::format("a subroutine at {} reaches END_OF_PROGRAM at {} before "
                                        "its RETURN",
                                        start,
                                        index);
                    return false;
                }
            } else if (inst.ends_program()) {
                return true;
            }
        }
    }
};

// --- text --------------------------------------------------------------------

using Out = std::back_insert_iterator<std::string>;

[[nodiscard]] char select_char(u32 sel) noexcept {
    constexpr std::string_view chars = "xyzw01?_";
    return chars[sel & 7U];
}

void condition(Out out, const CfInstruction& inst) {
    const CfWord1 word1 = inst.normal1();
    if (word1.cond == 0) {
        return;
    }
    constexpr std::array<std::string_view, 4> names{"", "FALSE", "BOOL", "NOT_BOOL"};
    fmt::format_to(out, " CND({}) CF_CONST({})", names[word1.cond & 3U], word1.cf_const.value());
}

void pop_count(Out out, const CfWord1 word1) {
    if (word1.pop_count != 0) {
        fmt::format_to(out, " POP_CNT({})", word1.pop_count.value());
    }
}

void valid_pixel(Out out, const CfWord1 word1) {
    if (word1.valid_pixel_mode) {
        fmt::format_to(out, " VALID_PIX");
    }
}

void no_barrier(Out out, bool barrier) {
    if (!barrier) {
        fmt::format_to(out, " NO_BARRIER");
    }
}

void normal_text(Out out, const CfInstruction& inst) {
    const CfWord1 word1 = inst.normal1();
    const CfInst id = word1.cf_inst;
    switch (id) {
        case CfInst::Tex:
            fmt::format_to(out, ": ADDR({}) CNT({})", inst.word0, fetch_count(inst));
            no_barrier(out, word1.barrier);
            condition(out, inst);
            if (word1.whole_quad_mode) {
                fmt::format_to(out, " WHOLE_QUAD");
            }
            valid_pixel(out, word1);
            return;
        case CfInst::Vtx:
        case CfInst::VtxTc:
            fmt::format_to(out, ": ADDR({}) CNT({})", inst.word0, fetch_count(inst));
            no_barrier(out, word1.barrier);
            condition(out, inst);
            return;
        case CfInst::LoopStart:
        case CfInst::LoopStartDx10:
        case CfInst::LoopStartNoAl:
        case CfInst::LoopEnd:
        case CfInst::LoopContinue:
        case CfInst::LoopBreak:
            condition(out, inst);
            if ((id == CfInst::LoopStart || id == CfInst::LoopEnd) && word1.cond == 0) {
                fmt::format_to(out, " CF_CONST({})", word1.cf_const.value());
            }
            if (id == CfInst::LoopEnd) {
                fmt::format_to(out, " PASS_JUMP_ADDR({})", inst.word0);
            } else if (id == CfInst::LoopContinue || id == CfInst::LoopBreak) {
                fmt::format_to(out, " ADDR({})", inst.word0);
            } else {
                fmt::format_to(out, " FAIL_JUMP_ADDR({})", inst.word0);
            }
            pop_count(out, word1);
            valid_pixel(out, word1);
            no_barrier(out, word1.barrier);
            return;
        case CfInst::Jump:
        case CfInst::Else:
        case CfInst::Call:
        case CfInst::CallFs:
        case CfInst::Return:
        case CfInst::PopJump:
            if (id == CfInst::Call && word1.call_count != 0) {
                fmt::format_to(out, " CALL_COUNT({})", word1.call_count.value());
            }
            condition(out, inst);
            pop_count(out, word1);
            if (id == CfInst::Call || id == CfInst::Else || id == CfInst::Jump ||
                id == CfInst::PopJump) {
                fmt::format_to(out, " ADDR({})", inst.word0);
            }
            valid_pixel(out, word1);
            no_barrier(out, word1.barrier);
            return;
        case CfInst::EmitVertex:
        case CfInst::EmitCutVertex:
        case CfInst::CutVertex:
            no_barrier(out, word1.barrier);
            return;
        case CfInst::Push:
        case CfInst::PushElse:
        case CfInst::Kill:
            condition(out, inst);
            [[fallthrough]];
        case CfInst::Pop:
        case CfInst::PopPush:
        case CfInst::PopPushElse:
            pop_count(out, word1);
            valid_pixel(out, word1);
            return;
        case CfInst::Nop:
        case CfInst::EndProgram:
        case CfInst::WaitAck:
        case CfInst::TexAck:
        case CfInst::VtxAck:
        case CfInst::VtxTcAck:
            return;
    }
}

void export_text(Out out, const CfInstruction& inst) {
    const CfExportWord0 word0 = inst.export0();
    const CfExportWord1 word1 = inst.export1();
    const ExportInst id = word1.cf_inst;
    const u32 type = word0.type;
    const u32 base = word0.array_base;
    const bool exp = id == ExportInst::Export || id == ExportInst::ExportDone;
    const bool read = type == static_cast<u32>(MemExportType::Read) ||
                      type == static_cast<u32>(MemExportType::ReadIndexed);
    const bool stream = static_cast<u32>(id) >= static_cast<u32>(ExportInst::MemStream0) &&
                        static_cast<u32>(id) <= static_cast<u32>(ExportInst::MemStream3);

    fmt::format_to(out, ":");
    if (exp) {
        switch (static_cast<ExportType>(type)) {
            case ExportType::Pixel:
                fmt::format_to(out, " PIX{}", base);
                break;
            case ExportType::Position:
                fmt::format_to(out, " POS{}", base - export_position_base);
                break;
            case ExportType::Parameter:
                fmt::format_to(out, " PARAM{}", base);
                break;
            default:
                fmt::format_to(out, " INVALID{}", base);
                break;
        }
    } else if (stream && read) {
        fmt::format_to(out, " INVALID_READ");
    } else {
        fmt::format_to(out, "{}", read ? " READ(" : " WRITE(");
        const bool vec4 = id == ExportInst::MemScratch || id == ExportInst::MemReduction;
        fmt::format_to(out, "{}", base * (vec4 ? 16U : 4U));
        if (type == static_cast<u32>(MemExportType::WriteIndexed) ||
            type == static_cast<u32>(MemExportType::ReadIndexed)) {
            fmt::format_to(out, " + R{}", word0.index_gpr.value());
        }
        fmt::format_to(out, ")");
    }

    if (word0.rw_rel) {
        fmt::format_to(out, ", R[AL + {}]", word0.rw_gpr.value());
    } else {
        fmt::format_to(out, ", R{}", word0.rw_gpr.value());
    }

    if (exp) {
        const CfExportWord1Swizzle swizzle{.raw = inst.word1};
        fmt::format_to(out,
                       ".{}{}{}{}",
                       select_char(swizzle.sel_x),
                       select_char(swizzle.sel_y),
                       select_char(swizzle.sel_z),
                       select_char(swizzle.sel_w));
    } else {
        const CfExportWord1Buffer buffer{.raw = inst.word1};
        const u32 mask = buffer.comp_mask;
        fmt::format_to(out,
                       ".{}{}{}{} ARRAY_SIZE({})",
                       (mask & 1U) != 0 ? 'x' : '_',
                       (mask & 2U) != 0 ? 'y' : '_',
                       (mask & 4U) != 0 ? 'z' : '_',
                       (mask & 8U) != 0 ? 'w' : '_',
                       buffer.array_size.value());
    }

    if (word0.elem_size != 0) {
        fmt::format_to(out, " ELEM_SIZE({})", word0.elem_size.value());
    }
    if (word1.burst_count != 0) {
        fmt::format_to(out, " BURSTCNT({})", word1.burst_count.value());
    }
    no_barrier(out, word1.barrier);
    if (word1.whole_quad_mode) {
        fmt::format_to(out, " WHOLE_QUAD");
    }
    if (word1.valid_pixel_mode) {
        fmt::format_to(out, " VALID_PIX");
    }
}

void kcache(Out out, u32 id, KcacheMode mode, u32 bank, u32 addr) {
    switch (mode) {
        case KcacheMode::Nop:
            return;
        case KcacheMode::Lock1:
            fmt::format_to(out, " KCACHE{}(CB{}:{}-{})", id, bank, 16 * addr, (16 * addr) + 15);
            return;
        case KcacheMode::Lock2:
            fmt::format_to(out, " KCACHE{}(CB{}:{}-{})", id, bank, 16 * addr, (16 * addr) + 31);
            return;
        case KcacheMode::LockLoopIndex:
            fmt::format_to(
                out, " KCACHE{}(CB{}:AL+{}-AL+{})", id, bank, 16 * addr, (16 * addr) + 31);
            return;
    }
}

void alu_text(Out out, const CfInstruction& inst) {
    const CfAluWord0 word0 = inst.alu0();
    const CfAluWord1 word1 = inst.alu1();
    fmt::format_to(out, ": ADDR({}) CNT({})", word0.addr.value(), word1.count + 1);
    no_barrier(out, word1.barrier);
    if (word1.whole_quad_mode) {
        fmt::format_to(out, " WHOLE_QUAD");
    }
    kcache(out, 0, word0.kcache_mode0, word0.kcache_bank0, word1.kcache_addr0);
    kcache(out, 1, word1.kcache_mode1, word0.kcache_bank1, word1.kcache_addr1);
}

void lines(std::string& out, const std::vector<CfInstruction>& list, const ClauseLines& after) {
    std::string indent;
    for (const CfInstruction& inst : list) {
        fmt::format_to(
            std::back_inserter(out), "{}{:02} {}\n", indent, inst.index, disassemble(inst));
        if (after) {
            after(out, inst, indent);
        }
        if (inst.is(CfInst::LoopStart) || inst.is(CfInst::LoopStartDx10) ||
            inst.is(CfInst::LoopStartNoAl)) {
            indent += "  ";
        } else if (inst.is(CfInst::LoopEnd) && indent.size() >= 2) {
            indent.resize(indent.size() - 2);
        }
    }
}

} // namespace

bool CfInstruction::ends_program() const noexcept {
    return !is_alu() && normal1().end_of_program;
}

std::string_view cf_inst_name(const CfInstruction& instruction) noexcept {
    switch (instruction.type()) {
        case CfInstType::Normal: {
            const u32 id = raw_cf_inst(instruction);
            return id < normal_names.size() ? normal_names[id] : std::string_view{"UNKNOWN"};
        }
        case CfInstType::Export: {
            const std::string_view name = export_name(raw_cf_inst(instruction));
            return name.empty() ? std::string_view{"UNKNOWN"} : name;
        }
        case CfInstType::Alu:
        case CfInstType::AluExtended:
            return alu_names[raw_alu_inst(instruction) - 8];
    }
    return "UNKNOWN";
}

bool decode_control_flow(std::span<const u32> program,
                         ProgramShape shape,
                         ControlFlow& out,
                         std::string& error) {
    out = ControlFlow{};
    Walker walker{.program = program, .out = out, .error = error, .calls = {}};
    if (!walker.walk(0, shape == ProgramShape::FetchShader, out.main)) {
        return false;
    }
    // A subroutine may call others; the list grows as they are found.
    for (std::size_t next = 0; next < walker.calls.size(); ++next) {
        Subroutine subroutine{.address = walker.calls[next], .instructions = {}};
        if (!walker.walk(subroutine.address, true, subroutine.instructions)) {
            return false;
        }
        out.subroutines.push_back(std::move(subroutine));
    }
    return true;
}

std::string disassemble(const CfInstruction& instruction) {
    std::string text{cf_inst_name(instruction)};
    const Out out = std::back_inserter(text);
    switch (instruction.type()) {
        case CfInstType::Normal:
            normal_text(out, instruction);
            break;
        case CfInstType::Export:
            export_text(out, instruction);
            break;
        case CfInstType::Alu:
        case CfInstType::AluExtended:
            alu_text(out, instruction);
            break;
    }
    return text;
}

std::string disassemble(const ControlFlow& flow, const ClauseLines& after) {
    std::string out;
    lines(out, flow.main, after);
    if (!flow.main.empty() && flow.main.back().ends_program()) {
        out += "END_OF_PROGRAM\n";
    }
    for (const Subroutine& subroutine : flow.subroutines) {
        fmt::format_to(std::back_inserter(out), "\nSUBROUTINE {:02}\n", subroutine.address);
        lines(out, subroutine.instructions, after);
    }
    return out;
}

} // namespace wiiu::latte
