// Phase 4.2 — the instruction table.
//
// One list. The opcode enum, the encoding patterns, the operand forms and the
// mnemonics all come out of it, so the decoder and the disassembler cannot
// drift apart, and the interpreter (Phase 4.4) and JIT (Phase 12) will hang
// their dispatch off the same identifiers.
//
// Adding an instruction is one line in WIIU_PPC_OPCODES.
#pragma once

#include <wiiu/common/types.hpp>
#include <wiiu/cpu/instruction.hpp>

#include <span>
#include <string_view>

namespace wiiu::cpu {

/// How a decoded instruction's operands are laid out, for formatting and for
/// the interpreter's argument extraction.
enum class Form : u8 {
    Invalid,
    XO,           ///< rD, rA, rB          add
    XO_AB,        ///< rD, rA              neg, addme
    DArith,       ///< rD, rA, SIMM        addi
    DLogic,       ///< rA, rS, UIMM        ori
    DLoad,        ///< rD, d(rA)           lwz
    DStore,       ///< rS, d(rA)           stw
    DLoadFp,      ///< frD, d(rA)          lfs
    DStoreFp,     ///< frS, d(rA)          stfs
    DCmp,         ///< crfD, L, rA, SIMM   cmpi
    DTrap,        ///< TO, rA, SIMM        twi
    XLogic,       ///< rA, rS, rB          and, or
    XSourceA,     ///< rA, rS              extsb, cntlzw
    XShift,       ///< rA, rS, SH          srawi
    XCmp,         ///< crfD, L, rA, rB     cmp
    XLoad,        ///< rD, rA, rB          lwzx
    XStore,       ///< rS, rA, rB          stwx
    XLoadFp,      ///< frD, rA, rB         lfsx
    XStoreFp,     ///< frS, rA, rB         stfsx
    XCache,       ///< rA, rB              dcbf
    XTrap,        ///< TO, rA, rB          tw
    XNone,        ///< (none)              sync, isync, rfi
    XMsrFrom,     ///< rD                  mfmsr
    XMsrTo,       ///< rS                  mtmsr
    XlBranch,     ///< BO, BI              bclr, bcctr
    XlCr,         ///< crbD, crbA, crbB    cror
    XlMcrf,       ///< crfD, crfS          mcrf
    XfxSprFrom,   ///< rD, SPR             mfspr
    XfxSprTo,     ///< SPR, rS             mtspr
    XfxMfcr,      ///< rD                  mfcr
    XfxMtcrf,     ///< CRM, rS             mtcrf
    XfxMftb,      ///< rD, TBR             mftb
    MImm,         ///< rA, rS, SH, MB, ME  rlwinm
    MReg,         ///< rA, rS, rB, MB, ME  rlwnm
    IBranch,      ///< target              b
    BBranch,      ///< BO, BI, target      bc
    Aabc,         ///< frD, frA, frC, frB  fmadd
    Aab,          ///< frD, frA, frB       fadd
    Aac,          ///< frD, frA, frC       fmul
    Ab,           ///< frD, frB            fres
    XFp2,         ///< frD, frB            fmr, fneg
    XFpCmp,       ///< crfD, frA, frB      fcmpu
    XMffs,        ///< frD                 mffs
    XMtfsf,       ///< FM, frB             mtfsf
    XMcrfs,       ///< crfD, crfS          mcrfs
    PsQuantized,  ///< frD, d(rA), W, I    psq_l
    PsQuantizedX, ///< frD, rA, rB, W, I   psq_lx
    Sc,           ///< (none)              sc
};

/// Suffix bits the mnemonic grows when set.
enum Flags : u8 {
    FlagNone = 0,
    FlagRc = 1U << 0, ///< trailing '.' — update CR0
    FlagOe = 1U << 1, ///< trailing 'o' — update XER overflow
    FlagLk = 1U << 2, ///< trailing 'l' — write LR
    FlagAa = 1U << 3, ///< trailing 'a' — absolute target
};

// --- encoding helpers ------------------------------------------------------
// Each expands to a (pattern, mask) pair.

#define WIIU_PPC_D(op) (static_cast<u32>(op) << 26), 0xFC000000U
#define WIIU_PPC_X(op, xo) ((static_cast<u32>(op) << 26) | ((xo) << 1)), 0xFC0007FEU
#define WIIU_PPC_XR(op, xo) ((static_cast<u32>(op) << 26) | ((xo) << 1)), 0xFC0007FFU
/// X-form where the record bit is part of the encoding, not a suffix (stwcx.).
#define WIIU_PPC_XRC(op, xo) ((static_cast<u32>(op) << 26) | ((xo) << 1) | 1U), 0xFC0007FFU
#define WIIU_PPC_XO(op, xo) ((static_cast<u32>(op) << 26) | ((xo) << 1)), 0xFC0003FEU
#define WIIU_PPC_A(op, xo) ((static_cast<u32>(op) << 26) | ((xo) << 1)), 0xFC00003EU
#define WIIU_PPC_EXACT(word) (word), 0xFFFFFFFFU

// --- the table -------------------------------------------------------------
//
//   X(id, mnemonic, encoding, form, flags)
//
// Ordered by category. Where two entries could match the same word, the more
// specific one (larger mask) is listed and the decoder prefers it.

// NOLINTBEGIN(*-macro-parentheses)
#define WIIU_PPC_OPCODES(X)                                                                        \
    /* integer arithmetic, XO-form */                                                              \
    X(Add, "add", WIIU_PPC_XO(31, 266), Form::XO, FlagOe | FlagRc)                                 \
    X(Addc, "addc", WIIU_PPC_XO(31, 10), Form::XO, FlagOe | FlagRc)                                \
    X(Adde, "adde", WIIU_PPC_XO(31, 138), Form::XO, FlagOe | FlagRc)                               \
    X(Addme, "addme", WIIU_PPC_XO(31, 234), Form::XO_AB, FlagOe | FlagRc)                          \
    X(Addze, "addze", WIIU_PPC_XO(31, 202), Form::XO_AB, FlagOe | FlagRc)                          \
    X(Divw, "divw", WIIU_PPC_XO(31, 491), Form::XO, FlagOe | FlagRc)                               \
    X(Divwu, "divwu", WIIU_PPC_XO(31, 459), Form::XO, FlagOe | FlagRc)                             \
    X(Mulhw, "mulhw", WIIU_PPC_XO(31, 75), Form::XO, FlagRc)                                       \
    X(Mulhwu, "mulhwu", WIIU_PPC_XO(31, 11), Form::XO, FlagRc)                                     \
    X(Mullw, "mullw", WIIU_PPC_XO(31, 235), Form::XO, FlagOe | FlagRc)                             \
    X(Neg, "neg", WIIU_PPC_XO(31, 104), Form::XO_AB, FlagOe | FlagRc)                              \
    X(Subf, "subf", WIIU_PPC_XO(31, 40), Form::XO, FlagOe | FlagRc)                                \
    X(Subfc, "subfc", WIIU_PPC_XO(31, 8), Form::XO, FlagOe | FlagRc)                               \
    X(Subfe, "subfe", WIIU_PPC_XO(31, 136), Form::XO, FlagOe | FlagRc)                             \
    X(Subfme, "subfme", WIIU_PPC_XO(31, 232), Form::XO_AB, FlagOe | FlagRc)                        \
    X(Subfze, "subfze", WIIU_PPC_XO(31, 200), Form::XO_AB, FlagOe | FlagRc)                        \
    /* integer arithmetic, D-form */                                                               \
    X(Addi, "addi", WIIU_PPC_D(14), Form::DArith, FlagNone)                                        \
    X(Addic, "addic", WIIU_PPC_D(12), Form::DArith, FlagNone)                                      \
    X(AddicRc, "addic.", WIIU_PPC_D(13), Form::DArith, FlagNone)                                   \
    X(Addis, "addis", WIIU_PPC_D(15), Form::DArith, FlagNone)                                      \
    X(Mulli, "mulli", WIIU_PPC_D(7), Form::DArith, FlagNone)                                       \
    X(Subfic, "subfic", WIIU_PPC_D(8), Form::DArith, FlagNone)                                     \
    /* logical, X-form */                                                                          \
    X(And, "and", WIIU_PPC_X(31, 28), Form::XLogic, FlagRc)                                        \
    X(Andc, "andc", WIIU_PPC_X(31, 60), Form::XLogic, FlagRc)                                      \
    X(Eqv, "eqv", WIIU_PPC_X(31, 284), Form::XLogic, FlagRc)                                       \
    X(Nand, "nand", WIIU_PPC_X(31, 476), Form::XLogic, FlagRc)                                     \
    X(Nor, "nor", WIIU_PPC_X(31, 124), Form::XLogic, FlagRc)                                       \
    X(Or, "or", WIIU_PPC_X(31, 444), Form::XLogic, FlagRc)                                         \
    X(Orc, "orc", WIIU_PPC_X(31, 412), Form::XLogic, FlagRc)                                       \
    X(Xor, "xor", WIIU_PPC_X(31, 316), Form::XLogic, FlagRc)                                       \
    X(Slw, "slw", WIIU_PPC_X(31, 24), Form::XLogic, FlagRc)                                        \
    X(Sraw, "sraw", WIIU_PPC_X(31, 792), Form::XLogic, FlagRc)                                     \
    X(Srawi, "srawi", WIIU_PPC_X(31, 824), Form::XShift, FlagRc)                                   \
    X(Srw, "srw", WIIU_PPC_X(31, 536), Form::XLogic, FlagRc)                                       \
    X(Cntlzw, "cntlzw", WIIU_PPC_X(31, 26), Form::XSourceA, FlagRc)                                \
    X(Extsb, "extsb", WIIU_PPC_X(31, 954), Form::XSourceA, FlagRc)                                 \
    X(Extsh, "extsh", WIIU_PPC_X(31, 922), Form::XSourceA, FlagRc)                                 \
    /* logical, D-form */                                                                          \
    X(Andi, "andi.", WIIU_PPC_D(28), Form::DLogic, FlagNone)                                       \
    X(Andis, "andis.", WIIU_PPC_D(29), Form::DLogic, FlagNone)                                     \
    X(Ori, "ori", WIIU_PPC_D(24), Form::DLogic, FlagNone)                                          \
    X(Oris, "oris", WIIU_PPC_D(25), Form::DLogic, FlagNone)                                        \
    X(Xori, "xori", WIIU_PPC_D(26), Form::DLogic, FlagNone)                                        \
    X(Xoris, "xoris", WIIU_PPC_D(27), Form::DLogic, FlagNone)                                      \
    /* compare */                                                                                  \
    X(Cmp, "cmp", WIIU_PPC_XR(31, 0), Form::XCmp, FlagNone)                                        \
    X(Cmpl, "cmpl", WIIU_PPC_XR(31, 32), Form::XCmp, FlagNone)                                     \
    X(Cmpi, "cmpi", WIIU_PPC_D(11), Form::DCmp, FlagNone)                                          \
    X(Cmpli, "cmpli", WIIU_PPC_D(10), Form::DCmp, FlagNone)                                        \
    /* rotate */                                                                                   \
    X(Rlwimi, "rlwimi", WIIU_PPC_D(20), Form::MImm, FlagRc)                                        \
    X(Rlwinm, "rlwinm", WIIU_PPC_D(21), Form::MImm, FlagRc)                                        \
    X(Rlwnm, "rlwnm", WIIU_PPC_D(23), Form::MReg, FlagRc)                                          \
    /* branches */                                                                                 \
    X(B, "b", WIIU_PPC_D(18), Form::IBranch, FlagLk | FlagAa)                                      \
    X(Bc, "bc", WIIU_PPC_D(16), Form::BBranch, FlagLk | FlagAa)                                    \
    X(Bclr, "bclr", WIIU_PPC_X(19, 16), Form::XlBranch, FlagLk)                                    \
    X(Bcctr, "bcctr", WIIU_PPC_X(19, 528), Form::XlBranch, FlagLk)                                 \
    /* condition register */                                                                       \
    X(Crand, "crand", WIIU_PPC_XR(19, 257), Form::XlCr, FlagNone)                                  \
    X(Crandc, "crandc", WIIU_PPC_XR(19, 129), Form::XlCr, FlagNone)                                \
    X(Creqv, "creqv", WIIU_PPC_XR(19, 289), Form::XlCr, FlagNone)                                  \
    X(Crnand, "crnand", WIIU_PPC_XR(19, 225), Form::XlCr, FlagNone)                                \
    X(Crnor, "crnor", WIIU_PPC_XR(19, 33), Form::XlCr, FlagNone)                                   \
    X(Cror, "cror", WIIU_PPC_XR(19, 449), Form::XlCr, FlagNone)                                    \
    X(Crorc, "crorc", WIIU_PPC_XR(19, 417), Form::XlCr, FlagNone)                                  \
    X(Crxor, "crxor", WIIU_PPC_XR(19, 193), Form::XlCr, FlagNone)                                  \
    X(Mcrf, "mcrf", WIIU_PPC_XR(19, 0), Form::XlMcrf, FlagNone)                                    \
    /* special registers */                                                                        \
    X(Mfspr, "mfspr", WIIU_PPC_XR(31, 339), Form::XfxSprFrom, FlagNone)                            \
    X(Mtspr, "mtspr", WIIU_PPC_XR(31, 467), Form::XfxSprTo, FlagNone)                              \
    X(Mftb, "mftb", WIIU_PPC_XR(31, 371), Form::XfxMftb, FlagNone)                                 \
    X(Mfcr, "mfcr", WIIU_PPC_XR(31, 19), Form::XfxMfcr, FlagNone)                                  \
    X(Mtcrf, "mtcrf", WIIU_PPC_XR(31, 144), Form::XfxMtcrf, FlagNone)                              \
    X(Mfmsr, "mfmsr", WIIU_PPC_XR(31, 83), Form::XMsrFrom, FlagNone)                               \
    X(Mtmsr, "mtmsr", WIIU_PPC_XR(31, 146), Form::XMsrTo, FlagNone)                                \
    /* loads, D-form */                                                                            \
    X(Lbz, "lbz", WIIU_PPC_D(34), Form::DLoad, FlagNone)                                           \
    X(Lbzu, "lbzu", WIIU_PPC_D(35), Form::DLoad, FlagNone)                                         \
    X(Lha, "lha", WIIU_PPC_D(42), Form::DLoad, FlagNone)                                           \
    X(Lhau, "lhau", WIIU_PPC_D(43), Form::DLoad, FlagNone)                                         \
    X(Lhz, "lhz", WIIU_PPC_D(40), Form::DLoad, FlagNone)                                           \
    X(Lhzu, "lhzu", WIIU_PPC_D(41), Form::DLoad, FlagNone)                                         \
    X(Lwz, "lwz", WIIU_PPC_D(32), Form::DLoad, FlagNone)                                           \
    X(Lwzu, "lwzu", WIIU_PPC_D(33), Form::DLoad, FlagNone)                                         \
    X(Lmw, "lmw", WIIU_PPC_D(46), Form::DLoad, FlagNone)                                           \
    /* stores, D-form */                                                                           \
    X(Stb, "stb", WIIU_PPC_D(38), Form::DStore, FlagNone)                                          \
    X(Stbu, "stbu", WIIU_PPC_D(39), Form::DStore, FlagNone)                                        \
    X(Sth, "sth", WIIU_PPC_D(44), Form::DStore, FlagNone)                                          \
    X(Sthu, "sthu", WIIU_PPC_D(45), Form::DStore, FlagNone)                                        \
    X(Stw, "stw", WIIU_PPC_D(36), Form::DStore, FlagNone)                                          \
    X(Stwu, "stwu", WIIU_PPC_D(37), Form::DStore, FlagNone)                                        \
    X(Stmw, "stmw", WIIU_PPC_D(47), Form::DStore, FlagNone)                                        \
    /* loads and stores, indexed */                                                                \
    X(Lbzx, "lbzx", WIIU_PPC_XR(31, 87), Form::XLoad, FlagNone)                                    \
    X(Lbzux, "lbzux", WIIU_PPC_XR(31, 119), Form::XLoad, FlagNone)                                 \
    X(Lhax, "lhax", WIIU_PPC_XR(31, 343), Form::XLoad, FlagNone)                                   \
    X(Lhzx, "lhzx", WIIU_PPC_XR(31, 279), Form::XLoad, FlagNone)                                   \
    X(Lhzux, "lhzux", WIIU_PPC_XR(31, 311), Form::XLoad, FlagNone)                                 \
    X(Lwzx, "lwzx", WIIU_PPC_XR(31, 23), Form::XLoad, FlagNone)                                    \
    X(Lwzux, "lwzux", WIIU_PPC_XR(31, 55), Form::XLoad, FlagNone)                                  \
    X(Stbx, "stbx", WIIU_PPC_XR(31, 215), Form::XStore, FlagNone)                                  \
    X(Sthx, "sthx", WIIU_PPC_XR(31, 407), Form::XStore, FlagNone)                                  \
    X(Stwx, "stwx", WIIU_PPC_XR(31, 151), Form::XStore, FlagNone)                                  \
    X(Stwux, "stwux", WIIU_PPC_XR(31, 183), Form::XStore, FlagNone)                                \
    X(Lhbrx, "lhbrx", WIIU_PPC_XR(31, 790), Form::XLoad, FlagNone)                                 \
    X(Lwbrx, "lwbrx", WIIU_PPC_XR(31, 534), Form::XLoad, FlagNone)                                 \
    X(Sthbrx, "sthbrx", WIIU_PPC_XR(31, 918), Form::XStore, FlagNone)                              \
    X(Stwbrx, "stwbrx", WIIU_PPC_XR(31, 662), Form::XStore, FlagNone)                              \
    /* reservations -- buggy on real Espresso; see plan Phase 4.9 */                               \
    X(Lwarx, "lwarx", WIIU_PPC_XR(31, 20), Form::XLoad, FlagNone)                                  \
    X(Stwcx, "stwcx.", WIIU_PPC_XRC(31, 150), Form::XStore, FlagNone)                              \
    /* cache and sync -- the CPU/GPU coherency model lives on these */                             \
    X(Dcbf, "dcbf", WIIU_PPC_XR(31, 86), Form::XCache, FlagNone)                                   \
    X(Dcbi, "dcbi", WIIU_PPC_XR(31, 470), Form::XCache, FlagNone)                                  \
    X(Dcbst, "dcbst", WIIU_PPC_XR(31, 54), Form::XCache, FlagNone)                                 \
    X(Dcbt, "dcbt", WIIU_PPC_XR(31, 278), Form::XCache, FlagNone)                                  \
    X(Dcbtst, "dcbtst", WIIU_PPC_XR(31, 246), Form::XCache, FlagNone)                              \
    X(Dcbz, "dcbz", WIIU_PPC_XR(31, 1014), Form::XCache, FlagNone)                                 \
    X(Icbi, "icbi", WIIU_PPC_XR(31, 982), Form::XCache, FlagNone)                                  \
    X(Isync, "isync", WIIU_PPC_XR(19, 150), Form::XNone, FlagNone)                                 \
    X(Sync, "sync", WIIU_PPC_XR(31, 598), Form::XNone, FlagNone)                                   \
    X(Eieio, "eieio", WIIU_PPC_XR(31, 854), Form::XNone, FlagNone)                                 \
    /* system */                                                                                   \
    X(Sc, "sc", WIIU_PPC_EXACT(0x44000002), Form::Sc, FlagNone)                                    \
    X(Rfi, "rfi", WIIU_PPC_XR(19, 50), Form::XNone, FlagNone)                                      \
    X(Tw, "tw", WIIU_PPC_XR(31, 4), Form::XTrap, FlagNone)                                         \
    X(Twi, "twi", WIIU_PPC_D(3), Form::DTrap, FlagNone)                                            \
    /* floating point loads and stores */                                                          \
    X(Lfs, "lfs", WIIU_PPC_D(48), Form::DLoadFp, FlagNone)                                         \
    X(Lfsu, "lfsu", WIIU_PPC_D(49), Form::DLoadFp, FlagNone)                                       \
    X(Lfd, "lfd", WIIU_PPC_D(50), Form::DLoadFp, FlagNone)                                         \
    X(Lfdu, "lfdu", WIIU_PPC_D(51), Form::DLoadFp, FlagNone)                                       \
    X(Stfs, "stfs", WIIU_PPC_D(52), Form::DStoreFp, FlagNone)                                      \
    X(Stfsu, "stfsu", WIIU_PPC_D(53), Form::DStoreFp, FlagNone)                                    \
    X(Stfd, "stfd", WIIU_PPC_D(54), Form::DStoreFp, FlagNone)                                      \
    X(Stfdu, "stfdu", WIIU_PPC_D(55), Form::DStoreFp, FlagNone)                                    \
    X(Lfsx, "lfsx", WIIU_PPC_XR(31, 535), Form::XLoadFp, FlagNone)                                 \
    X(Lfdx, "lfdx", WIIU_PPC_XR(31, 599), Form::XLoadFp, FlagNone)                                 \
    X(Stfsx, "stfsx", WIIU_PPC_XR(31, 663), Form::XStoreFp, FlagNone)                              \
    X(Stfdx, "stfdx", WIIU_PPC_XR(31, 727), Form::XStoreFp, FlagNone)                              \
    /* floating point arithmetic, double */                                                        \
    X(Fadd, "fadd", WIIU_PPC_A(63, 21), Form::Aab, FlagRc)                                         \
    X(Fsub, "fsub", WIIU_PPC_A(63, 20), Form::Aab, FlagRc)                                         \
    X(Fmul, "fmul", WIIU_PPC_A(63, 25), Form::Aac, FlagRc)                                         \
    X(Fdiv, "fdiv", WIIU_PPC_A(63, 18), Form::Aab, FlagRc)                                         \
    X(Fmadd, "fmadd", WIIU_PPC_A(63, 29), Form::Aabc, FlagRc)                                      \
    X(Fmsub, "fmsub", WIIU_PPC_A(63, 28), Form::Aabc, FlagRc)                                      \
    X(Fnmadd, "fnmadd", WIIU_PPC_A(63, 31), Form::Aabc, FlagRc)                                    \
    X(Fnmsub, "fnmsub", WIIU_PPC_A(63, 30), Form::Aabc, FlagRc)                                    \
    X(Fsel, "fsel", WIIU_PPC_A(63, 23), Form::Aabc, FlagRc)                                        \
    X(Frsqrte, "frsqrte", WIIU_PPC_A(63, 26), Form::Ab, FlagRc)                                    \
    /* floating point arithmetic, single */                                                        \
    X(Fadds, "fadds", WIIU_PPC_A(59, 21), Form::Aab, FlagRc)                                       \
    X(Fsubs, "fsubs", WIIU_PPC_A(59, 20), Form::Aab, FlagRc)                                       \
    X(Fmuls, "fmuls", WIIU_PPC_A(59, 25), Form::Aac, FlagRc)                                       \
    X(Fdivs, "fdivs", WIIU_PPC_A(59, 18), Form::Aab, FlagRc)                                       \
    X(Fmadds, "fmadds", WIIU_PPC_A(59, 29), Form::Aabc, FlagRc)                                    \
    X(Fmsubs, "fmsubs", WIIU_PPC_A(59, 28), Form::Aabc, FlagRc)                                    \
    X(Fnmadds, "fnmadds", WIIU_PPC_A(59, 31), Form::Aabc, FlagRc)                                  \
    X(Fnmsubs, "fnmsubs", WIIU_PPC_A(59, 30), Form::Aabc, FlagRc)                                  \
    X(Fres, "fres", WIIU_PPC_A(59, 24), Form::Ab, FlagRc)                                          \
    /* floating point moves and conversions */                                                     \
    X(Fmr, "fmr", WIIU_PPC_X(63, 72), Form::XFp2, FlagRc)                                          \
    X(Fneg, "fneg", WIIU_PPC_X(63, 40), Form::XFp2, FlagRc)                                        \
    X(Fabs, "fabs", WIIU_PPC_X(63, 264), Form::XFp2, FlagRc)                                       \
    X(Fnabs, "fnabs", WIIU_PPC_X(63, 136), Form::XFp2, FlagRc)                                     \
    X(Frsp, "frsp", WIIU_PPC_X(63, 12), Form::XFp2, FlagRc)                                        \
    X(Fctiw, "fctiw", WIIU_PPC_X(63, 14), Form::XFp2, FlagRc)                                      \
    X(Fctiwz, "fctiwz", WIIU_PPC_X(63, 15), Form::XFp2, FlagRc)                                    \
    X(Fcmpu, "fcmpu", WIIU_PPC_XR(63, 0), Form::XFpCmp, FlagNone)                                  \
    X(Fcmpo, "fcmpo", WIIU_PPC_XR(63, 32), Form::XFpCmp, FlagNone)                                 \
    X(Mffs, "mffs", WIIU_PPC_X(63, 583), Form::XMffs, FlagRc)                                      \
    X(Mtfsf, "mtfsf", WIIU_PPC_X(63, 711), Form::XMtfsf, FlagRc)                                   \
    X(Mcrfs, "mcrfs", WIIU_PPC_XR(63, 64), Form::XMcrfs, FlagNone)                                 \
    /* paired singles -- the Gekko/Espresso extension */                                           \
    X(PsAdd, "ps_add", WIIU_PPC_A(4, 21), Form::Aab, FlagRc)                                       \
    X(PsSub, "ps_sub", WIIU_PPC_A(4, 20), Form::Aab, FlagRc)                                       \
    X(PsMul, "ps_mul", WIIU_PPC_A(4, 25), Form::Aac, FlagRc)                                       \
    X(PsDiv, "ps_div", WIIU_PPC_A(4, 18), Form::Aab, FlagRc)                                       \
    X(PsMadd, "ps_madd", WIIU_PPC_A(4, 29), Form::Aabc, FlagRc)                                    \
    X(PsMsub, "ps_msub", WIIU_PPC_A(4, 28), Form::Aabc, FlagRc)                                    \
    X(PsNmadd, "ps_nmadd", WIIU_PPC_A(4, 31), Form::Aabc, FlagRc)                                  \
    X(PsNmsub, "ps_nmsub", WIIU_PPC_A(4, 30), Form::Aabc, FlagRc)                                  \
    X(PsSel, "ps_sel", WIIU_PPC_A(4, 23), Form::Aabc, FlagRc)                                      \
    X(PsRes, "ps_res", WIIU_PPC_A(4, 24), Form::Ab, FlagRc)                                        \
    X(PsRsqrte, "ps_rsqrte", WIIU_PPC_A(4, 26), Form::Ab, FlagRc)                                  \
    X(PsMuls0, "ps_muls0", WIIU_PPC_A(4, 12), Form::Aac, FlagRc)                                   \
    X(PsMuls1, "ps_muls1", WIIU_PPC_A(4, 13), Form::Aac, FlagRc)                                   \
    X(PsMadds0, "ps_madds0", WIIU_PPC_A(4, 14), Form::Aabc, FlagRc)                                \
    X(PsMadds1, "ps_madds1", WIIU_PPC_A(4, 15), Form::Aabc, FlagRc)                                \
    X(PsMr, "ps_mr", WIIU_PPC_X(4, 72), Form::XFp2, FlagRc)                                        \
    X(PsNeg, "ps_neg", WIIU_PPC_X(4, 40), Form::XFp2, FlagRc)                                      \
    X(PsAbs, "ps_abs", WIIU_PPC_X(4, 264), Form::XFp2, FlagRc)                                     \
    X(PsNabs, "ps_nabs", WIIU_PPC_X(4, 136), Form::XFp2, FlagRc)                                   \
    X(PsMerge00, "ps_merge00", WIIU_PPC_X(4, 528), Form::Aab, FlagRc)                              \
    X(PsMerge01, "ps_merge01", WIIU_PPC_X(4, 560), Form::Aab, FlagRc)                              \
    X(PsMerge10, "ps_merge10", WIIU_PPC_X(4, 592), Form::Aab, FlagRc)                              \
    X(PsMerge11, "ps_merge11", WIIU_PPC_X(4, 624), Form::Aab, FlagRc)                              \
    X(PsqL, "psq_l", WIIU_PPC_D(56), Form::PsQuantized, FlagNone)                                  \
    X(PsqLu, "psq_lu", WIIU_PPC_D(57), Form::PsQuantized, FlagNone)                                \
    X(PsqSt, "psq_st", WIIU_PPC_D(60), Form::PsQuantized, FlagNone)                                \
    X(PsqStu, "psq_stu", WIIU_PPC_D(61), Form::PsQuantized, FlagNone)
// NOLINTEND(*-macro-parentheses)

enum class Opcode : u16 {
    Invalid = 0,
#define WIIU_PPC_ENUM(id, ...) id,
    WIIU_PPC_OPCODES(WIIU_PPC_ENUM)
#undef WIIU_PPC_ENUM
        Count
};

struct OpcodeInfo {
    Opcode id = Opcode::Invalid;
    std::string_view name;
    u32 pattern = 0;
    u32 mask = 0;
    Form form = Form::Invalid;
    u8 flags = FlagNone;

    [[nodiscard]] constexpr bool matches(u32 word) const noexcept {
        return (word & mask) == pattern;
    }
};

/// Every entry, in declaration order.
[[nodiscard]] std::span<const OpcodeInfo> opcode_table() noexcept;

/// Information for a decoded opcode. Always valid: an unknown opcode returns
/// the Invalid entry rather than nullptr, so callers need no null check.
[[nodiscard]] const OpcodeInfo& opcode_info(Opcode id) noexcept;

/// Decode a word to its opcode. Opcode::Invalid if nothing matches.
[[nodiscard]] Opcode decode(u32 word) noexcept;

[[nodiscard]] inline Opcode decode(Instruction instruction) noexcept {
    return decode(instruction.raw);
}

/// Lowercase mnemonic with the suffixes the encoding calls for: 'o' for
/// overflow, '.' for record, 'l' and 'a' for branches.
[[nodiscard]] std::string decorated_name(Opcode id, Instruction instruction);

} // namespace wiiu::cpu
