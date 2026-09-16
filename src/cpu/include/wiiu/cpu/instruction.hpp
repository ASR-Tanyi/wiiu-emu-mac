// Phase 4.2 — instruction encoding.
//
// PowerPC documentation numbers bits from the most significant end: the
// primary opcode is "bits 0-5", the Rc bit is "bit 31". `BitField` numbers
// from the least significant end, so a documented field [a..b] becomes
// position 31-b, length b-a+1.
//
// That conversion happens exactly once, here, with the documented range in a
// comment beside every field. Everywhere else in the CPU reads a named field
// and never thinks about bit order again.
#pragma once

#include <wiiu/common/bitfield.hpp>
#include <wiiu/common/types.hpp>

namespace wiiu::cpu {

/// A 32-bit PowerPC instruction word, with every field the Espresso needs.
///
/// All the forms overlay the same word; only the fields belonging to the
/// decoded form are meaningful. That is how the architecture works, and
/// splitting it into one struct per form would just mean casting between them.
union Instruction {
    u32 raw;

    // --- common ---------------------------------------------------------
    BitField<26, 6, u32> opcd; ///< bits 0-5   primary opcode
    BitField<0, 1, bool> rc;   ///< bit 31     record: update CR0
    BitField<0, 1, bool> lk;   ///< bit 31     link: write LR
    BitField<1, 1, bool> aa;   ///< bit 30     absolute address
    BitField<10, 1, bool> oe;  ///< bit 21     overflow enable

    // --- registers ------------------------------------------------------
    BitField<21, 5, u32> rd; ///< bits 6-10  destination
    BitField<21, 5, u32> rs; ///< bits 6-10  source (same field as rD)
    BitField<16, 5, u32> ra; ///< bits 11-15
    BitField<11, 5, u32> rb; ///< bits 16-20

    BitField<21, 5, u32> frd; ///< bits 6-10  floating-point destination
    BitField<21, 5, u32> frs; ///< bits 6-10
    BitField<16, 5, u32> fra; ///< bits 11-15
    BitField<11, 5, u32> frb; ///< bits 16-20
    BitField<6, 5, u32> frc;  ///< bits 21-25

    // --- immediates -----------------------------------------------------
    BitField<0, 16, s32> simm; ///< bits 16-31 signed immediate
    BitField<0, 16, u32> uimm; ///< bits 16-31 unsigned immediate
    BitField<0, 16, s32> d;    ///< bits 16-31 load/store displacement

    // --- extended opcodes ------------------------------------------------
    BitField<1, 10, u32> xo_x; ///< bits 21-30 X, XL, XFX, XS forms
    BitField<1, 9, u32> xo_xo; ///< bits 22-30 XO form (OE takes bit 21)
    BitField<1, 5, u32> xo_a;  ///< bits 26-30 A form

    // --- branches --------------------------------------------------------
    /// bits 6-29, the displacement with its two implied zero bits removed.
    /// The byte displacement is `li() * 4`.
    BitField<2, 24, s32> li;
    /// bits 16-29, likewise: the byte displacement is `bd() * 4`.
    BitField<2, 14, s32> bd;
    BitField<21, 5, u32> bo; ///< bits 6-10  branch options
    BitField<16, 5, u32> bi; ///< bits 11-15 condition register bit

    // --- rotate ----------------------------------------------------------
    BitField<11, 5, u32> sh; ///< bits 16-20 shift amount
    BitField<6, 5, u32> mb;  ///< bits 21-25 mask begin
    BitField<1, 5, u32> me;  ///< bits 26-30 mask end

    // --- condition register ----------------------------------------------
    BitField<23, 3, u32> crfd; ///< bits 6-8   destination CR field
    BitField<18, 3, u32> crfs; ///< bits 11-13 source CR field
    BitField<21, 5, u32> crbd; ///< bits 6-10  destination CR bit
    BitField<16, 5, u32> crba; ///< bits 11-15
    BitField<11, 5, u32> crbb; ///< bits 16-20
    BitField<21, 1, bool> l;   ///< bit 10     compare width (0 = 32-bit)
    BitField<12, 8, u32> crm;  ///< bits 12-19 CR field mask for mtcrf

    // --- special registers ------------------------------------------------
    /// bits 11-20. The two halves are stored swapped; use `spr_number()`.
    BitField<11, 10, u32> spr_field;

    // --- paired singles (psq_l / psq_st) -----------------------------------
    BitField<15, 1, bool> ps_w; ///< bit 16     0 = load both slots, 1 = one
    BitField<12, 3, u32> ps_i;  ///< bits 17-19 which GQR to quantise through
    BitField<0, 12, s32> ps_d;  ///< bits 20-31 12-bit displacement

    // --- misc -------------------------------------------------------------
    BitField<11, 5, u32> nb;   ///< bits 16-20 byte count for lswi/stswi
    BitField<21, 5, u32> to;   ///< bits 6-10  trap conditions
    BitField<17, 8, u32> fm;   ///< bits 7-14  FPSCR field mask
    BitField<11, 5, u32> imm4; ///< bits 16-19 immediate for mtfsfi

    /// The SPR number, with the two 5-bit halves put back the right way round.
    ///
    /// The encoding stores spr[5:9] first: the field for LR (SPR 8) is 0x100,
    /// not 0x008. Forgetting this decodes every mtspr/mfspr as the wrong
    /// register, which is the kind of bug that looks like a corrupt stack.
    [[nodiscard]] constexpr u32 spr_number() const noexcept {
        const u32 field = spr_field.value();
        return ((field & 0x1F) << 5) | (field >> 5);
    }
};

static_assert(sizeof(Instruction) == 4);
static_assert(std::is_trivially_copyable_v<Instruction>);

/// Well-known SPR numbers.
enum class Spr : u32 {
    Xer = 1,
    Lr = 8,
    Ctr = 9,
    Dsisr = 18,
    Dar = 19,
    Dec = 22,
    Sdr1 = 25,
    Srr0 = 26,
    Srr1 = 27,
    Sprg0 = 272,
    Sprg1 = 273,
    Sprg2 = 274,
    Sprg3 = 275,
    Ear = 282,
    Tbl = 284,
    Tbu = 285,
    Pvr = 287,
    Hid0 = 1008,
    Hid1 = 1009,
    Hid2 = 920, ///< Gekko/Espresso: locked cache and paired-singles control
    Gqr0 = 912, ///< paired-singles quantisation registers
    Gqr1 = 913,
    Gqr2 = 914,
    Gqr3 = 915,
    Gqr4 = 916,
    Gqr5 = 917,
    Gqr6 = 918,
    Gqr7 = 919,
    DmaU = 922, ///< locked-cache DMA
    DmaL = 923,
    Wpar = 921,
    Scr = 947, ///< Espresso: inter-core interrupts
};

} // namespace wiiu::cpu
