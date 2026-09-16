#include <wiiu/cpu/opcodes.hpp>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>

using namespace wiiu;
using namespace wiiu::cpu;

TEST_CASE("every table entry decodes to itself", "[cpu][decode]") {
    // The failure this catches: an entry whose pattern is shadowed by an
    // earlier, broader one, so it can never be reached.
    for (const OpcodeInfo& info : opcode_table()) {
        INFO("opcode " << info.name);
        CHECK(decode(info.pattern) == info.id);
    }
}

TEST_CASE("no two entries share an encoding", "[cpu][decode]") {
    std::set<std::pair<u32, u32>> seen;
    for (const OpcodeInfo& info : opcode_table()) {
        INFO("opcode " << info.name);
        CHECK(seen.emplace(info.pattern, info.mask).second);
    }
}

TEST_CASE("a pattern's fixed bits are covered by its mask", "[cpu][decode]") {
    for (const OpcodeInfo& info : opcode_table()) {
        INFO("opcode " << info.name);
        CHECK((info.pattern & ~info.mask) == 0U);
        CHECK(info.mask != 0U);
        CHECK_FALSE(info.name.empty());
        CHECK(info.form != Form::Invalid);
    }
}

TEST_CASE("real instructions decode to the right opcode", "[cpu][decode]") {
    CHECK(decode(0x7C642A14) == Opcode::Add);    // add 3,4,5
    CHECK(decode(0x80640008) == Opcode::Lwz);    // lwz 3,8(4)
    CHECK(decode(0x9421FFE0) == Opcode::Stwu);   // stwu 1,-32(1)
    CHECK(decode(0x4E800020) == Opcode::Bclr);   // blr
    CHECK(decode(0x44000002) == Opcode::Sc);     // sc
    CHECK(decode(0x60000000) == Opcode::Ori);    // nop
    CHECK(decode(0x54644228) == Opcode::Rlwinm); // rlwinm 4,3,8,8,20
    CHECK(decode(0xEC22182A) == Opcode::Fadds);  // fadds 1,2,3
    CHECK(decode(0x1022182A) == Opcode::PsAdd);  // ps_add 1,2,3
    CHECK(decode(0xE0242008) == Opcode::PsqL);   // psq_l 1,8(4),0,2
    CHECK(decode(0x7C64292D) == Opcode::Stwcx);  // stwcx. 3,4,5
    CHECK(decode(0x7C0004AC) == Opcode::Sync);   // sync
}

TEST_CASE("an unallocated encoding decodes as invalid", "[cpu][decode]") {
    CHECK(decode(0x00000000) == Opcode::Invalid); // primary opcode 0
    CHECK(decode(0x14000000) == Opcode::Invalid); // primary opcode 5
    // Primary opcode 31 with an extended opcode nobody uses.
    CHECK(decode(0x7C0007FE) == Opcode::Invalid);

    // Not 0xFFFFFFFF: that is a real instruction, `fnmadd. f31,f31,f31,f31`,
    // confirmed against binutils. An all-ones word is a tempting "obviously
    // invalid" value and is nothing of the kind.
    CHECK(decode(0xFFFFFFFF) == Opcode::Fnmadd);
}

TEST_CASE("an exact encoding beats the general form it sits inside", "[cpu][decode]") {
    // sc is a single word; nothing else in primary opcode 17 is allocated.
    CHECK(decode(0x44000002) == Opcode::Sc);
    CHECK(decode(0x44000000) == Opcode::Invalid);
}

TEST_CASE("SPR field halves are swapped", "[cpu][decode]") {
    // mtlr r0 == mtspr 8,0. The field holds 0x100, not 0x008; reading it
    // straight would name the wrong register on every mtspr in the program.
    const Instruction mtlr{.raw = 0x7C0803A6};
    CHECK(decode(mtlr) == Opcode::Mtspr);
    CHECK(mtlr.spr_field == 0x100U);
    CHECK(mtlr.spr_number() == 8U);

    const Instruction mtctr{.raw = 0x7C6903A6};
    CHECK(mtctr.spr_number() == 9U);

    const Instruction mflr{.raw = 0x7C0802A6};
    CHECK(decode(mflr) == Opcode::Mfspr);
    CHECK(mflr.spr_number() == 8U);
}

TEST_CASE("mnemonic suffixes follow the encoding", "[cpu][decode]") {
    // subf. 3,4,5 — record bit set.
    const Instruction record{.raw = 0x7C642851};
    CHECK(decorated_name(decode(record), record) == "subf.");

    // addo. 3,4,5 — overflow and record.
    const Instruction both{.raw = 0x7C642E15};
    CHECK(decorated_name(decode(both), both) == "addo.");

    // add 3,4,5 — neither.
    const Instruction plain{.raw = 0x7C642A14};
    CHECK(decorated_name(decode(plain), plain) == "add");

    // bl 0x100 — link bit set.
    const Instruction link{.raw = 0x48000101};
    CHECK(decorated_name(decode(link), link) == "bl");
}

TEST_CASE("instruction fields come out where the documentation says", "[cpu][decode]") {
    // lwz 3,8(4) — documented as rD=3 (bits 6-10), rA=4 (11-15), D=8 (16-31).
    const Instruction lwz{.raw = 0x80640008};
    CHECK(lwz.opcd == 32U);
    CHECK(lwz.rd == 3U);
    CHECK(lwz.ra == 4U);
    CHECK(lwz.d == 8);

    // stwu 1,-32(1) — a negative displacement, sign-extended.
    const Instruction stwu{.raw = 0x9421FFE0};
    CHECK(stwu.rs == 1U);
    CHECK(stwu.ra == 1U);
    CHECK(stwu.d == -32);

    // rlwinm 4,3,8,8,20
    const Instruction rlwinm{.raw = 0x54644228};
    CHECK(rlwinm.ra == 4U);
    CHECK(rlwinm.rs == 3U);
    CHECK(rlwinm.sh == 8U);
    CHECK(rlwinm.mb == 8U);
    CHECK(rlwinm.me == 20U);

    // psq_st 1,-16(4),1,3 — the 12-bit displacement, W and I fields.
    const Instruction psq{.raw = 0xF024BFF0};
    CHECK(decode(psq) == Opcode::PsqSt);
    CHECK(psq.frd == 1U);
    CHECK(psq.ra == 4U);
    CHECK(psq.ps_d == -16);
    CHECK(psq.ps_w == true);
    CHECK(psq.ps_i == 3U);

    // mtcrf 128,3 — CRM at bits 12-19.
    const Instruction mtcrf{.raw = 0x7C680120};
    CHECK(decode(mtcrf) == Opcode::Mtcrf);
    CHECK(mtcrf.crm == 0x80U);
    CHECK(mtcrf.rs == 3U);
}

TEST_CASE("branch displacements are scaled and sign-extended", "[cpu][decode]") {
    // bl 0x100 assembled at address 0: displacement 0x100.
    const Instruction bl{.raw = 0x48000101};
    CHECK(bl.li.value() * 4 == 0x100);
    CHECK(bl.lk == true);
    CHECK(bl.aa == false);

    // beq 0x20 == bc 12,2,0x20
    const Instruction beq{.raw = 0x41820020};
    CHECK(decode(beq) == Opcode::Bc);
    CHECK(beq.bo == 12U);
    CHECK(beq.bi == 2U);
    CHECK(beq.bd.value() * 4 == 0x20);

    // A backward branch: b -8 encodes as 0x4BFFFFF8.
    const Instruction back{.raw = 0x4BFFFFF8};
    CHECK(decode(back) == Opcode::B);
    CHECK(back.li.value() * 4 == -8);
}
