// Phase 3.4 — address translation.
//
// The interesting failures here are quiet ones: a translation that lands on
// the wrong page still reads *something*, and a protection check that never
// fires only shows up as a game writing over its own code. So these tests
// check where an access actually lands, not just that it succeeded.

#include <wiiu/common/log.hpp>
#include <wiiu/cpu/exceptions.hpp>
#include <wiiu/cpu/interpreter.hpp>
#include <wiiu/cpu/mmu.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::cpu;

namespace {

constexpr ppcaddr_t kLoadAddress = 0x02000000;

/// A page table in the data heap: physical 0x20100000 is the emulator's
/// 0x10100000, and HTABORG only has room for the top sixteen bits, so it has
/// to be 64 KiB aligned.
constexpr physaddr_t kPageTablePhysical = 0x20100000;
constexpr u32 kSdr1 = kPageTablePhysical; // HTABMASK 0: the smallest table

/// A segment we can hand out without colliding with the Cafe OS map.
constexpr u32 kSegment = 3;
constexpr ppcaddr_t kSegmentBase = kSegment << 28;
constexpr u32 kVsid = 0x000123;

// Instruction encodings, assembled with binutils.
constexpr u32 kEnc_mtsr_3_r4 = 0x7C8301A4;       // mtsr 3,r4
constexpr u32 kEnc_mfsr_r5_3 = 0x7CA304A6;       // mfsr r5,3
constexpr u32 kEnc_mtsrin_r4_r5 = 0x7C8029E4;    // mtsrin r4,r5
constexpr u32 kEnc_mfsrin_r6_r5 = 0x7CC02D26;    // mfsrin r6,r5
constexpr u32 kEnc_tlbie_r3 = 0x7C001A64;        // tlbie r3
constexpr u32 kEnc_tlbsync = 0x7C00046C;         // tlbsync
constexpr u32 kEnc_mtspr_sdr1_r3 = 0x7C7903A6;   // mtspr 25,r3
constexpr u32 kEnc_mfspr_r4_sdr1 = 0x7C9902A6;   // mfspr r4,25
constexpr u32 kEnc_mtspr_dbat0u_r3 = 0x7C7883A6; // mtspr 536,r3
constexpr u32 kEnc_mtspr_dbat0l_r4 = 0x7C9983A6; // mtspr 537,r4
constexpr u32 kEnc_mfspr_r5_dbat0u = 0x7CB882A6; // mfspr r5,536
constexpr u32 kEnc_stw_r5_0_r4 = 0x90A40000;     // stw r5,0(r4)
constexpr u32 kEnc_lwz_r5_0_r4 = 0x80A40000;     // lwz r5,0(r4)

/// BAT halves, built the way software does.
[[nodiscard]] constexpr u32 bat_upper(ppcaddr_t block, u32 length, bool supervisor = true) {
    return (block & BatBits::bepi_mask) | (length << 2) |
           (supervisor ? BatBits::valid_supervisor : BatBits::valid_problem);
}

[[nodiscard]] constexpr u32 bat_lower(physaddr_t physical, u32 protection) {
    return (physical & BatBits::brpn_mask) | protection;
}

struct MmuFixture {
    memory::Memory memory;
    CoreState state;
    Interpreter interpreter{state, memory};

    MmuFixture() {
        // Exception vectors are not part of the Cafe OS map, so Deliver mode
        // needs them added -- the same finding the Phase 4.10 tests record.
        std::vector<memory::Region> regions(memory::default_regions().begin(),
                                            memory::default_regions().end());
        regions.insert(regions.begin(),
                       memory::Region{"vectors",
                                      0x00000000,
                                      0x4000,
                                      0,
                                      memory::Backing::None,
                                      memory::read_write_execute});

        std::string error;
        const bool ok = memory.initialize(regions, error);
        INFO("initialize: " << error);
        REQUIRE(ok);

        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        interpreter.set_exception_mode(ExceptionMode::Deliver);
        state.pc = kLoadAddress;
    }

    StepResult exec(u32 word) {
        REQUIRE(memory.try_write<u32>(kLoadAddress, word));
        state.pc = kLoadAddress;
        return interpreter.step();
    }

    /// Point SDR1 at a cleared page table and give the test segment a VSID.
    void enable_page_table() {
        REQUIRE(memory.fill(Mmu::space_address(memory, kPageTablePhysical),
                            static_cast<u32>(Mmu::page_table_size(kSdr1)),
                            0));
        state.sdr1 = kSdr1;
        state.sr[kSegment] = kVsid;
    }

    [[nodiscard]] ppcaddr_t pte_of(ppcaddr_t effective, u32 slot = 0) const {
        const u32 hash = Mmu::page_hash(kVsid, (effective >> 12) & 0xFFFF);
        return Mmu::space_address(memory, Mmu::pteg_address(kSdr1, hash) + slot * 8);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Page table arithmetic
// ---------------------------------------------------------------------------

TEST_CASE("the page table hash mixes VSID and page index", "[cpu][mmu]") {
    // Nineteen bits, and the secondary hash is the one's complement of the
    // primary -- which is what makes the two searches cover different groups.
    STATIC_REQUIRE(Mmu::page_hash(0, 0) == 0);
    STATIC_REQUIRE(Mmu::page_hash(0x5, 0x3) == 0x6);
    STATIC_REQUIRE(Mmu::page_hash(0xFFFFFFFF, 0) == 0x7FFFF);
}

TEST_CASE("SDR1 places the page table and sizes it", "[cpu][mmu]") {
    // HTABORG is the top sixteen bits, so the smallest table is 64 KiB and
    // must be aligned to its own size.
    STATIC_REQUIRE(Mmu::page_table_size(0x20100000) == 0x10000);
    STATIC_REQUIRE(Mmu::page_table_size(0x20100001) == 0x20000);
    STATIC_REQUIRE(Mmu::pteg_address(0x20100000, 0) == 0x20100000);

    // Bits 9-18 of the hash select the group; the low ten bits index within
    // the 64-byte groups.
    STATIC_REQUIRE(Mmu::pteg_address(0x20100000, 1) == 0x20100040);
    STATIC_REQUIRE(Mmu::pteg_address(0x20100000, 0x3FF) == 0x2010FFC0);

    // With HTABMASK set, the hash reaches beyond the first 64 KiB.
    STATIC_REQUIRE(Mmu::pteg_address(0x20100001, 0x400) == 0x20110000);
}

TEST_CASE_METHOD(MmuFixture, "a real address maps back into the guest space", "[cpu][mmu]") {
    // MEM1 is physically at zero but Cafe OS shows it at 0xF4000000, and the
    // emulator's address space is keyed the way the guest sees it.
    CHECK(Mmu::space_address(memory, 0x00001000) == memory::mem1_base + 0x1000);
    CHECK(Mmu::space_address(memory, 0x20000000) == memory::data_heap_base);

    // Nothing backs this: it comes back unchanged and faults later, as it
    // would on hardware.
    CHECK(Mmu::space_address(memory, 0xCAFE0000) == 0xCAFE0000);
}

// ---------------------------------------------------------------------------
// Translation off
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture, "with MSR[DR] clear the address is used as it is", "[cpu][mmu]") {
    REQUIRE_FALSE(interpreter.mmu().enabled(AccessKind::Read));

    const Translation translation = interpreter.mmu().translate(0x10004000, AccessKind::Read);
    CHECK(translation.ok());
    CHECK(translation.address == 0x10004000);
    CHECK_FALSE(translation.translated);

    // And nothing was walked to find that out.
    CHECK(interpreter.mmu().stats().walks == 0);
}

// ---------------------------------------------------------------------------
// Block address translation
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture, "a BAT redirects a whole block", "[cpu][mmu][bat]") {
    // 128 KiB of the test segment onto the start of MEM1.
    state.dbat_u[0] = bat_upper(kSegmentBase, 0);
    state.dbat_l[0] = bat_lower(memory::mem1_physical, 2); // read/write
    state.msr |= MsrBits::dr;

    state.gpr[4] = kSegmentBase + 0x40;
    state.gpr[5] = 0xDEADBEEF;
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);

    // The store has to have landed in MEM1, not merely succeeded.
    CHECK(memory.read<u32>(memory::mem1_base + 0x40) == 0xDEADBEEFU);
    CHECK(interpreter.mmu().stats().bat_hits == 1);

    state.gpr[5] = 0;
    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    CHECK(state.gpr[5] == 0xDEADBEEFU);

    // The second access came from the TLB rather than a fresh match.
    CHECK(interpreter.mmu().stats().hits == 1);
}

TEST_CASE_METHOD(MmuFixture, "a read-only BAT refuses a store", "[cpu][mmu][bat]") {
    state.dbat_u[0] = bat_upper(kSegmentBase, 0);
    state.dbat_l[0] = bat_lower(memory::mem1_physical, 1); // read only
    state.msr |= MsrBits::dr;

    state.gpr[4] = kSegmentBase + 0x40;
    state.gpr[5] = 0xDEADBEEF;

    // Delivered as a data storage exception, with DSISR saying the mapping
    // exists but forbids the access.
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Dsi));
    CHECK(state.dar == kSegmentBase + 0x40);
    CHECK((state.dsisr & DsisrBits::protection) != 0);
    CHECK((state.dsisr & DsisrBits::is_store) != 0);
    CHECK((state.dsisr & DsisrBits::not_translated) == 0);
    CHECK(memory.read<u32>(memory::mem1_base + 0x40) == 0U);

    // A load through the same BAT is fine.
    state.pc = kLoadAddress;
    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    CHECK(state.gpr[5] == 0U);
}

TEST_CASE_METHOD(MmuFixture, "a BAT only answers for its own access kind", "[cpu][mmu][bat]") {
    // Data BATs do not translate instruction fetches, which is the whole
    // reason there are two sets.
    state.dbat_u[0] = bat_upper(kSegmentBase, 0);
    state.dbat_l[0] = bat_lower(memory::mem1_physical, 2);

    const Mmu& mmu = interpreter.mmu();
    state.msr |= MsrBits::dr | MsrBits::ir;

    CHECK(mmu.inspect(kSegmentBase, AccessKind::Read).ok());
    CHECK(mmu.inspect(kSegmentBase, AccessKind::Execute).fault == TranslationFault::NotTranslated);
}

TEST_CASE_METHOD(MmuFixture, "the upper BAT pairs wait for HID4", "[cpu][mmu][bat]") {
    // The 750CL's second set of four is not live until software enables it.
    state.dbat_u[5] = bat_upper(kSegmentBase, 0);
    state.dbat_l[5] = bat_lower(memory::mem1_physical, 2);
    state.msr |= MsrBits::dr;

    CHECK(interpreter.mmu().inspect(kSegmentBase, AccessKind::Read).fault ==
          TranslationFault::NotTranslated);

    state.hid4 = hid4_extra_bats;
    CHECK(interpreter.mmu().inspect(kSegmentBase, AccessKind::Read).ok());
}

TEST_CASE_METHOD(MmuFixture, "a block covers exactly its length", "[cpu][mmu][bat]") {
    // BL is a mask, not a count: 0x7FF is the largest block at 256 MiB, and 1
    // means 256 KiB.
    state.dbat_u[0] = bat_upper(kSegmentBase, 1);
    state.dbat_l[0] = bat_lower(memory::mem1_physical, 2);
    state.msr |= MsrBits::dr;

    const Mmu& mmu = interpreter.mmu();
    CHECK(mmu.inspect(kSegmentBase + 0x3FFFF, AccessKind::Read).physical == 0x3FFFF);
    CHECK(mmu.inspect(kSegmentBase + 0x40000, AccessKind::Read).fault ==
          TranslationFault::NotTranslated);
}

// ---------------------------------------------------------------------------
// Page table
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture, "a page table entry translates one page", "[cpu][mmu][page]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1010;
    state.gpr[5] = 0x1234ABCD;
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);

    CHECK(memory.read<u32>(memory::mem1_base + 0x1010) == 0x1234ABCDU);
    CHECK(interpreter.mmu().stats().walks == 1);

    // The neighbouring page has no entry and must not be translated by the
    // one that does.
    state.gpr[4] = kSegmentBase + 0x2010;
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Dsi));
    CHECK((state.dsisr & DsisrBits::not_translated) != 0);
}

TEST_CASE_METHOD(MmuFixture, "a walk records reference and change", "[cpu][mmu][page]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    const ppcaddr_t pte = pte_of(kSegmentBase + 0x1000);
    REQUIRE((memory.read<u32>(pte + 4) & (PteBits::referenced | PteBits::changed)) == 0);

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1010;

    // A load references the page but does not change it.
    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    CHECK((memory.read<u32>(pte + 4) & PteBits::referenced) != 0);
    CHECK((memory.read<u32>(pte + 4) & PteBits::changed) == 0);

    // A store does both -- and has to reach the PTE even though the
    // translation itself is already cached.
    state.gpr[5] = 1;
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(interpreter.mmu().stats().hits == 1);
    CHECK((memory.read<u32>(pte + 4) & PteBits::changed) != 0);
}

TEST_CASE_METHOD(MmuFixture, "the secondary hash is searched too", "[cpu][mmu][page]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(memory,
                             kSdr1,
                             kVsid,
                             kSegmentBase + 0x5000,
                             memory::mem1_physical + 0x5000,
                             2,
                             0,
                             /*secondary=*/true));

    state.msr |= MsrBits::dr;
    const Translation translation =
        interpreter.mmu().inspect(kSegmentBase + 0x5004, AccessKind::Read);
    CHECK(translation.ok());
    CHECK(translation.physical == memory::mem1_physical + 0x5004);
}

TEST_CASE_METHOD(MmuFixture, "the segment key selects the protection", "[cpu][mmu][page]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 1));

    state.msr |= MsrBits::dr;
    const Mmu& mmu = interpreter.mmu();

    // PP=01 with no key is read/write; with the key it is read only. Same
    // page table entry, different answer -- which is the point of the key.
    CHECK(mmu.inspect(kSegmentBase + 0x1000, AccessKind::Write).ok());

    state.sr[kSegment] = kVsid | SegmentBits::key_supervisor;
    CHECK(mmu.inspect(kSegmentBase + 0x1000, AccessKind::Read).ok());
    CHECK(mmu.inspect(kSegmentBase + 0x1000, AccessKind::Write).fault ==
          TranslationFault::Protection);
}

TEST_CASE_METHOD(MmuFixture, "a no-execute segment refuses a fetch", "[cpu][mmu][page]") {
    enable_page_table();
    state.sr[kSegment] = kVsid | SegmentBits::no_execute;
    state.msr |= MsrBits::ir;
    state.pc = kSegmentBase + 0x1000;

    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Isi));
    CHECK((state.srr1 & IsiCause::no_execute) != 0);
    CHECK((state.srr1 & IsiCause::not_translated) == 0);
}

TEST_CASE_METHOD(MmuFixture, "a fetch with no mapping is an ISI, not a DSI", "[cpu][mmu][page]") {
    enable_page_table();
    state.msr |= MsrBits::ir;
    state.pc = kSegmentBase + 0x1000;

    REQUIRE(interpreter.step() == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Isi));
    CHECK((state.srr1 & IsiCause::not_translated) != 0);
    CHECK(state.srr0 == kSegmentBase + 0x1000);

    // DSISR belongs to data accesses and must not have been touched.
    CHECK(state.dsisr == 0U);
}

// ---------------------------------------------------------------------------
// The TLB
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture,
                 "a cached translation outlives the entry it came from",
                 "[cpu][mmu][tlb]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1000;
    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);

    // Invalidate the entry behind the emulator's back. Hardware would keep
    // using its TLB copy until told otherwise, and so does this.
    const ppcaddr_t pte = pte_of(kSegmentBase + 0x1000);
    REQUIRE(memory.try_write<u32>(pte, 0));

    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    CHECK(interpreter.mmu().stats().hits == 1);

    // tlbie is how software says so.
    state.gpr[3] = kSegmentBase + 0x1000;
    REQUIRE(exec(kEnc_tlbie_r3) == StepResult::Ok);

    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Dsi));
    CHECK((state.dsisr & DsisrBits::not_translated) != 0);
}

TEST_CASE_METHOD(MmuFixture, "tlbie only retires the page it names", "[cpu][mmu][tlb]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x9000, memory::mem1_physical + 0x9000, 2));

    state.msr |= MsrBits::dr;
    Mmu& mmu = interpreter.mmu();
    REQUIRE(mmu.translate(kSegmentBase + 0x1000, AccessKind::Read).ok());
    REQUIRE(mmu.translate(kSegmentBase + 0x9000, AccessKind::Read).ok());
    mmu.reset_stats();

    mmu.invalidate(kSegmentBase + 0x1000);

    CHECK(mmu.translate(kSegmentBase + 0x9000, AccessKind::Read).ok());
    CHECK(mmu.stats().hits == 1); // untouched
    CHECK(mmu.translate(kSegmentBase + 0x1000, AccessKind::Read).ok());
    CHECK(mmu.stats().walks == 1); // walked again
}

TEST_CASE_METHOD(MmuFixture, "changing a segment register retires the TLB", "[cpu][mmu][tlb]") {
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    state.msr |= MsrBits::dr;
    REQUIRE(interpreter.mmu().translate(kSegmentBase + 0x1000, AccessKind::Read).ok());

    // A different VSID means different translations; software is not required
    // to invalidate anything for that to take effect.
    state.gpr[4] = kVsid + 1;
    REQUIRE(exec(kEnc_mtsr_3_r4) == StepResult::Ok);

    CHECK(interpreter.mmu().translate(kSegmentBase + 0x1000, AccessKind::Read).fault ==
          TranslationFault::NotTranslated);
}

TEST_CASE_METHOD(MmuFixture, "entering an exception retires the TLB", "[cpu][mmu][tlb]") {
    // MSR[PR] decides which segment key applies, so a cached translation made
    // in user mode cannot be reused in supervisor mode.
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    state.msr |= MsrBits::dr | MsrBits::pr;
    REQUIRE(interpreter.mmu().translate(kSegmentBase + 0x1000, AccessKind::Read).ok());
    interpreter.mmu().reset_stats();

    interpreter.raise(Exception::Decrementer);
    REQUIRE((state.msr & MsrBits::pr) == 0);

    state.msr |= MsrBits::dr;
    CHECK(interpreter.mmu().translate(kSegmentBase + 0x1000, AccessKind::Read).ok());
    CHECK(interpreter.mmu().stats().hits == 0);
}

// ---------------------------------------------------------------------------
// The registers themselves
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture, "segment registers move both ways", "[cpu][mmu][spr]") {
    state.gpr[4] = 0xABCDEF12;
    REQUIRE(exec(kEnc_mtsr_3_r4) == StepResult::Ok);
    CHECK(state.sr[3] == 0xABCDEF12U);

    REQUIRE(exec(kEnc_mfsr_r5_3) == StepResult::Ok);
    CHECK(state.gpr[5] == 0xABCDEF12U);

    // The indirect forms take the segment from the top four bits of rB, which
    // is how software walks the segments in a loop.
    state.gpr[5] = 0x70000000; // segment 7
    state.gpr[4] = 0x0000BEEF;
    REQUIRE(exec(kEnc_mtsrin_r4_r5) == StepResult::Ok);
    CHECK(state.sr[7] == 0x0000BEEFU);

    REQUIRE(exec(kEnc_mfsrin_r6_r5) == StepResult::Ok);
    CHECK(state.gpr[6] == 0x0000BEEFU);
}

TEST_CASE_METHOD(MmuFixture, "SDR1 and the BATs are ordinary SPRs", "[cpu][mmu][spr]") {
    state.gpr[3] = kSdr1;
    REQUIRE(exec(kEnc_mtspr_sdr1_r3) == StepResult::Ok);
    CHECK(state.sdr1 == kSdr1);

    REQUIRE(exec(kEnc_mfspr_r4_sdr1) == StepResult::Ok);
    CHECK(state.gpr[4] == kSdr1);

    // 536 and 537 are DBAT0's two halves; a wrong split here would silently
    // map the wrong memory.
    state.gpr[3] = bat_upper(kSegmentBase, 0);
    state.gpr[4] = bat_lower(memory::mem1_physical, 2);
    REQUIRE(exec(kEnc_mtspr_dbat0u_r3) == StepResult::Ok);
    REQUIRE(exec(kEnc_mtspr_dbat0l_r4) == StepResult::Ok);
    CHECK(state.dbat_u[0] == bat_upper(kSegmentBase, 0));
    CHECK(state.dbat_l[0] == bat_lower(memory::mem1_physical, 2));

    REQUIRE(exec(kEnc_mfspr_r5_dbat0u) == StepResult::Ok);
    CHECK(state.gpr[5] == bat_upper(kSegmentBase, 0));
}

TEST_CASE_METHOD(MmuFixture,
                 "tlbsync is a real instruction, not an illegal one",
                 "[cpu][mmu][spr]") {
    // Nothing here runs the TLBs concurrently, so it has nothing to wait for,
    // but guest code emits it after tlbie and must not trap.
    CHECK(exec(kEnc_tlbsync) == StepResult::Ok);
}

// ---------------------------------------------------------------------------
// Translation and the rest of the machine
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture, "cache maintenance translates like any other access", "[cpu][mmu]") {
    // dcbf names a cache block by effective address. Flushing a translated
    // address must reach the block the guest meant, not the same number in
    // some other space.
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    std::vector<std::pair<memory::CacheOp, ppcaddr_t>> seen;
    memory.set_cache_listener(
        [&seen](memory::CacheOp op, ppcaddr_t address, u32) { seen.emplace_back(op, address); });

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1000;

    constexpr u32 kEnc_dcbf_0_r4 = 0x7C0020AC; // dcbf 0,r4
    REQUIRE(exec(kEnc_dcbf_0_r4) == StepResult::Ok);

    REQUIRE(seen.size() == 1);
    CHECK(seen[0].first == memory::CacheOp::Flush);
    CHECK(seen[0].second == memory::mem1_base + 0x1000);
}

// ---------------------------------------------------------------------------
// Findings from the review pass
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(MmuFixture, "an access spanning two pages translates both", "[cpu][mmu][page]") {
    // Consecutive guest pages are almost never consecutive once translated, so
    // a load that straddles the boundary has to be split. Mapping the second
    // page somewhere far from the first is what makes a single translation
    // visible: it would read the bytes after the first page instead.
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x2000, memory::mem1_physical + 0x9000, 2));

    REQUIRE(memory.try_write<u16>(memory::mem1_base + 0x1FFE, 0xAABB));
    REQUIRE(memory.try_write<u16>(memory::mem1_base + 0x9000, 0xCCDD));
    REQUIRE(memory.try_write<u16>(memory::mem1_base + 0x2000, 0x1122)); // the decoy

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1FFE;

    constexpr u32 kEnc_lwz_r5_0_r4_split = 0x80A40000; // lwz r5,0(r4)
    REQUIRE(exec(kEnc_lwz_r5_0_r4_split) == StepResult::Ok);
    CHECK(state.gpr[5] == 0xAABBCCDDU);

    // And the same for a store.
    state.gpr[5] = 0x01020304;
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(memory.read<u16>(memory::mem1_base + 0x1FFE) == 0x0102U);
    CHECK(memory.read<u16>(memory::mem1_base + 0x9000) == 0x0304U);
    CHECK(memory.read<u16>(memory::mem1_base + 0x2000) == 0x1122U); // untouched
}

TEST_CASE_METHOD(MmuFixture,
                 "a split store checks the far page before writing",
                 "[cpu][mmu][page]") {
    // Half a store is worse than none: the guest's handler would see memory
    // the instruction is about to be blamed for not writing.
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    REQUIRE(memory.try_write<u16>(memory::mem1_base + 0x1FFE, 0xAABB));

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1FFE;
    state.gpr[5] = 0x01020304;

    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Dsi));
    CHECK((state.dsisr & DsisrBits::not_translated) != 0);
    CHECK(memory.read<u16>(memory::mem1_base + 0x1FFE) == 0xAABBU);

    // DAR names the address the instruction asked for, the same as it would
    // for an access that never crossed a page at all.
    CHECK(state.dar == kSegmentBase + 0x1FFE);
}

TEST_CASE_METHOD(MmuFixture,
                 "a page table entry is encoded the way hardware reads it",
                 "[cpu][mmu][page]") {
    // Everything else about the page table is checked by writing an entry and
    // reading it back, which cannot catch two fields being consistently
    // swapped. This pins the actual words.
    enable_page_table();
    REQUIRE(Mmu::install_pte(
        memory, kSdr1, kVsid, kSegmentBase + 0x1000, memory::mem1_physical + 0x1000, 2));

    const ppcaddr_t pte = pte_of(kSegmentBase + 0x1000);

    // word0: V=1 (bit 0), VSID in bits 1-24, H=0, API = EA bits 4-9 = 0.
    CHECK(memory.read<u32>(pte) == (0x80000000U | (kVsid << 7)));

    // word1: RPN in bits 0-19, R and C clear, WIMG clear, PP = 10.
    CHECK(memory.read<u32>(pte + 4) == (0x00001000U | 0x2U));

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1000;
    state.gpr[5] = 1;

    // A load sets R (bit 23 -> 0x100) alone. Checking after a store would not
    // distinguish R from C, since a store sets both and the two constants are
    // adjacent -- swap them and every other test here still passes.
    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    CHECK(memory.read<u32>(pte + 4) == (0x00001000U | 0x100U | 0x2U));

    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(memory.read<u32>(pte + 4) == (0x00001000U | 0x180U | 0x2U));
}

TEST_CASE_METHOD(MmuFixture, "a cached read does not license a write", "[cpu][mmu][tlb]") {
    // The TLB stores what the mapping permits rather than what the access that
    // filled it asked for. Reading first and storing second is the case where
    // the difference shows.
    enable_page_table();
    REQUIRE(Mmu::install_pte(memory,
                             kSdr1,
                             kVsid,
                             kSegmentBase + 0x1000,
                             memory::mem1_physical + 0x1000,
                             /*protection=*/3)); // read only
    state.sr[kSegment] = kVsid | SegmentBits::key_supervisor;

    state.msr |= MsrBits::dr;
    state.gpr[4] = kSegmentBase + 0x1000;

    REQUIRE(exec(kEnc_lwz_r5_0_r4) == StepResult::Ok);
    REQUIRE(state.pc == kLoadAddress + 4);

    state.gpr[5] = 0xDEADBEEF;
    REQUIRE(exec(kEnc_stw_r5_0_r4) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Dsi));
    CHECK((state.dsisr & DsisrBits::protection) != 0);
    CHECK(memory.read<u32>(memory::mem1_base + 0x1000) == 0U);
}

TEST_CASE_METHOD(MmuFixture, "user mode cannot reprogram translation", "[cpu][mmu][spr]") {
    // Without this the MMU is decorative: a guest in problem state could give
    // itself any mapping it liked.
    state.msr |= MsrBits::pr;
    state.gpr[4] = 0xDEADBEEF;
    state.gpr[3] = kSdr1;

    // Taking the exception drops back to supervisor mode, so each attempt has
    // to re-enter problem state.
    const auto in_user_mode = [this](u32 word) {
        state.msr |= MsrBits::pr;
        return exec(word);
    };

    REQUIRE(in_user_mode(kEnc_mtsr_3_r4) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Program));
    CHECK((state.srr1 & ProgramCause::privileged) != 0);
    CHECK(state.sr[3] == 0U);

    REQUIRE(in_user_mode(kEnc_mtspr_sdr1_r3) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Program));
    CHECK(state.sdr1 == 0U);

    REQUIRE(in_user_mode(kEnc_tlbie_r3) == StepResult::Ok);
    CHECK(state.pc == static_cast<u32>(Exception::Program));

    // LR is not a privileged SPR: the check is per register, not per
    // instruction.
    constexpr u32 kEnc_mtspr_lr_r3 = 0x7C6803A6; // mtspr 8,r3
    REQUIRE(in_user_mode(kEnc_mtspr_lr_r3) == StepResult::Ok);
    CHECK(state.pc == kLoadAddress + 4);
    CHECK(state.lr == kSdr1);
}
