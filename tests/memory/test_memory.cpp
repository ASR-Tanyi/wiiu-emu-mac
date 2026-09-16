#include <wiiu/memory/memory.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::memory;

namespace {

/// Brings up a real 4 GiB address space. Cheap: the reservation is lazy and
/// only touched pages are ever committed for real.
struct MemoryFixture {
    Memory memory;

    MemoryFixture() {
        std::string error;
        const bool ok = memory.initialize(error);
        INFO("initialize: " << error);
        REQUIRE(ok);
    }
};

/// A device that records what it was asked to do.
class RecordingDevice final : public MmioDevice {
public:
    struct Access {
        u32 offset;
        u32 size;
        u32 value;
        bool is_write;
    };

    u32 read(u32 offset, u32 size) override {
        accesses.push_back({offset, size, next_value, false});
        return next_value;
    }

    void write(u32 offset, u32 value, u32 size) override {
        accesses.push_back({offset, size, value, true});
    }

    [[nodiscard]] std::string_view name() const override { return "recorder"; }

    std::vector<Access> accesses;
    u32 next_value = 0;
};

} // namespace

TEST_CASE_METHOD(MemoryFixture,
                 "the mapped regions are readable and writable",
                 "[memory][access]") {
    CHECK(memory.ready());

    for (const Region& region : memory.regions()) {
        INFO("region " << region.name);
        CHECK(memory.is_mapped(region.base));
        CHECK(memory.is_mapped(region.end() - 1));
        CHECK(has(memory.access_of(region.base), Access::Read));
    }
}

TEST_CASE_METHOD(MemoryFixture,
                 "values round-trip through every writable region",
                 "[memory][access]") {
    for (const Region& region : memory.regions()) {
        if (!has(region.access, Access::Write)) {
            continue;
        }
        INFO("region " << region.name);

        const ppcaddr_t address = region.base + 0x40;
        REQUIRE(memory.try_write<u32>(address, 0xDEADBEEF));
        CHECK(memory.read<u32>(address) == 0xDEADBEEFU);

        REQUIRE(memory.try_write<u16>(address + 8, 0xBEEF));
        CHECK(memory.read<u16>(address + 8) == 0xBEEFU);

        REQUIRE(memory.try_write<u8>(address + 16, 0x42));
        CHECK(memory.read<u8>(address + 16) == 0x42U);

        REQUIRE(memory.try_write<u64>(address + 24, 0x0123456789ABCDEF));
        CHECK(memory.read<u64>(address + 24) == 0x0123456789ABCDEFULL);
    }
}

TEST_CASE_METHOD(MemoryFixture, "guest memory is big-endian", "[memory][access]") {
    const ppcaddr_t address = data_heap_base;
    REQUIRE(memory.try_write<u32>(address, 0x12345678));

    // The bytes in memory are big-endian, whatever the host is.
    std::array<u8, 4> bytes{};
    REQUIRE(memory.read_bytes(address, bytes));
    CHECK(bytes[0] == 0x12);
    CHECK(bytes[1] == 0x34);
    CHECK(bytes[2] == 0x56);
    CHECK(bytes[3] == 0x78);

    // And a byte read sees the most significant byte first.
    CHECK(memory.read<u8>(address) == 0x12U);
    CHECK(memory.read<u8>(address + 3) == 0x78U);
}

TEST_CASE_METHOD(MemoryFixture, "signed and floating-point values survive", "[memory][access]") {
    const ppcaddr_t address = data_heap_base + 0x100;

    REQUIRE(memory.try_write<s32>(address, -2));
    CHECK(memory.read<s32>(address) == -2);

    REQUIRE(memory.try_write<f32>(address + 8, 1.0F));
    CHECK(memory.read<f32>(address + 8) == 1.0F);
    CHECK(memory.read<u32>(address + 8) == 0x3F800000U); // the guest-visible pattern

    REQUIRE(memory.try_write<f64>(address + 16, 0.5));
    CHECK(memory.read<f64>(address + 16) == 0.5);
}

TEST_CASE_METHOD(MemoryFixture, "translate_unchecked is a plain offset", "[memory][fastmem]") {
    // The property fastmem (P3) depends on: guest address to host pointer is
    // base + address, with nothing in between.
    const u8* const zero = memory.translate_unchecked(0);
    CHECK(memory.translate_unchecked(mem1_base) == zero + mem1_base);
    CHECK(memory.translate_unchecked(0xFFFFFFFF) == zero + 0xFFFFFFFFULL);
}

TEST_CASE_METHOD(MemoryFixture, "unmapped access faults instead of corrupting", "[memory][fault]") {
    const ppcaddr_t hole = 0x70000000; // the gap after the data heap

    CHECK_FALSE(memory.is_mapped(hole));
    CHECK(memory.translate(hole, 4, Access::Read) == nullptr);

    u32 value = 0xFFFFFFFF;
    CHECK_FALSE(memory.try_read<u32>(hole, value));
    CHECK(value == 0U);
    CHECK(memory.fault_count() == 1);
    CHECK(memory.last_fault().address == hole);
    CHECK(memory.last_fault().unmapped);
    CHECK_FALSE(memory.last_fault().is_write);

    CHECK_FALSE(memory.try_write<u32>(hole, 1));
    CHECK(memory.fault_count() == 2);
    CHECK(memory.last_fault().is_write);
}

TEST_CASE_METHOD(MemoryFixture, "writing read-only memory faults", "[memory][fault]") {
    // System fonts live in a read-only region.
    CHECK(memory.read<u32>(shared_data_base) == 0U); // readable
    CHECK(memory.fault_count() == 0);

    CHECK_FALSE(memory.try_write<u32>(shared_data_base, 1));
    CHECK(memory.fault_count() == 1);
    CHECK_FALSE(memory.last_fault().unmapped); // mapped, but not writable
}

TEST_CASE_METHOD(MemoryFixture,
                 "a range straddling a boundary is rejected whole",
                 "[memory][fault]") {
    // The last region ends at 0xFF280000; four bytes starting two before the
    // end are half in and half out.
    const Region& last = memory.regions().back();
    const ppcaddr_t straddle = last.end() - 2;

    CHECK(memory.is_mapped(straddle));         // the first half is fine
    CHECK_FALSE(memory.is_mapped(last.end())); // the second is not
    CHECK(memory.translate(straddle, 4, Access::Read) == nullptr);
    CHECK_FALSE(memory.range_is(straddle, 4, Access::Read));

    u32 value = 0;
    CHECK_FALSE(memory.try_read<u32>(straddle, value));
}

TEST_CASE_METHOD(MemoryFixture, "a fault handler is notified", "[memory][fault]") {
    std::vector<Fault> seen;
    memory.set_fault_handler([&seen](const Fault& fault) { seen.push_back(fault); });

    u32 value = 0;
    CHECK_FALSE(memory.try_read<u32>(0x70000000, value));
    REQUIRE(seen.size() == 1);
    CHECK(seen[0].address == 0x70000000U);
    CHECK(seen[0].size == 4U);
}

TEST_CASE_METHOD(MemoryFixture, "bulk transfers", "[memory][access]") {
    const ppcaddr_t address = data_heap_base + 0x200;
    const std::array<u8, 8> source{1, 2, 3, 4, 5, 6, 7, 8};

    REQUIRE(memory.write_bytes(address, source));

    std::array<u8, 8> destination{};
    REQUIRE(memory.read_bytes(address, destination));
    CHECK(destination == source);

    REQUIRE(memory.fill(address, 8, 0xAA));
    REQUIRE(memory.read_bytes(address, destination));
    CHECK(destination[0] == 0xAA);
    CHECK(destination[7] == 0xAA);

    // Bulk transfers do not swap: a block of guest data is whatever the guest
    // put there.
    const std::array<u8, 4> pattern{0x12, 0x34, 0x56, 0x78};
    REQUIRE(memory.write_bytes(address, pattern));
    CHECK(memory.read<u32>(address) == 0x12345678U);
}

TEST_CASE_METHOD(MemoryFixture, "MMIO reads and writes reach the device", "[memory][mmio]") {
    RecordingDevice device;
    const ppcaddr_t base = 0x70000000; // in the hole, so nothing is overlapped
    std::string error;

    REQUIRE(memory.map_device(base, page_size, device, error));
    CHECK(memory.is_mmio(base));
    CHECK(memory.is_mapped(base));

    // MMIO has no bytes to point at, so translate refuses it.
    CHECK(memory.translate(base, 4, Access::Read) == nullptr);

    device.next_value = 0xCAFEBABE;
    CHECK(memory.read<u32>(base + 0x10) == 0xCAFEBABEU);

    memory.write<u32>(base + 0x20, 0x12345678);

    REQUIRE(device.accesses.size() == 2);
    CHECK(device.accesses[0].offset == 0x10U); // relative to the device base
    CHECK_FALSE(device.accesses[0].is_write);
    CHECK(device.accesses[1].offset == 0x20U);
    CHECK(device.accesses[1].is_write);
    CHECK(device.accesses[1].value == 0x12345678U);

    // No faults: these were legitimate accesses.
    CHECK(memory.fault_count() == 0);
}

TEST_CASE_METHOD(MemoryFixture, "a device cannot overlap real memory", "[memory][mmio]") {
    RecordingDevice device;
    std::string error;

    CHECK_FALSE(memory.map_device(data_heap_base, page_size, device, error));
    CHECK(error.find("overlaps") != std::string::npos);

    CHECK_FALSE(memory.map_device(0x70000800, page_size, device, error));
    CHECK(error.find("aligned") != std::string::npos);
}

TEST_CASE_METHOD(MemoryFixture, "unmapping a device leaves a hole", "[memory][mmio]") {
    RecordingDevice device;
    std::string error;
    REQUIRE(memory.map_device(0x70000000, page_size, device, error));

    memory.unmap_device(0x70000000, page_size);
    CHECK_FALSE(memory.is_mmio(0x70000000));
    CHECK_FALSE(memory.is_mapped(0x70000000));
}

TEST_CASE_METHOD(MemoryFixture, "cache operations notify on line granularity", "[memory][cache]") {
    struct Event {
        CacheOp op;
        ppcaddr_t address;
        u32 size;
    };

    std::vector<Event> events;
    memory.set_cache_listener([&events](CacheOp op, ppcaddr_t address, u32 size) {
        events.push_back({op, address, size});
    });

    // A one-byte flush publishes the whole 32-byte line, as the hardware does.
    memory.dc_flush_range(data_heap_base + 5, 1);
    REQUIRE(events.size() == 1);
    CHECK(events[0].op == CacheOp::Flush);
    CHECK(events[0].address == data_heap_base);
    CHECK(events[0].size == 32U);

    // A range crossing a line boundary covers both lines.
    memory.dc_flush_range(data_heap_base + 30, 4);
    REQUIRE(events.size() == 2);
    CHECK(events[1].address == data_heap_base);
    CHECK(events[1].size == 64U);

    memory.dc_invalidate_range(data_heap_base, 32);
    CHECK(events.back().op == CacheOp::Invalidate);

    memory.ic_invalidate_range(application_base, 4);
    CHECK(events.back().op == CacheOp::CodeChanged);
}

TEST_CASE_METHOD(MemoryFixture, "dc_zero_range actually zeroes", "[memory][cache]") {
    const ppcaddr_t address = data_heap_base + 0x300;
    REQUIRE(memory.fill(address, 64, 0xFF));

    memory.dc_zero_range(address, 32);

    CHECK(memory.read<u32>(address) == 0U);
    CHECK(memory.read<u32>(address + 28) == 0U);
    CHECK(memory.read<u32>(address + 32) == 0xFFFFFFFFU); // the next line is untouched
}

TEST_CASE_METHOD(MemoryFixture, "watchpoints fire on overlapping access", "[memory][watchpoint]") {
    struct Hit {
        std::string name;
        ppcaddr_t address;
        bool is_write;
    };

    std::vector<Hit> hits;

    memory.set_watch_listener(
        [&hits](const Watchpoint& watchpoint, ppcaddr_t address, u32, bool is_write) {
            hits.push_back({watchpoint.name, address, is_write});
        });

    memory.add_watchpoint({"heap-head", data_heap_base, 16, WatchKind::ReadWrite});

    memory.write<u32>(data_heap_base + 4, 1);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].name == "heap-head");
    CHECK(hits[0].is_write);

    static_cast<void>(memory.read<u32>(data_heap_base + 8));
    REQUIRE(hits.size() == 2);
    CHECK_FALSE(hits[1].is_write);

    // Just past the watched range: no hit.
    memory.write<u32>(data_heap_base + 16, 1);
    CHECK(hits.size() == 2);

    // Straddling the end of the range: a hit, because it overlaps.
    memory.write<u32>(data_heap_base + 14, 1);
    CHECK(hits.size() == 3);
}

TEST_CASE_METHOD(MemoryFixture, "a write watchpoint ignores reads", "[memory][watchpoint]") {
    int hits = 0;
    memory.set_watch_listener([&hits](const Watchpoint&, ppcaddr_t, u32, bool) { ++hits; });
    memory.add_watchpoint({"writes-only", data_heap_base, 16, WatchKind::Write});

    static_cast<void>(memory.read<u32>(data_heap_base));
    CHECK(hits == 0);

    memory.write<u32>(data_heap_base, 1);
    CHECK(hits == 1);

    memory.remove_watchpoints("writes-only");
    memory.write<u32>(data_heap_base, 2);
    CHECK(hits == 1);
    CHECK(memory.watchpoints().empty());
}

TEST_CASE("a second initialize is refused", "[memory][lifecycle]") {
    Memory memory;
    std::string error;
    REQUIRE(memory.initialize(error));
    CHECK_FALSE(memory.initialize(error));
    CHECK(error.find("already") != std::string::npos);

    memory.shutdown();
    CHECK_FALSE(memory.ready());
    CHECK(memory.initialize(error)); // and can be brought back up
}
