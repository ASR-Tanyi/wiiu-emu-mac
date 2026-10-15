#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace wiiu;
using namespace wiiu::memory;

TEST_CASE("regions are page aligned and ordered", "[memory][map]") {
    const auto regions = default_regions();
    REQUIRE_FALSE(regions.empty());

    ppcaddr_t previous_end = 0;
    for (const Region& region : regions) {
        INFO("region " << region.name);
        CHECK(region.size > 0);
        CHECK(region.base % page_size == 0);
        CHECK(region.size % page_size == 0);

        // Ordered, and non-overlapping.
        CHECK(region.base >= previous_end);
        previous_end = region.end();
    }
}

TEST_CASE("the documented addresses are where they should be", "[memory][map]") {
    const auto regions = default_regions();

    const Region* const mem1 = find_region(regions, mem1_base);
    REQUIRE(mem1 != nullptr);
    CHECK(mem1->name == "mem1");
    CHECK(mem1->size == 0x02000000U); // 32 MiB
    CHECK(mem1->physical == 0x00000000U);
    CHECK(mem1->backing == Backing::Mem1);

    const Region* const data = find_region(regions, data_heap_base);
    REQUIRE(data != nullptr);
    CHECK(data->name == "data");
    CHECK(data->physical == 0x20000000U);

    // The application region always ends where the data heap begins, whatever
    // cos.xml says about where it starts.
    const Region* const app = find_region(regions, application_base);
    REQUIRE(app != nullptr);
    CHECK(app->end() == application_end);
    CHECK(app->end() == data_heap_base);
}

TEST_CASE("shared data is read-only", "[memory][map]") {
    const Region* const shared = find_region(default_regions(), shared_data_base);
    REQUIRE(shared != nullptr);
    CHECK(has(shared->access, Access::Read));
    CHECK_FALSE(has(shared->access, Access::Write));
}

TEST_CASE("the codegen area has no physical backing", "[memory][map]") {
    const Region* const codegen = find_region(default_regions(), codegen_base);
    REQUIRE(codegen != nullptr);
    CHECK(codegen->backing == Backing::None);

    physaddr_t physical = 0;
    CHECK_FALSE(to_physical(*codegen, codegen_base, physical));
}

TEST_CASE("virtual to physical translation", "[memory][map]") {
    const Region* const mem1 = find_region(default_regions(), mem1_base);
    REQUIRE(mem1 != nullptr);

    physaddr_t physical = 0;
    REQUIRE(to_physical(*mem1, mem1_base, physical));
    CHECK(physical == 0x00000000U);

    REQUIRE(to_physical(*mem1, mem1_base + 0x1000, physical));
    CHECK(physical == 0x00001000U);

    // Out of the region.
    CHECK_FALSE(to_physical(*mem1, mem1_base + mem1->size, physical));
}

TEST_CASE("physical to virtual is the way back", "[memory][map]") {
    const auto regions = default_regions();

    // Every backed region round-trips at its start, middle and last byte.
    for (const Region& region : regions) {
        if (region.backing == Backing::None) {
            continue;
        }
        for (const u32 offset : {0U, region.size / 2, region.size - 1}) {
            physaddr_t physical = 0;
            REQUIRE(to_physical(region, region.base + offset, physical));
            ppcaddr_t back = 0;
            INFO(region.name << " + " << offset);
            REQUIRE(to_virtual(regions, physical, back));
            CHECK(back == region.base + offset);
        }
    }

    // MEM1 at physical 0 is the GPU's favourite address and lives at
    // 0xF4000000 virtual.
    ppcaddr_t address = 0;
    REQUIRE(to_virtual(regions, 0x00000100, address));
    CHECK(address == mem1_base + 0x100);

    // Physical memory nothing maps finds nothing, rather than a neighbour.
    CHECK_FALSE(to_virtual(regions, 0xF0000000, address));
}

TEST_CASE("addresses outside every region find nothing", "[memory][map]") {
    const auto regions = default_regions();
    CHECK(find_region(regions, 0x00000000) == nullptr); // below everything
    CHECK(find_region(regions, 0xFFFFFFFF) == nullptr); // above the kernel heap
    CHECK(find_region(regions, 0x70000000) == nullptr); // the gap after the data heap
}

TEST_CASE("access flags combine", "[memory][map]") {
    STATIC_REQUIRE(has(read_write, Access::Read));
    STATIC_REQUIRE(has(read_write, Access::Write));
    STATIC_REQUIRE_FALSE(has(read_write, Access::Execute));
    STATIC_REQUIRE(has(read_write_execute, Access::Execute));
    STATIC_REQUIRE_FALSE(has(read_only, Access::Write));
}
