// Phase 8.11 — an indexed draw's indices, converted for Metal.
//
// What is guarded: each of the fetch's byte swaps does to the bytes what
// Latte's index fetch does, GX2's four index types among them, an odd count
// of 16-bit indices included; the converted copy is made once and found
// again while the bytes are the same, made again -- and the old one let go --
// when the title rewrites them; and what cannot be read or converted is
// refused.

#include <wiiu/common/log.hpp>
#include <wiiu/gpu/null_backend.hpp>
#include <wiiu/latte/index_cache.hpp>
#include <wiiu/memory/map.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace wiiu;
using namespace wiiu::latte;

namespace {

// GX2IndexType: VGT_DMA_INDEX_TYPE as it stands.
constexpr u32 kU16Le = 0x0;
constexpr u32 kU32Le = 0x1;
constexpr u32 kU16 = 0x4; // big-endian: a swap within 16 bits
constexpr u32 kU32 = 0x9; // big-endian: a swap within 32 bits

constexpr ppcaddr_t kIndices = 0x10010000;

struct IndexFixture {
    memory::Memory memory;
    gpu::NullBackend backend;
    std::unique_ptr<IndexCache> cache;

    IndexFixture() {
        wiiu::log::set_all_levels(wiiu::log::Level::Off);
        std::string error;
        REQUIRE(memory.initialize(memory::default_regions(), error));
        REQUIRE(backend.initialize(error));
        cache = std::make_unique<IndexCache>(backend, memory);
    }

    ~IndexFixture() { cache.reset(); }

    IndexFixture(const IndexFixture&) = delete;
    IndexFixture& operator=(const IndexFixture&) = delete;

    void put(const std::vector<u8>& bytes) { REQUIRE(memory.write_bytes(kIndices, bytes)); }

    /// What the GPU will read, `size` bytes of it.
    [[nodiscard]] std::vector<u8> contents(const IndexBuffer& indices, std::size_t size) {
        const auto* const bytes = static_cast<const u8*>(backend.buffer_contents(indices.buffer));
        REQUIRE(bytes != nullptr);
        return {bytes, bytes + size};
    }
};

} // namespace

TEST_CASE("the fetch's swaps do to the bytes what Latte's index fetch does",
          "[latte][draw][index]") {
    const std::vector<u8> stored{0x00, 0x01, 0x00, 0x02, 0x00, 0x03};
    // GX2's big-endian 16-bit indices: 1, 2, 3, read little-endian after.
    CHECK(convert_indices(stored, kU16) == std::vector<u8>{0x01, 0x00, 0x02, 0x00, 0x03, 0x00});
    // As stored, for the little-endian types.
    CHECK(convert_indices(stored, kU16Le) == stored);

    const std::vector<u8> words{0x00, 0x00, 0x00, 0x05, 0x01, 0x02, 0x03, 0x04};
    CHECK(convert_indices(words, kU32) ==
          std::vector<u8>{0x05, 0x00, 0x00, 0x00, 0x04, 0x03, 0x02, 0x01});
    CHECK(convert_indices(words, kU32Le) == words);
    // The halves of each word swapped (swap mode 3).
    CHECK(convert_indices(words, kU32Le | (3U << 2)) ==
          std::vector<u8>{0x00, 0x05, 0x00, 0x00, 0x03, 0x04, 0x01, 0x02});

    // A size Latte does not have converts nothing.
    CHECK(convert_indices(words, 2).empty());
}

TEST_CASE_METHOD(IndexFixture,
                 "indices are converted once, and again when rewritten",
                 "[latte][draw][index]") {
    put({0x00, 0x01, 0x00, 0x02, 0x00, 0x03});
    std::string error;
    const std::optional<IndexBuffer> first = cache->indices(kIndices, 3, kU16, error);
    INFO(error);
    REQUIRE(first.has_value());
    CHECK_FALSE(first->is_32bit);
    CHECK(contents(*first, 6) == std::vector<u8>{0x01, 0x00, 0x02, 0x00, 0x03, 0x00});

    // The same bytes: the same copy.
    const std::optional<IndexBuffer> again = cache->indices(kIndices, 3, kU16, error);
    REQUIRE(again.has_value());
    CHECK(again->buffer == first->buffer);
    CHECK(cache->stats().hits == 1);
    CHECK(cache->stats().conversions == 1);

    // The title rewrites them: converted again, the old copy let go.
    put({0x00, 0x02, 0x00, 0x01, 0x00, 0x00});
    const std::optional<IndexBuffer> rewritten = cache->indices(kIndices, 3, kU16, error);
    REQUIRE(rewritten.has_value());
    CHECK(rewritten->buffer != first->buffer);
    CHECK(contents(*rewritten, 6) == std::vector<u8>{0x02, 0x00, 0x01, 0x00, 0x00, 0x00});
    CHECK(cache->stats().conversions == 2);
    CHECK(cache->size() == 1);
    CHECK(backend.live_count(gpu::ResourceKind::Buffer) == 1);

    // The same memory read as 32-bit indices is another copy.
    const std::optional<IndexBuffer> wide = cache->indices(kIndices, 1, kU32, error);
    REQUIRE(wide.has_value());
    CHECK(wide->is_32bit);
    CHECK(contents(*wide, 4) == std::vector<u8>{0x01, 0x00, 0x02, 0x00});
    CHECK(cache->size() == 2);
    CHECK(backend.rejected() == 0);

    cache->clear();
    CHECK(backend.live_count(gpu::ResourceKind::Buffer) == 0);
}

TEST_CASE_METHOD(IndexFixture,
                 "indices that cannot be read or converted are refused",
                 "[latte][draw][index]") {
    std::string error;
    CHECK_FALSE(cache->indices(0x00000100, 3, kU16, error).has_value());
    CHECK(error.find("not readable guest memory") != std::string::npos);
    CHECK_FALSE(cache->indices(kIndices, 0, kU16, error).has_value());
    CHECK(error.find("not a draw's worth") != std::string::npos);
    CHECK_FALSE(cache->indices(kIndices, 3, 2, error).has_value());
    CHECK(error.find("no index size") != std::string::npos);
    CHECK(cache->size() == 0);
}
