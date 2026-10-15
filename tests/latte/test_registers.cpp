// Phase 8.4 — the register file.
//
// The windows are what matter: a `SET_*` packet addresses registers relative
// to its own window, and getting a base wrong writes a plausible value into
// the wrong register, which shows up frames later as the wrong blend mode.

#include <wiiu/latte/registers.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <string>

using namespace wiiu;
using namespace wiiu::latte;

TEST_CASE("the windows sit where R600 puts them", "[latte][registers]") {
    CHECK(window_range(Window::Config).base == 0x08000);
    CHECK(window_range(Window::Context).base == 0x28000);
    CHECK(window_range(Window::AluConst).base == 0x30000);
    CHECK(window_range(Window::Resource).base == 0x38000);
    CHECK(window_range(Window::Sampler).base == 0x3C000);
    CHECK(window_range(Window::LoopConst).base == 0x3E200);
    CHECK(window_range(Window::BoolConst).base == 0x3E380);

    // 256 ALU constants of four words each, per the R600 register map.
    CHECK(window_range(Window::AluConst).words() == 0x800);

    // No two windows overlap, and each starts where the one before it ends
    // or later.
    constexpr std::array windows{Window::Config,
                                 Window::Context,
                                 Window::AluConst,
                                 Window::Resource,
                                 Window::Sampler,
                                 Window::CtlConst,
                                 Window::LoopConst,
                                 Window::BoolConst};
    for (std::size_t index = 1; index < windows.size(); ++index) {
        CHECK(window_range(windows[index]).base >= window_range(windows[index - 1]).end);
    }
    CHECK(window_range(Window::BoolConst).end == register_space_bytes);
}

TEST_CASE("a window write lands relative to its window", "[latte][registers]") {
    auto registers = std::make_unique<RegisterFile>();
    std::string error;

    // SET_CONTEXT_REG at offset 0x10 is byte address 0x28040.
    const std::array<u32, 2> values{0xAAAA, 0xBBBB};
    REQUIRE(registers->write_window(Window::Context, 0x10, values, error));
    CHECK(registers->read(0x28040) == 0xAAAA);
    CHECK(registers->read(0x28044) == 0xBBBB);
    CHECK(registers->read(0x28048) == 0);
    CHECK(registers->writes() == 2);

    // The same offset in another window is another register.
    REQUIRE(registers->write_window(Window::Config, 0x10, values, error));
    CHECK(registers->read(0x08040) == 0xAAAA);
}

TEST_CASE("a window write past its window is refused whole", "[latte][registers]") {
    auto registers = std::make_unique<RegisterFile>();
    std::string error;

    // The last word of the sampler window is fine; one more is not, and a
    // refused write writes nothing -- not the part that fitted.
    const u32 last = window_range(Window::Sampler).words() - 1;
    const std::array<u32, 2> two{1, 2};
    CHECK_FALSE(registers->write_window(Window::Sampler, last, two, error));
    CHECK(error.find("sampler") != std::string::npos);
    CHECK(registers->read(window_range(Window::Sampler).base + (last * 4)) == 0);
    CHECK(registers->writes() == 0);

    const std::array<u32, 1> one{7};
    CHECK(registers->write_window(Window::Sampler, last, one, error));
}

TEST_CASE("a Type 0 write is absolute and bounded", "[latte][registers]") {
    auto registers = std::make_unique<RegisterFile>();
    std::string error;

    const std::array<u32, 3> values{1, 2, 3};
    REQUIRE(registers->write_type0(0x08958 / 4, values, error));
    CHECK(registers->read(reg::vgt_primitive_type) == 1);
    CHECK(registers->read(reg::vgt_index_type) == 2);

    CHECK_FALSE(registers->write_type0(register_count - 1, values, error));
    CHECK(error.find("runs past") != std::string::npos);
}

TEST_CASE("reads outside the space or unaligned are zero", "[latte][registers]") {
    auto registers = std::make_unique<RegisterFile>();
    CHECK(registers->write(0x28000, 5));
    CHECK(registers->read(0x28002) == 0);
    CHECK(registers->read(register_space_bytes) == 0);
    CHECK_FALSE(registers->write(register_space_bytes, 1));
    CHECK_FALSE(registers->write(0x28001, 1));

    registers->reset();
    CHECK(registers->read(0x28000) == 0);
    CHECK(registers->writes() == 0);
}
