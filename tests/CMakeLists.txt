# Phase 1.7 — test harness.
#
# One executable per subsystem so a failure names the area, and so the CPU
# instruction corpus (Phase 4.12) can grow without slowing everything else.

include(Catch)

# Test data (RPX fixtures and the like) lives beside the tests.
add_compile_definitions(WIIU_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

function(wiiu_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})

    set(target "test_${name}")
    add_executable(${target} ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain ${ARG_DEPS})
    wiiu_set_warnings(${target})
    set_target_properties(${target} PROPERTIES FOLDER "tests")

    catch_discover_tests(${target}
        TEST_PREFIX "${name}."
        REPORTER console)
endfunction()

wiiu_add_test(common
    SOURCES
        common/test_bitfield.cpp
        common/test_bits.cpp
        common/test_config.cpp
        common/test_endian.cpp
        common/test_log.cpp
        common/test_types.cpp
    DEPS
        wiiu::common)

wiiu_add_test(memory
    SOURCES
        memory/test_map.cpp
        memory/test_memory.cpp
    DEPS
        wiiu::memory)

wiiu_add_test(cpu
    SOURCES
        cpu/test_backtrace.cpp
        cpu/test_disassembler.cpp
        cpu/test_exceptions.cpp
        cpu/test_fpu.cpp
        cpu/test_fpscr.cpp
        cpu/test_interpreter.cpp
        cpu/test_mmu.cpp
        cpu/test_opcodes.cpp
    DEPS
        wiiu::cpu
        wiiu::memory)

# Phase 4.3's real check: every instruction the table knows about is
# disassembled, fed back through binutils' PowerPC assembler, and required to
# produce the identical word. Unlike comparing against objdump's text, this
# does not care about spelling conventions -- only that the disassembly means
# the same instruction. Skipped where no cross-assembler is installed.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_test(NAME cpu.roundtrip
        COMMAND ${Python3_EXECUTABLE}
                "${PROJECT_SOURCE_DIR}/scripts/ppc_roundtrip.py"
                "$<TARGET_FILE:ppcdis>")
    set_tests_properties(cpu.roundtrip PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The RPX fixtures are generated rather than committed. This project ships no
# Wii U code at all, and a checked-in executable -- even one we assembled
# ourselves -- is the wrong habit for a repository that must never contain
# anyone else's. Where the cross-assembler is available they are rebuilt as
# part of the build, so a fresh clone has them.
find_program(WIIU_PPC_AS powerpc-linux-gnu-as)
if(Python3_FOUND AND WIIU_PPC_AS)
    set(fixture_dir "${CMAKE_CURRENT_SOURCE_DIR}/loader/data")
    set(fixtures
        "${fixture_dir}/hello.rpx"
        "${fixture_dir}/libtest.rpl"
        "${fixture_dir}/homebrew.rpx"
        "${fixture_dir}/threaded.rpx"
        "${fixture_dir}/clock.rpx")

    add_custom_command(
        OUTPUT ${fixtures}
        COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/make_rpx.py"
        DEPENDS "${PROJECT_SOURCE_DIR}/scripts/make_rpx.py"
        COMMENT "Assembling RPX test fixtures"
        VERBATIM)
    add_custom_target(test_fixtures DEPENDS ${fixtures})
endif()

set(WIIU_GPU_TEST_SOURCES gpu/test_backend.cpp)
if(WIIU_PLATFORM_APPLE)
    # The one test that talks to a real GPU. It needs no window -- the
    # headless path it uses is the same one the compatibility runner (10.5)
    # will boot titles through -- so it runs anywhere the hardware is.
    list(APPEND WIIU_GPU_TEST_SOURCES gpu/test_metal.cpp)
endif()

wiiu_add_test(gpu
    SOURCES ${WIIU_GPU_TEST_SOURCES}
    DEPS wiiu::gpu)

# Phase 1.8 — the JIT probe, run on every build rather than by hand.
#
# The probe source is compiled straight into the test: it belongs to the
# frontend executable, and making a library of two files to share it would be
# more machinery than the sharing is worth.
if(WIIU_PLATFORM_APPLE)
    wiiu_add_test(platform
        SOURCES
            platform/test_jit.cpp
            "${PROJECT_SOURCE_DIR}/src/frontend/platform/jit_probe_macos.cpp"
        DEPS
            wiiu::common)

    target_include_directories(test_platform PRIVATE "${PROJECT_SOURCE_DIR}/src/frontend")

    # MAP_JIT is refused unless the *binary making the call* carries the
    # entitlement -- not the project, not the app bundle, the binary. So the
    # test gets the same signature the app does; without this it would fail
    # for a reason that has nothing to do with the code under test.
    add_custom_command(TARGET test_platform POST_BUILD
        COMMAND codesign --force --sign - --timestamp=none --options runtime
                --entitlements "${PROJECT_SOURCE_DIR}/src/frontend/wiiu-emu.entitlements"
                "$<TARGET_FILE:test_platform>"
        COMMENT "Signing test_platform with the JIT entitlement (Phase 1.8)"
        VERBATIM)
endif()

wiiu_add_test(loader
    SOURCES
        loader/test_metadata.cpp
        loader/test_relocations.cpp
        loader/test_rpl.cpp
        loader/test_vfs.cpp
    DEPS
        wiiu::loader
        wiiu::cpu)

wiiu_add_test(hle
    SOURCES
        hle/test_callbacks.cpp
        hle/test_coreinit.cpp
        hle/test_hle.cpp
        hle/test_ios.cpp
        hle/test_recording.cpp
        hle/test_proc_ui.cpp
        hle/test_scheduler.cpp
    DEPS
        wiiu::hle)

if(TARGET test_fixtures)
    add_dependencies(test_loader test_fixtures)
    add_dependencies(test_hle test_fixtures)
endif()
