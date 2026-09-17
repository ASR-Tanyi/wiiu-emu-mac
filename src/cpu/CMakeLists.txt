# wiiu::cpu — Espresso state, the instruction table, decoding and disassembly.
# The interpreter (Phase 4.4 onward) and the JIT (Phase 12) both dispatch off
# the same table.

wiiu_add_library(cpu
    SOURCES
        src/backtrace.cpp
        src/disassembler.cpp
        src/exceptions.cpp
        src/interpreter.cpp
        src/interpreter_fpu.cpp
        src/mmu.cpp
        src/opcodes.cpp
    HEADERS
        include/wiiu/cpu/backtrace.hpp
        include/wiiu/cpu/disassembler.hpp
        include/wiiu/cpu/exceptions.hpp
        include/wiiu/cpu/fpscr.hpp
        include/wiiu/cpu/instruction.hpp
        include/wiiu/cpu/interpreter.hpp
        include/wiiu/cpu/mmu.hpp
        include/wiiu/cpu/opcodes.hpp
        include/wiiu/cpu/state.hpp
    DEPS
        wiiu::common
        wiiu::memory)
