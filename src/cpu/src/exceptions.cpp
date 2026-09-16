#include <wiiu/cpu/exceptions.hpp>

namespace wiiu::cpu {

std::string_view exception_name(Exception exception) noexcept {
    switch (exception) {
        case Exception::SystemReset:
            return "system reset";
        case Exception::MachineCheck:
            return "machine check";
        case Exception::Dsi:
            return "data storage";
        case Exception::Isi:
            return "instruction storage";
        case Exception::ExternalInterrupt:
            return "external interrupt";
        case Exception::Alignment:
            return "alignment";
        case Exception::Program:
            return "program";
        case Exception::FpUnavailable:
            return "floating point unavailable";
        case Exception::Decrementer:
            return "decrementer";
        case Exception::SystemCall:
            return "system call";
        case Exception::Trace:
            return "trace";
    }
    return "?";
}

} // namespace wiiu::cpu
