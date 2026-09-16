#include <wiiu/memory/address_space.hpp>

#include <fmt/format.h>

#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <utility>

namespace wiiu::memory {
namespace {

[[nodiscard]] int to_host_protection(Access access) noexcept {
    int protection = 0;
    if (has(access, Access::Read)) {
        protection |= PROT_READ;
    }
    if (has(access, Access::Write)) {
        protection |= PROT_WRITE;
    }
    // Guest execute permission is not host execute permission: guest code is
    // interpreted or recompiled, never run in place. Mapping it executable
    // would hand a guest bug a way out of the sandbox for no benefit.
    return protection;
}

} // namespace

AddressSpace::~AddressSpace() {
    release();
}

AddressSpace::AddressSpace(AddressSpace&& other) noexcept
    : base_(std::exchange(other.base_, nullptr)) {}

AddressSpace& AddressSpace::operator=(AddressSpace&& other) noexcept {
    if (this != &other) {
        release();
        base_ = std::exchange(other.base_, nullptr);
    }
    return *this;
}

bool AddressSpace::reserve(std::string& error) {
    if (base_ != nullptr) {
        error = "address space already reserved";
        return false;
    }

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    // Linux would otherwise account 4 GiB against overcommit limits. macOS
    // does not define this and does not need it.
    flags |= MAP_NORESERVE;
#endif

    void* const mapping = ::mmap(nullptr, address_space_size, PROT_NONE, flags, -1, 0);
    if (mapping == MAP_FAILED) {
        error = fmt::format("could not reserve {} GiB of address space: {}",
                            address_space_size >> 30,
                            std::strerror(errno));
        return false;
    }

    base_ = static_cast<u8*>(mapping);
    return true;
}

bool AddressSpace::commit(ppcaddr_t base, u32 size, Access access, std::string& error) {
    if (base_ == nullptr) {
        error = "address space not reserved";
        return false;
    }
    if (size == 0) {
        return true;
    }
    if ((base % page_size) != 0 || (size % page_size) != 0) {
        error = fmt::format("commit range {:#010x}+{:#x} is not page aligned", base, size);
        return false;
    }
    if (static_cast<u64>(base) + size > address_space_size) {
        error = fmt::format("commit range {:#010x}+{:#x} runs past the address space", base, size);
        return false;
    }

    void* const target = base_ + base;
    const int protection = to_host_protection(access);

    // MAP_FIXED over our own reservation: the range is already ours, so this
    // replaces the PROT_NONE placeholder rather than stealing someone else's
    // mapping.
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif

    void* const mapping = ::mmap(target, size, protection, flags, -1, 0);
    if (mapping == MAP_FAILED) {
        error =
            fmt::format("could not commit {:#010x}+{:#x}: {}", base, size, std::strerror(errno));
        return false;
    }

    return true;
}

bool AddressSpace::decommit(ppcaddr_t base, u32 size, std::string& error) {
    if (base_ == nullptr) {
        error = "address space not reserved";
        return false;
    }
    if (size == 0) {
        return true;
    }

    // Back to PROT_NONE rather than munmap, so the reservation stays whole and
    // nothing else can claim the hole.
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif

    void* const mapping = ::mmap(base_ + base, size, PROT_NONE, flags, -1, 0);
    if (mapping == MAP_FAILED) {
        error =
            fmt::format("could not decommit {:#010x}+{:#x}: {}", base, size, std::strerror(errno));
        return false;
    }
    return true;
}

void AddressSpace::release() noexcept {
    if (base_ != nullptr) {
        ::munmap(base_, address_space_size);
        base_ = nullptr;
    }
}

} // namespace wiiu::memory
