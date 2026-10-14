// Phase 1.4 — ownership for metal-cpp objects.
//
// metal-cpp is a thin C++ shim over Objective-C, and it does not bring ARC
// with it: every object carries a retain count that the caller is responsible
// for. The rules are the Objective-C ones, which are easy to state and easy
// to get wrong at three in the morning:
//
//   * anything from a method whose name begins `new`, `alloc`, `copy` or
//     `mutableCopy` arrives owned, and must be released exactly once;
//   * anything else arrives *autoreleased*, and is valid only until the
//     enclosing pool drains -- keeping one past that is a use-after-free that
//     usually survives long enough to look like something else's bug.
//
// So this file is written before any Metal code rather than after the first
// leak. `Ref` owns; `Ref::retained` adopts something it does not own; and
// `AutoreleasePool` is what makes the second rule safe -- `nextDrawable` and
// most of the descriptor factories return autoreleased objects, so a frame
// loop without a pool per frame grows without bound while looking correct.
#pragma once

#include <utility>

namespace wiiu::gpu::metal {

/// An owning pointer to a metal-cpp object: release on destruction, move
/// freely, and copy only through an explicit retain.
template <typename T>
class Ref {
public:
    Ref() noexcept = default;

    /// Adopt an object this code already owns -- the result of a `new*`,
    /// `alloc*` or `copy*` method.
    explicit Ref(T* owned) noexcept : pointer_(owned) {}

    /// Take a reference to an object owned by someone else, such as an
    /// autoreleased one that has to outlive its pool.
    [[nodiscard]] static Ref retained(T* borrowed) noexcept {
        if (borrowed != nullptr) {
            borrowed->retain();
        }
        return Ref{borrowed};
    }

    ~Ref() { reset(); }

    Ref(const Ref& other) noexcept : pointer_(other.pointer_) {
        if (pointer_ != nullptr) {
            pointer_->retain();
        }
    }

    Ref& operator=(const Ref& other) noexcept {
        if (this != &other) {
            T* const adopted = other.pointer_;
            if (adopted != nullptr) {
                adopted->retain();
            }
            reset();
            pointer_ = adopted;
        }
        return *this;
    }

    Ref(Ref&& other) noexcept : pointer_(std::exchange(other.pointer_, nullptr)) {}

    Ref& operator=(Ref&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }

    void reset(T* owned = nullptr) noexcept {
        if (pointer_ != nullptr) {
            pointer_->release();
        }
        pointer_ = owned;
    }

    /// Give up ownership without releasing.
    [[nodiscard]] T* disown() noexcept { return std::exchange(pointer_, nullptr); }

    [[nodiscard]] T* get() const noexcept { return pointer_; }

    T* operator->() const noexcept { return pointer_; }

    explicit operator bool() const noexcept { return pointer_ != nullptr; }

private:
    T* pointer_ = nullptr;
};

/// A scope in which autoreleased objects are collected, drained on exit.
///
/// One per frame at minimum. Metal hands back autoreleased drawables,
/// descriptors and command buffers, and without a pool they accumulate until
/// the process ends -- which on a long run is the difference between steady
/// memory and a gigabyte an hour.
class AutoreleasePool {
public:
    AutoreleasePool();
    ~AutoreleasePool();

    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;
    AutoreleasePool(AutoreleasePool&&) = delete;
    AutoreleasePool& operator=(AutoreleasePool&&) = delete;

private:
    void* pool_ = nullptr;
};

} // namespace wiiu::gpu::metal
