#pragma once
/**
 * @file box_typed.hpp
 * @brief Defines the BoxTyped template for storing concrete objects in a Box.
 */

#include "box.hpp"
#include <memory>   // std::unique_ptr
#include <typeinfo> // typeid

/**
 * @class BoxTyped
 * @brief Concrete Box that stores an object of type T by value.
 *
 * BoxTyped<T> implements the Box interface for a specific type T:
 *  - it stores a T inside,
 *  - reports T's type via type(),
 *  - and implements clone() by allocating a new BoxTyped<T> with a copy of T.
 *
 * This is the building block used by type-erased wrappers to hold
 * different concrete types behind a single Box* / std::unique_ptr<Box>.
 *
 * @tparam T The concrete type being stored.
 */
template <typename T>
class BoxTyped : public Box {
public:
    /// Default constructor: default-constructs the stored T.
    BoxTyped() = default;

    /// Construct from a const reference to T (copies value into storage).
    explicit BoxTyped(const T& value)
        : value_(value)
    {}

    /// Copying disabled; cloning is performed via clone().
    BoxTyped(const BoxTyped&) = delete;
    BoxTyped& operator=(const BoxTyped&) = delete;

    /// Virtual destructor.
    ~BoxTyped() override = default;

    /**
     * @brief Returns the dynamic type of the stored T.
     *
     * This lets users query the type at runtime via the Box interface.
     */
    const std::type_info& type() const noexcept override {
        return typeid(T);
    }

    /**
     * @brief Creates a polymorphic deep copy of this BoxTyped<T>.
     *
     * Allocates a new BoxTyped<T> on the heap, initialized with a copy of the
     * stored T, and returns it as a std::unique_ptr<Box>.
     */
    std::unique_ptr<Box> clone() const override {
        return std::make_unique<BoxTyped<T>>(value_);
    }

    /// Access the stored T (non-const).
    T& get() noexcept { return value_; }

    /// Access the stored T (const).
    const T& get() const noexcept { return value_; }

    /// Convenience dereference operators to access T directly.
    T& operator*() noexcept { return value_; }
    const T& operator*() const noexcept { return value_; }

    T* operator->() noexcept { return &value_; }
    const T* operator->() const noexcept { return &value_; }

private:
    T value_;  ///< The stored object of type T.
};
