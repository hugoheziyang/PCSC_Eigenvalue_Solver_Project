#pragma once
/**
 * @file box.hpp
 * @brief Defines the abstract base class for type-erased storage.
 */

#include <typeinfo>
#include <memory>

namespace EigSol {

/**
 * @class Box
 * @brief Abstract base class for type-erased objects.
 *
 * Box defines the minimal interface required for runtime type-erasure:
 *
 *  - querying the dynamic type of the stored object (via type()),
 *  - creating a polymorphic deep copy of the object (via clone()),
 *  - ensuring correct cleanup through a virtual destructor.
 *
 * Concrete storage is implemented by the templated BoxTyped<T>, which
 * holds an object of type T by value and exposes it through this base
 * interface. 
 *
 * This design enables higher-level wrappers or containers to store
 * arbitrary types without requiring templates at the wrapper level.
 * Wrappers can interact with stored objects uniformly through Box*
 * while preserving correct type information and deep-copy behavior
 * when needed.
 */
class Box {
public:
    /// Default constructor.
    Box() = default;

    /// Copying disabled; use clone() in derived classes instead.
    Box(const Box&) = delete;
    Box& operator=(const Box&) = delete;
    
    /// Virtual destructor to ensure proper cleanup through base pointer.
    virtual ~Box() = default;

    /**
     * @brief Returns the dynamic type of the stored object.
     *
     * Derived classes must return the std::type_info corresponding
     * to the concrete stored type.
     */
    virtual const std::type_info& type() const noexcept = 0;

    /**
     * @brief Creates a polymorphic deep copy of the stored object.
     *
     * This function provides the ability to copy an object whose concrete
     * type is known only at runtime. It is the type-erased analogue of a
     * virtual copy constructor.
     *
     * This mechanism is intentionally preserved because it enables the possibility 
     * of type-erased wrappers to support deep-copy semantics.
     *
     * Example use case (hypothetical future wrapper):
     *
     *     TypeErasedValue a = TypeErasedValue::make<T>(value);
     *     TypeErasedValue b = a.clone_value();   // b owns a deep copy of T
     *
     * In such wrappers, clone() allows:
     *   - copying objects stored behind std::unique_ptr<Box>,
     *   - duplicating values without knowing the concrete type T,
     *   - keeping the wrapper itself non-templated while still allowing
     *     generic storage of arbitrary types.
     *
     * @note
     * clone() must return std::unique_ptr<Box>, not std::unique_ptr<BoxTyped<T>>,
     * so that the caller can store the copy via a base-class pointer without
     * needing to know the underlying type T.
     *
     * @return A std::unique_ptr<Box> owning a deep copy of the stored object.
     */
    virtual std::unique_ptr<Box> clone() const = 0;
};

} // end namespace