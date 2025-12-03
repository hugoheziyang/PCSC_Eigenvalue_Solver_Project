#pragma once
/**
 * @file box.hpp
 * @brief Defines the abstract base class for type-erased storage.
 */

#include <typeinfo>
#include <memory>

/**
 * @class Box
 * @brief Abstract base class for type-erased objects.
 *
 * Box provides the interface required for type-erasure: the ability to store
 * an object of unknown compile-time type and interact with it through a
 * uniform base class. It supports:
 *
 *  - querying the runtime type (via type()),
 *  - creating a deep copy of the underlying object (via clone()),
 *  - and safe destruction through a virtual destructor.
 *
 * Concrete storage is implemented in BoxTyped<T>.
 * This mechanism allows higher-level wrappers (e.g. Matrix) to store dense or
 * sparse Eigen matrices of any type without knowing the concrete type at
 * compile time.
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
     * This function is essential for enabling copy operations on a
     * type-erased wrapper such as:
     *
     *     Matrix B = A;   // deep copy
     *
     * In such a wrapper, the underlying stored object is held via a
     * pointer to Box (e.g. std::unique_ptr<Box>). Because the concrete
     * type is unknown at compile time, the normal copy constructor cannot
     * be used. Instead, clone() provides a virtual "copy constructor"
     * that each BoxTyped<T> implements to return a new heap-allocated
     * copy of the contained T.
     *
     * @note
     * clone() must return std::unique_ptr<Box> (the base type), not
     * std::unique_ptr<BoxTyped<T>>. Returning the base pointer preserves
     * type-erasure:
     *
     *  - If clone() returned std::unique_ptr<BoxTyped<T>>, then Matrix
     *    would need to know the concrete type T in order to store it.
     *  - This would force Matrix to become a template, or require a
     *    separate storage path for every possible BoxTyped<T> type.
     *  - Returning std::unique_ptr<Box> allows Matrix to remain a single,
     *    non-templated type-erased wrapper that can hold *any* T.
     *
     * @return A std::unique_ptr<Box> owning a deep copy of this object.
     */
    virtual std::unique_ptr<Box> clone() const = 0;
};
