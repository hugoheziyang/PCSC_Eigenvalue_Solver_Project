#pragma once
/**
 * @file matrix.hpp
 * @brief Type-erased wrapper for dense or sparse Eigen matrices.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../box/box.hpp"
#include "../box/box_typed.hpp"
#include "../core/types.hpp"

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>
#include <cstddef>

/**
 * @class Matrix
 * @brief Type-erased wrapper storing a dense or sparse Eigen matrix.
 *
 * This class enforces the following invariants:
 *  - A Matrix object always owns a valid matrix value
 *  - The stored matrix is always one of:
 *        Dense<Scalar>  = Eigen::Matrix<Scalar, Dynamic, Dynamic>
 *        Sparse<Scalar> = Eigen::SparseMatrix<Scalar>
 *  - No empty Matrix state exists
 *
 * The class uses Box and BoxTyped internally for type-erasure while
 * keeping the public interface non-templated.
 */
class Matrix {
public:
    /// Canonical dense matrix type alias.
    template <ScalarConcept Scalar>
    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    /// Canonical sparse matrix type alias.
    template <ScalarConcept Scalar>
    using Sparse = Eigen::SparseMatrix<Scalar>;

    /// Default constructor deleted: a Matrix must always hold a value.
    Matrix() = delete;

    /// Copy constructor deleted: Matrix is non-copyable by design.
    Matrix(const Matrix&) = delete;

    /// Copy assignment deleted: Matrix is non-copyable by design.
    Matrix& operator=(const Matrix&) = delete;

    /// Move constructor deleted: moved-from empty states are forbidden.
    Matrix(Matrix&&) = delete;

    /// Move assignment deleted: moved-from empty states are forbidden.
    Matrix& operator=(Matrix&&) = delete;

    /**
     * @brief Construct a Matrix from any Eigen dense expression.
     *
     * The input expression is evaluated into the canonical dense type
     * Dense<Scalar> and stored by value.
     *
     * @tparam Derived Eigen dense expression type derived from MatrixBase.
     * @param m Dense Eigen expression to copy into the wrapper.
     */
    template <typename Derived>
        requires ScalarConcept<typename Derived::Scalar>
    explicit Matrix(const Eigen::MatrixBase<Derived>& m)
        : isDense_(true)
        , scalarType_(&typeid(typename Derived::Scalar))
        , ptr_(make_dense(m))
    {}

    /**
     * @brief Construct a Matrix from an Eigen sparse matrix.
     *
     * The input matrix is converted into the canonical Sparse<Scalar> type
     * and stored by value. Storage options and index types are normalized.
     *
     * @tparam Scalar       Scalar type of the matrix.
     * @tparam Options      Eigen storage options.
     * @tparam StorageIndex Index type used by the input sparse matrix.
     * @param m             Input sparse matrix.
     */
    template <ScalarConcept Scalar, int Options, typename StorageIndex>
    explicit Matrix(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& m)
        : isDense_(false)
        , scalarType_(&typeid(Scalar))
        , ptr_(make_sparse(m))
    {}

    /**
     * @brief Construct a dense matrix from a flat std::vector in row-major order.
     *
     * The data is interpreted in row-major order and copied into a canonical
     * Dense<Scalar> matrix stored by value.
     *
     * @tparam Scalar Scalar type of the entries.
     * @param data    Flat row-major data buffer.
     * @param rows    Number of rows.
     * @param cols    Number of columns.
     *
     * @throws std::runtime_error If data.size() != rows * cols.
     */
    template <ScalarConcept Scalar>
    Matrix(const std::vector<Scalar>& data,
           std::size_t rows,
           std::size_t cols)
        : isDense_(true)
        , scalarType_(&typeid(Scalar))
        , ptr_(make_from_vector(data, rows, cols))
    {}

    /**
     * @brief Check whether the stored matrix is dense.
     *
     * @return True if the stored matrix is dense, false if it is sparse.
     */
    bool isDense() const { return isDense_; }

    /**
     * @brief Get the scalar type of the stored matrix.
     *
     * The returned std::type_info corresponds to the Scalar template
     * parameter of the underlying Eigen type.
     *
     * @return Reference to a std::type_info describing the scalar type.
     */
    const std::type_info& scalar_type() const {
        return *scalarType_;
    }

    /**
     * @brief Get the dynamic type of the stored object.
     *
     * The returned std::type_info corresponds to either Dense<Scalar> or
     * Sparse<Scalar> for some scalar type Scalar.
     *
     * @return Reference to a std::type_info describing the stored type.
     */
    const std::type_info& type() const {
        return ptr_->type();
    }

    /**
     * @brief Retrieve the stored object with its exact static type.
     *
     * This function attempts to downcast the internal BoxTyped to the
     * requested type T and returns a reference to the contained matrix.
     *
     * @tparam T Expected concrete matrix type (Dense<Scalar> or Sparse<Scalar>).
     * @return Reference to the stored matrix of type T.
     *
     * @throws std::bad_cast If the stored type does not match T exactly.
     */
    template <typename T>
    T& cast() {
        check_type<T>();
        auto* typed = static_cast<BoxTyped<T>*>(ptr_.get());
        return typed->get();
    }

    /**
     * @brief Retrieve the stored object with its exact static type (const).
     *
     * Const-qualified overload of cast() returning a const reference to
     * the underlying matrix.
     *
     * @tparam T Expected concrete matrix type (Dense<Scalar> or Sparse<Scalar>).
     * @return Const reference to the stored matrix of type T.
     *
     * @throws std::bad_cast If the stored type does not match T exactly.
     */
    template <typename T>
    const T& cast() const {
        check_type<T>();
        auto* typed = static_cast<const BoxTyped<T>*>(ptr_.get());
        return typed->get();
    }

private:
    // Helper factory for dense matrices
    template <typename Derived>
        requires ScalarConcept<typename Derived::Scalar> 
    static std::unique_ptr<Box>
    make_dense(const Eigen::MatrixBase<Derived>& m)
    {
        using Scalar = typename Derived::Scalar;
        Dense<Scalar> dense(m);
        return std::make_unique<BoxTyped<Dense<Scalar>>>(dense);
    }

    // Helper factory for sparse matrices
    template <ScalarConcept Scalar, int Options, typename StorageIndex>
    static std::unique_ptr<Box>
    make_sparse(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& m)
    {
        Sparse<Scalar> sparse(m);
        return std::make_unique<BoxTyped<Sparse<Scalar>>>(sparse);
    }

    // Helper factory for construction from std::vector
    template <ScalarConcept Scalar>
    static std::unique_ptr<Box>
    make_from_vector(const std::vector<Scalar>& data,
                     std::size_t rows,
                     std::size_t cols)
    {
        if (rows * cols != data.size()) {
            throw std::runtime_error("Matrix: size mismatch in vector constructor");
        }

        Dense<Scalar> mat(
            static_cast<Eigen::Index>(rows),
            static_cast<Eigen::Index>(cols)
        );

        for (std::size_t i = 0; i < data.size(); ++i) {
            const std::size_t r = i / cols;
            const std::size_t c = i % cols;
            mat(
                static_cast<Eigen::Index>(r),
                static_cast<Eigen::Index>(c)
            ) = data[i];
        }

        return std::make_unique<BoxTyped<Dense<Scalar>>>(mat);
    }

    // Internal type check used by cast
    template <typename T>
    void check_type() const {
        if (ptr_->type() != typeid(T)) {
            throw std::bad_cast{};
        }
    }

private:
    bool isDense_;
    const std::type_info* scalarType_;
    std::unique_ptr<Box> ptr_;
};
