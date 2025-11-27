#pragma once
/**
 * @file matrix.hpp
 * @brief Matrix wrapper that can store dense or sparse Eigen matrices via Box.
 */

#include "box.hpp"
#include "box_typed.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>    // std::unique_ptr
#include <typeinfo>  // std::type_info, typeid
#include <stdexcept> // std::runtime_error
#include <cstddef>   // std::size_t

/**
 * @class Matrix
 * @brief Type-erased wrapper that stores dense or sparse Eigen matrices.
 *
 * Internally, Matrix holds a std::unique_ptr<Box> that actually points to a
 * BoxTyped<T> where T is either:
 *   - a dense Eigen::Matrix<Scalar, Dynamic, Dynamic>, or
 *   - a sparse Eigen::SparseMatrix<Scalar, Options, StorageIndex>.
 *
 * For std::vector<Scalar> input, the data is always converted into a dense
 * Eigen::Matrix<Scalar, Dynamic, Dynamic>.
 */
class Matrix {
public:
    /// Default constructor: empty Matrix, no stored object.
    Matrix() = default;

    /**
     * @brief Construct from any Eigen *dense* expression.
     *
     * This constructor accepts any Eigen dense expression derived from
     * Eigen::MatrixBase<Derived> (including Eigen::MatrixXd, fixed-size
     * matrices, blocks, transposes, sums like A+B, etc.) and converts it
     * into a dynamic-size dense Eigen::Matrix<Scalar, Dynamic, Dynamic>
     * stored inside BoxTyped.
     *
     * @tparam Derived Any Eigen dense expression type derived from MatrixBase.
     * @param m        Input Eigen expression to be copied into the wrapper.
     */
    template <typename Derived>
    explicit Matrix(const Eigen::MatrixBase<Derived>& m)
        : isDense_(true)
        , scalarType_(&typeid(typename Derived::Scalar))
    {
        using Scalar   = typename Derived::Scalar;
        using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        DenseMat dense(m);  // Eigen converts any dense expression to a dynamic matrix
        ptr_ = std::make_unique<BoxTyped<DenseMat>>(dense);
    }

    /**
     * @brief Construct from an Eigen sparse matrix.
     *
     * This constructor accepts Eigen::SparseMatrix<Scalar, Options, StorageIndex>
     * and stores it directly in BoxTyped without conversion.
     */
    template <typename Scalar, int Options, typename StorageIndex>
    explicit Matrix(
        const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& m
    )
        : isDense_(false)
        , scalarType_(&typeid(Scalar))
    {
        using SparseMat = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;
        ptr_ = std::make_unique<BoxTyped<SparseMat>>(m);
    }

    /**
     * @brief Construct from a flat std::vector<Scalar> and shape (rows, cols).
     *
     * The data is interpreted in row-major order and copied into a
     * Eigen::Matrix<Scalar, Dynamic, Dynamic> stored inside BoxTyped.
     * This constructor always produces a DENSE Eigen matrix regardless
     * of any future support for sparse types.
     */
    template <typename Scalar>
    Matrix(const std::vector<Scalar>& data, std::size_t rows, std::size_t cols)
        : isDense_(true)
        , scalarType_(&typeid(Scalar))
    {
        if (rows * cols != data.size()) {
            throw std::runtime_error("Matrix: size mismatch in vector constructor");
        }

        using DenseMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMat mat(static_cast<Eigen::Index>(rows),
                     static_cast<Eigen::Index>(cols)); // cast from std::size_t to Eigen::Index used in Eigen::Matrix

        for (std::size_t i = 0; i < data.size(); ++i) {
            std::size_t r = i / cols;
            std::size_t c = i % cols;
            mat(static_cast<Eigen::Index>(r),
                static_cast<Eigen::Index>(c)) = data[i];
        }

        ptr_ = std::make_unique<BoxTyped<DenseMat>>(mat);
    }

    /**
     * @brief Create a Matrix wrapper from an arbitrary user-defined matrix type.
     *
     * This factory stores @p m by value inside a BoxTyped<T>. The stored object
     * can later be accessed via cast<T>().
     *
     * This is intended for user-defined non-Eigen types. For Eigen dense and sparse matrices,
     * prefer the dedicated constructors so that isDense_ and scalar_type() are set appropriately. 
     * User-specified algorithms using Matrix objects initialised from this function should not depend 
     * on isDense() or scalar_type().
     * 
     * Inside from_custom, ptr_ cannot be accessed directly because it's static but 
     * once Matrix object is created, ptr_ can be accessed, hence Matrix objects initialised 
     * from this function can use all features of Matrix class.
     *
     * ### Example:
     * @code
     *   struct MyMatrix { ... };  // user-defined matrix-like type
     * 
     *   MyMatrix A;
     *   Matrix M  = Matrix::from_custom(A);      // wrap A
     *   Matrix M2 = M;                          // copy Matrix
     *
     *   // Recover the stored object:
     *   MyMatrix& ref = M.cast<MyMatrix>();
     * @endcode
     *
     * @tparam T  User-defined matrix-like type.
     * @param m   Object to store inside the Matrix wrapper.
     * @return    A Matrix instance holding a copy of @p m.
     */
    template <typename T>
    static Matrix from_custom(const T& m) { // 
        Matrix M;
        M.isDense_ = true;               // treat as "dense-like" by default
        M.scalarType_ = nullptr;            // unknown scalar type for generic T
        M.ptr_  = std::make_unique<BoxTyped<T>>(m);
        return M;
    }

    /**
     * @brief Copy constructor: performs a deep copy of the stored object.
     *
     * If other holds a BoxTyped<T>, this constructor will call clone()
     * on that Box, which creates a new BoxTyped<T> with a copy of the
     * underlying T. The Matrix wrapper itself does not need to know T.
     */
    Matrix(const Matrix& other)
        : isDense_(other.isDense_)
        , scalarType_(other.scalarType_)
    {
        if (other.ptr_) {
            ptr_ = other.ptr_->clone();  // polymorphic deep copy
        }
    }

    /**
     * @brief Copy assignment: deep copy via clone(), with self-assignment guard.
     */
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (other.ptr_) {
                ptr_ = other.ptr_->clone();
            } else {
                ptr_.reset();
            }
            isDense_   = other.isDense_;
            scalarType_ = other.scalarType_;
        }
        return *this;
    }

    /// Virtual destructor not needed; Matrix is not intended as a base class.
    ~Matrix() = default;

    /**
     * @brief Returns true if this Matrix currently holds a value.
     */
    bool has_value() const noexcept {
        return static_cast<bool>(ptr_);
    }

    /**
     * @brief Returns the dynamic type of the stored (matrix) object.
     *
     * @throw std::runtime_error if the Matrix is empty.
     */
    const std::type_info& type() const {
        if (!ptr_) {
            throw std::runtime_error("Matrix::type() called on empty Matrix");
        }
        return ptr_->type();
    }

    /**
     * @brief Returns the scalar type (float, double, complex, etc.).
     *
     * @return std::type_info describing the scalar type.
     * @throw std::runtime_error if the Matrix is empty.
     */
    const std::type_info& scalar_type() const {
        if (!scalarType_) {
            throw std::runtime_error("Matrix::scalar_type() called on empty Matrix");
        }
        return *scalarType_;
    }

    /**
     * @brief Returns true if the stored matrix is dense.
     */
    bool isDense() const noexcept { return isDense_; }

    /**
     * @brief Downcast to the concrete stored type T (non-const).
     *
     * Because Matrix uses type-erasure internally (the stored object is only
     * known as a Box pointer), this function retrieves the actual underlying
     * Eigen object by checking that the runtime type matches T. If so, it returns
     * a reference to the stored Eigen matrix of type T.
     *
     * ### Example:
     * @code
     *   Eigen::MatrixXd A = Eigen::MatrixXd::Random(3,3);
     *   Matrix M(A);               // stored as BoxTyped<Eigen::MatrixXd>
     *
     *   // Recover the underlying Eigen matrix:
     *   Eigen::MatrixXd& R = M.cast<Eigen::MatrixXd>();
     *   std::cout << R(0,0) << std::endl;
     *
     *   // If the type does not match, this throws std::bad_cast:
     *   // auto& wrong = M.cast<Eigen::SparseMatrix<double>>();
     * @endcode
     *
     * @tparam T The expected concrete Eigen matrix type.
     * @return Reference to the underlying stored matrix of type T.
     *
     * @throw std::runtime_error  If the Matrix is empty.
     * @throw std::bad_cast       If the requested type T does not match the stored type.
     */
    template <typename T>
    T& cast() {
        if (!ptr_) {
            throw std::runtime_error("Matrix::cast() called on empty Matrix");
        }

        if (ptr_->type() != typeid(T)) {
            throw std::bad_cast{};
        }

        auto* typed = static_cast<BoxTyped<T>*>(ptr_.get()); // get() here is method of in-built unique_ptr, ptr_.get() returns a raw pointer to Box
        return typed->get(); // get() here is method of BoxTyped<T>
    }

    /**
     * @brief Downcast to the concrete stored type T (const version).
     *
     * Same logic as the non-const cast(), but returns a const reference to the
     * underlying matrix. This is used when the Matrix object itself is const.
     *
     * @tparam T  The expected concrete matrix type.
     * @return const T& Const reference to the stored matrix of type T.
     *
     * @throw std::runtime_error  If the Matrix is empty (no ptr_).
     * @throw std::bad_cast       If the stored type is not T.
     */
    template <typename T>
    const T& cast() const {
        if (!ptr_) {
            throw std::runtime_error("Matrix::cast() called on empty Matrix");
        }

        if (ptr_->type() != typeid(T)) {
            throw std::bad_cast{};
        }

        auto* typed = static_cast<const BoxTyped<T>*>(ptr_.get());
        return typed->get();
    }

private:
    /// @brief True if the stored matrix is dense; false if sparse.
    bool isDense_ = false;

    /// @brief Pointer to the std::type_info describing the scalar type (double, float, complex, etc.).
    const std::type_info* scalarType_ = nullptr;

    /// @brief Polymorphic storage for the underlying Eigen matrix (dense or sparse).
    std::unique_ptr<Box> ptr_;
};
