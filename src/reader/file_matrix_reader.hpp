/**
 * @file file_matrix_reader.hpp
 * @brief Functions for reading dense or sparse matrices from a file
 *
 * The routines interpret a simple text-based format:
 * - a keyword describing the storage type,
 * - two integers giving the number of rows as well as the number of columns,
 * - the matrix entries according to the chosen storage layout.
 */

#include "../core/types.hpp"
#include "../matrix/matrix.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

namespace EigSol {


/**
 * @brief Reads a dense matrix from an already opened input stream.
 *
 * @tparam Scalar scalar type satisfying ScalarConcept
 * @param in input stream used for extraction
 * @param rows number of rows
 * @param cols number of columns
 * @return a dense matrix wrapped inside the Matrix type
 *
 * Values are read element by element in row-major order.
 * For a complex scalar, two real numbers are expected for each entry.
 * An exception is thrown when extraction fails for any value.
 */
template <ScalarConcept Scalar>
Matrix readInsideDenseMatrix(std::ifstream& in, int rows, int cols) {
    if (rows < 0 || cols < 0) {
        throw std::runtime_error("Negative matrix dimensions");
    }

    using Dense = Matrix::Dense<Scalar>;
    Dense dense(rows, cols);

    if constexpr (is_complex_of_floating<Scalar>::value) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                using Real = typename Scalar::value_type;
                Real re{};
                Real im{};
                if (!(in >> re >> im)) {
                    throw std::runtime_error("Failed to read complex entry in dense matrix");
                }
                dense(r, c) = Scalar(re, im);
            }
        }
    } else {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                Scalar v{};
                if (!(in >> v)) {
                    throw std::runtime_error("Failed to read scalar entry in dense matrix");
                }
                dense(r, c) = v;
            }
        }
    }
    return Matrix(dense);
}

/**
 * @brief Reads a sparse matrix from an already opened input stream.
 *
 * @tparam Scalar scalar type satisfying ScalarConcept
 * @param in input stream
 * @param rows number of rows
 * @param cols number of columns
 * @return a sparse matrix wrapped inside the Matrix type
 *
 * The stream must provide:
 * - one integer giving the number of non-zero entries at the start
 * - for each entry, a row index followed by a column index
 * - the value associated with that position
 *
 * Any invalid index or failed extraction results in an exception.
 */
template <ScalarConcept Scalar>
Matrix readInsideSparseMatrix(std::ifstream& in, int rows, int cols) {
    if (rows < 0 || cols < 0) {
        throw std::runtime_error("Negative matrix dimensions");
    }

    int nnz = 0;
    if (!(in >> nnz)) {
        throw std::runtime_error("Cannot read number of non-zero entries in the sparse matrix");
    }
    if (nnz <= 0) {
        throw std::runtime_error("number of non-zero entries must be positive in a sparse matrix");
    }

    Matrix::Sparse<Scalar> sparse(rows, cols);
    sparse.reserve(nnz);

    for (int k = 0; k < nnz; ++k) {
        int r{};
        int c{};

        if (!(in >> r >> c)) {
            throw std::runtime_error("Error when trying to read indices in sparse matrix");
        }

        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::runtime_error("Sparse indices out of range");
        }

        if constexpr (is_complex_of_floating<Scalar>::value) {
            using Real = typename Scalar::value_type;
            Real re{};
            Real im{};
            if (!(in >> re >> im)) {
                throw std::runtime_error("Failed to read scalar entry in sparse matrix");
            }
            sparse.insert(r, c) = Scalar(re, im);
        } else {
            Scalar v{};
            if (!(in >> v)) {
                throw std::runtime_error("Failed to read scalar entry in sparse matrix");
            }
            sparse.insert(r, c) = v;
        }
    }
    
    sparse.makeCompressed();
    return Matrix(sparse);
}



/**
 * @brief Storage format used for matrix construction.
 */
enum class StorageType {
    Dense,
    Sparse
};

/**
 * @brief Converts a string into a StorageType.
 *
 * @param s text label
 * @return the corresponding StorageType
 *
 * Throws an exception when the label does not match any known format.
 */
inline StorageType translateStorageType(const std::string& s){
    if (s == "dense") return StorageType::Dense;
    if (s == "sparse") return StorageType::Sparse;
    throw std::runtime_error("Unknown storage type: " + s);
}


/**
 * @brief Reads a matrix from a text file.
 *
 * @tparam Scalar scalar type satisfying ScalarConcept
 * @param filename path to the file to open
 * @return the matrix, dense or sparse, created according to the file content wrapped inside the Matrix type 
 *
 * The file must begin with the storage label followed by the dimensions
 * The remaining content is delegated to the appropriate internal routine
 * If the file cannot be opened or its contents are malformed, an exception is thrown
 */
template <ScalarConcept Scalar>
Matrix readMatrixFromFile(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Impossible to open the file: " + filename);
    }

    std::string storageName;
    if (!(in >> storageName)) {
        throw std::runtime_error("Failed to read matrix storage type");
    }
    StorageType storageType = translateStorageType(storageName);

    int rows = 0;
    int cols = 0;
    if (!(in >> rows >> cols)) {
        throw std::runtime_error("Failed to read matrix dimensions");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Matrix dimensions must be positive");
    }

    switch (storageType) {
        case StorageType::Dense:
            return readInsideDenseMatrix<Scalar>(in, rows, cols);
        case StorageType::Sparse:
            return readInsideSparseMatrix<Scalar>(in, rows, cols);
    }

    throw std::runtime_error("Reached end of the function unexpectedly");
}

} // end namespace