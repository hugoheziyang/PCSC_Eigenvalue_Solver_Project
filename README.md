### PCSC Eigenvalue Solver Project

---

## Project overview
This C++ project implements a solver to find all eigenvalues of a given matrix A. The methods used include 
the **power method**, **shifted inverse power method**, and the **QR method**. For a detailed description of each
methodology, one can refer to the respective webpages [Power iteration](https://en.wikipedia.org/wiki/Power_iteration), 
[Inverse iteration](https://en.wikipedia.org/wiki/Inverse_iteration), [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm).

---

## Authors
- Clement Froidevaux
- Ziyang He  

---

## 5. Validating tests

All tests are written with GoogleTest and are contained in the ```test/``` directory. They can be built and run with:

```bash
cd build
cmake --build . --target matrix_wrapper_test solve_shifted_test qr_algorithms_test
ctest
```

The following subsections describes the purpose of each Google Test included in the project.

# 5.1 Tests for Matrix class wrapper in ```matrix_wrapper_test.cpp```

Test: ConstructFromEigenDense
- Verifies that wrapping a dense Eigen matrix correctly marks the Matrix as dense.
- Ensures scalar_type() matches the matrix scalar type.
- Ensures values inside the wrapper match the original Eigen matrix.

Test: ConstructFromEigenSparse
- Verifies correct wrapping of a sparse Eigen matrix.
- Ensures Matrix is marked as sparse and scalar_type() is correct.
- Confirms non-zero entries survive round-trip through the wrapper.

Test: ConstructFromStdVector
- Tests Matrix construction from std::vector (row-major interpretation).
- Ensures elements appear in the correct positions after wrapping.

Test: CastThrowsOnWrongType
- Ensures M.cast<T>() throws std::bad_cast when T is not the stored type.
- Ensures correct cast does not throw.

Test: TypeQueries
- Checks that type() returns the stored Eigen type.
- Checks that scalar_type() returns the correct scalar type.


# 5.2 Tests for solving linear system (A − λI)x = b in ```solve_shifted_test.cpp```

Test: DenseIdentity
- Uses A = I, so A − λI = (1 − λ)I.
- For λ = 2 ⇒ matrix becomes −I ⇒ x = −b.
- Tests correctness of dense branch + residual accuracy.

Test: DenseGeneral2x2
- Solves an arbitrary 2×2 shifted system.
- Compares solve_shifted() with Eigen’s PartialPivLU to ensure exact match.
- Validates numerical stability.

Test: SparseIdentity
- A = I in sparse form.
- Tests the sparse branch using Eigen::SparseLU.
- Confirms solution is −b for λ = 2.

Test: DenseComplex2x2
- Ensures complex-valued dense systems solve correctly.
- Compares against Eigen’s LU solver.

Test: ThrowsOnNonSquareDense
- Confirms solve_shifted() rejects non-square matrices.

Test: ThrowsOnNonSquareSparse
- Same as above but for sparse matrices.

Test: ThrowsOnSizeMismatchDense
- Ensures vector dimension mismatches trigger an exception.

Test: ThrowsOnScalarTypeMismatch
- Confirms the solver rejects calls where template scalar and actual matrix scalar differ.


# 5.3 Tests for Hessenberg reduction in ```qr_algorithms_test.cpp```

Test: Real3x3ProducesUpperHessenberg
- Ensures to_hessenberg_dense produces correct upper-Hessenberg structure.
- Verifies wrapper-based version matches dense version exactly.

Test: Complex3x3ProducesUpperHessenberg
- Same as above, but with complex matrices.

Test: ThrowsOnNonSquare
- Ensures non-square matrices are rejected.

Test: EigenvaluesPreservedReal
- Confirms the Hessenberg transformation is similarity-preserving:
  eigenvalues(H) = eigenvalues(A).
- Validates algorithm correctness using Eigen::EigenSolver.


# 5.4 Tests for QR decomposition in ```qr_algorithms_test.cpp```

Test: RealRectangular3x2
- Checks QR factorization satisfies:
    * A ≈ Q R
    * QᵀQ ≈ I
    * R is upper triangular.
- Confirms wrapper version (qr_decompose) matches the dense version.

Test: ComplexSquare2x2
- Same as above for complex matrices:
    * A ≈ QR
    * Q* Q ≈ I (unitarity)

Test: ThrowsOnEmpty
- Ensures qr_decompose_dense throws for empty (0×0) matrices.


# 5.5 Tests for QR iteration eigenvalues in ```qr_algorithms_test.cpp```

Test: Real2x2KnownEigenvalues
- Uses a symmetric matrix with known eigenvalues λ = 3, 1.
- Ensures QR iteration converges and returns expected eigenvalues.
- Confirms wrapper version matches dense version.

Test: Complex2x2SameEigenvalues
- Same test as above but stored as complex<double>.
- Ensures eigenvalues are still correctly recovered.

Test: ThrowsOnNonSquare
- Ensures eigenvalue solver rejects non-square matrices.


---

## 6. Limitations and known issues

# 6.1 Limitations of the matrix wrapper

- No copying or moving is allowed.
    The Matrix class intentionally disables:
        * default construction
        * copy construction / assignment
        * move construction / assignment
    This prevents accidental aliasing, but also means:
        * Matrix A = B; is NOT allowed
        * std::vector<Matrix> may not be usable in some patterns

- No resizing or in-place modification through the wrapper.
    The stored Eigen object can be accessed via cast<T>(), but:
        * The wrapper itself does not provide resizing utilities.
        * Modifying the underlying matrix is allowed but unsafe if done inconsistently.

- Sparse matrices must match Eigen::SparseMatrix<Scalar> EXACTLY.
    If a matrix is stored with different Options or StorageIndex,
    the wrapper canonicalizes it — but cast<> will only succeed for:
        Eigen::SparseMatrix<Scalar>   (column-major, int index)

- No runtime type conversion.
    If the user tries to read a double matrix as a complex matrix, or vice-versa,
    the wrapper will throw std::bad_cast. No automatic conversion exists.

- from_custom(T) is minimal.
    It stores arbitrary user types but:
        * They are treated as dense-like.
        * scalar_type() becomes nullptr.
        * Many algorithms (QR, Hessenberg, shifted solve) cannot operate on them.


# 6.2 Limitations of the ```solve_shifted``` method for linear systems

- Only solves square systems.
    Both dense and sparse branches require:
        A.rows() == A.cols()
    Otherwise an exception is thrown.

- Only solves linear systems of the form (A − λI)x = b.
    Arbitrary Ax = b or (A − B)x problems are not supported.

- Sparse solver uses Eigen::SparseLU.
    SparseLU:
        * may fail for nearly singular or badly conditioned matrices,
        * may perform poorly without reordering strategies (AMD, COLAMD),
        * does not support pivoting as robustly as dense LU.

- No fallback for rank-deficient matrices.
    If (A − λI) is singular or nearly singular:
        * Dense solver may return NaNs.
        * Sparse solver may fail at factorization.

- No automatic handling of complex shifts in real matrices.
    If Scalar = double but λ is complex, the user must manually promote to complex.

- No multi-right-hand-side support.
    solve_shifted() solves only vector b, not a full matrix B.


# 6.3 Limitations of the Hessenburg reduction

- Supports **dense matrices only**.
    Sparse Hessenberg reduction is not implemented.

- No accumulation of similarity transformations.
    The algorithm produces H only.
    Q such that H = Qᵀ A Q is not returned or tracked.

- Numerically stable but not optimal.
    This is a basic Householder Hessenberg routine:
        * No blocking (unlike LAPACK),
        * No explicit T-matrices,
        * Performance degrades for large matrices.

- Complex sign definition uses phase = x(0)/|x(0)|.
    Works correctly, but:
        * Can behave differently from LAPACK's choice.
        * Slightly affects stability for small imaginary parts.

- No deflation or zero-detection logic.
    Even if subdiagonal elements are ~0, algorithm does not exploit it.


# 6.4 Limitations of the QR decomposition

- Supports **dense matrices only**.
    Sparse QR is not implemented.

- Unblocked Householder QR.
    Performance is significantly slower than Eigen’s built-in QR or LAPACK.

- No column pivoting.
    For rank-deficient or nearly rank-deficient matrices:
        * R may be unstable,
        * Q*R may deviate from A.

- Q is always square (m×m).
    Even if A is m×n (m>n), Q is full square.
    No "thin QR" (economy QR) is provided.

- R is returned in full size (m×n).
    Superfluous rows below diagonal remain zero but are present.

- No check that R’s below-diagonal entries are numerically ~0.
    Returned R may contain tiny numerical noise.


# 6.5 Limitations of the QR iteration to find eigenvalues

- **Unshifted** QR iteration.
    Convergence can be extremely slow for:
        * non-symmetric matrices,
        * matrices with clustered eigenvalues,
        * defective matrices.

- No implicit shifts, Wilkinson shifts, or Rayleigh quotient shifts.
    Modern QR eigenvalue solvers rely on shifting for speed.
    Unshifted QR is mostly pedagogical.

- No deflation.
    Even when H(i,i−1) becomes numerically zero:
        * the algorithm treats matrix as full-size,
        * slowing convergence dramatically.

- No eigenvector computation.
    Only eigenvalues are returned.

- Complex arithmetic is supported but:
        * real matrices are not automatically promoted,
        * complex-conjugate pairs are not handled explicitly.

- Sensitive to tolerance settings.
    If tolerance is too small, iterations may not converge.
    If too large, convergence may be reported prematurely.

- Iteration count is fixed by opts.maxIterations.
    No adaptive iteration stopping.

---

