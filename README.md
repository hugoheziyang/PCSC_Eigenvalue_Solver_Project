# PCSC Eigenvalue Solver Project

---

## Project overview

This C++ project implements a solver to find all eigenvalues of a given matrix A. The methods used include 
the **power method**, **shifted inverse power method**, and the **QR method**. For a detailed description of each
methodology, one can refer to the respective webpages [Power iteration](https://en.wikipedia.org/wiki/Power_iteration), 
[Inverse iteration](https://en.wikipedia.org/wiki/Inverse_iteration), [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm).

---

## Authors

- Clément Froidevaux
- Ziyang He  

---

## 1. Compilation of program

After cloning the repository, run the following command to populate the ```eigen``` directory:
```bash
git submodule update --init eigen
```

To compile the whole repository, execute the following in command line at the root of repository:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

After initial compilation, if certain files are edited (e.g. ```main.cpp```), to recompile the changes, execute from the root of repository:
```bash
cd build
make -j
```

To launch the ```main``` program, simply execute from the root of repository:
```bash
cd build
./main
```

---

## 2. Documentation and repository structure

In depth documentation is done using ```doxygen``` software. To create the documentation, run the command in the root of repository:
```bash
doxygen Doxyfile
```
To view the documentation, open ```docs/html/index.html``` in your favourite browser. For example:
```bash
google-chrome docs/html/index.html
```

After compiling the repository and generating the documentation, the repository structure at the first level should look like (comments preceded by ```//```):
```
.
├── build                   // Directory of executables
├── CMakeLists.txt          
├── data                    // Directory of matrix inputs in .txt format
├── docs                    // Directory of documentation in html and latex format
├── Doxyfile                // Documentation executable
├── eigen                   // Directory of eigen library git submodule
├── main.cpp
├── README.md
├── src                     // Directory of eigensolver source code of project
└── test                    // Directory of tests for functions in src/
```

---

## 3. Program logic and typical execution

To use the eigensolver, the user can choose to create an ```EigSol::Matrix``` object by any of the three following ways:
- Input a ```.txt``` file with specific layout.
- Create an ```eigen``` matrix object and wrapping it with ```EigSol::Matrix```.
- Create a ```std::vector``` matrix object and wrapping it with ```EigSol::Matrix```.
The three ways are described in the following subsections, as well as the application of eigensolver functions onto the ```EigSol::Matrix``` objects.


### 3.1 Input a ```.txt``` file with specific layout.
For dense matrices, the ```.txt``` file should be written in the following format (replace <.> with desired numbers):
```
dense
<number_of_rows> <number_of_columns>
<dense_matrix_representation>
```
An example of representing a real dense matrix: 
```math
\begin{pmatrix}
    1 & 3 & 1 \\
    0 & 2 & 3 \\
    0 & 0 & 5 
\end{pmatrix}
```
is given by replacing ```<dense_matrix_representation>``` by the following:
```
1 3 1
0 2 3
0 0 5
```
An example of representing a complex dense matrix: 
```math
\begin{pmatrix}
    1+3i & 3+5i & 1+4i \\
    0 & 2+4i & 3+2i \\
    0 & 0 & 5-i 
\end{pmatrix}
```
is given by replacing ```<dense_matrix_representation>``` by the following:
```
1 3   3 5   1 4 
0 0   2 4   3 2
0 0   0 0   5 -1
```

For sparse matrices, the ```.txt``` file should be written in the following format (replace <.> with desired numbers):
```
sparse
<number_of_rows> <number_of_columns>
<number_of_nonzero_values>
<sparse_matrix_representation>
```
An example of representing a real sparse matrix: 
```math
\begin{pmatrix}
    1 & 3 & 1 \\
    0 & 2 & 3 \\
    0 & 0 & 5 
\end{pmatrix}
```
is given by replacing ```<number_of_nonzero_values>``` by 6 and ```<sparse_matrix_representation>``` by the following:
```
0 0 1
0 1 3
0 2 1 
1 1 2
1 2 3
2 2 5
```
where for each row, the syntax is:
```
<row_index> <column_index> <value>
```
An example of representing a complex sparse matrix: 
```math
\begin{pmatrix}
    1+3i & 3+5i & 1+4i \\
    0 & 2+4i & 3+2i \\
    0 & 0 & 5-i 
\end{pmatrix}
```
is given by replacing ```<number_of_nonzero_values>``` by 6 and ```<sparse_matrix_representation>``` by the following:
```
0 0 1 3
0 1 3 5
0 2 1 4
1 1 2 4
1 2 3 2
2 2 5 -1
```
where for each row, the syntax is:
```
<row_index> <column_index> <real_part> <imaginary_part>
```

To create an ```EigSol::Matrix A``` containing a ```double``` typed matrix (either dense or sparse) from ```A.txt```, the user can implement the following C++ code:

```cpp
#include "src/reader/file_matrix_reader.hpp"
#include "src/matrix/matrix.hpp"

const std::string fileA = "A.txt";
EigSol::Matrix A = EigSol::readMatrixFromFile<double>(fileA)
```

### 3.2 Create an ```eigen``` or ```std::vector``` matrix object and wrapping with ```EigSol::Matrix```
The user can directly wrap a ```eigen``` or ```std::vector``` matrix object by using the constructor of ```EigSol::Matrix```. For example:

```cpp
#include "src/matrix/matrix.hpp"
#include <Eigen/Dense>
#include <vector>

Eigen::Matrix<double, 2, 2> A;
A << 1.0, 2.0, 
     3.0, 4.0;
EigSol::Matrix M1(A);   // M1 is constructed from A

std::vector<double> B = {1.0, 2.0, 3.0, 4.0};     // Input 2×2 matrix
EigSol::Matrix M2(B, 2, 2);     // M2 is constructed from B

// Shape and values of M1 and M2 are the same
```


### 3.3 Eigensolver functions
We show an example usage of power method, shifted inverse power method, and QR method. Assume a ```double``` typed matrix wrapped in ```EigSol::Matrix A``` has already been created. 

```cpp
#include "src/option/solver_option.hpp"
#include "src/option/shifted_solver_option.hpp"
#include "src/power_method/power_method.hpp"
#include "src/power_method/shifted_inverse_power_solver.hpp"
#include "src/qr_method/qr_eigenvalues.hpp"

// --- Power method ---
EigSol::SolverOptions opts;
opts.maxIterations = 1000;
opts.tolerance = 1e-10;

auto powerResult = EigSol::powerMethod<double>(A, opts)    

// --- Shifted inverse power method ---
EigSol::ShiftedSolverOptions<double> shiftedOpts;
shiftedOpts.shift = 3.1;
shiftedOpts.maxIterations = 1000;
shiftedOpts.tolerance = 1e-12;

auto shiftInvPowResult = EigSol::shiftedInversePowerMethod<double>(A, shiftedOpts);   

// --- QR method ---
EigSol::SolverOptions qrOpts;
qrOpts.maxIterations = 1000;
qrOpts.tolerance = 1e-10;

auto qrResult = EigSol::qr_eigenvalues<double>(A, qrOpts);
```

In the code implementation above, ```powerResult``` and ```shiftInvPowResult``` both have attributes:
- eigenvalue (single eigenvalue)
- eigenvector (single eigenvector)
- iterations (number of iterations used)
- converged (true or false)

The QR method result ```qrResult``` has attributes:
- eigenvalue (list of all eigenvalues)
- iterations (number of iterations used)
- converged (true or false)


---

## 4. Use of AI

AI chatbots were used to write most of ```src/core/types.hpp``` and all the tests in the ```test/``` directory. However, editing was done to these files after AI generation. AI chatbots were also used to tidy up and clean ```doxygen``` comments for better presentation in documentation.

---

## 5. Validating tests

All tests are written with GoogleTest and are contained in the ```test/``` directory. They can be built and run with:

```bash
cd build
cmake --build . --target matrix_wrapper_test solve_shifted_test power_method_test shifted_inverse_power_method_test qr_algorithms_test 
ctest
```


- ```matrix_wrapper_test.cpp``` tests correct construction of the Matrix wrapper from dense or sparse Eigen matrices or ```std::vector``` with proper type reporting casting rules and value consistency.

- ```solve_shifted_test.cpp``` tests solving the shifted system (A − λI)x = b for dense or sparse real or complex matrices and checks that invalid dimensions or scalar mismatches raise the expected errors.

- ```power_method_test.cpp``` tests convergence of the power method to the dominant eigenpair on dense or sparse matrices and verifies correct handling of nonconvergence and invalid shapes.

- ```shifted_inverse_power_method_test.cpp``` tests convergence of the shifted inverse power method toward the eigenvalue nearest the chosen shift on dense or sparse matrices with proper handling of nonsquare or zero-size inputs.

- ```qr_algorithms_test.cpp``` includes:
  - Hessenberg reduction tests verifying correct upper-Hessenberg structure preservation of eigenvalues and rejection of nonsquare matrices.
  - QR decomposition tests checking that QR reconstructs the matrix with orthogonal or unitary Q and upper-triangular R and that empty inputs are rejected.
  - QR iteration tests ensuring correct eigenvalue recovery on real or complex matrices and rejection of nonsquare matrices.

---

## 6. Limitations and known issues

- The solver is fully templated on the scalar type so the code lives header files which increases compile times and makes separate compilation impossible. 

- If the user whats to change the scalar type, he will need to recompile the whole project.

- Error messages could be more informative.

- The project relies on Eigen for core numerical functions such as LU factorization because of time limits.

- The implementation prioritises a user-friendly interface and a structure that makes adding new features easy rather than focusing on high-performance numerical routines.
