## Transpose

The symbol T (written as a superscript T, e.g., \(A^T\)) means the transpose of a matrix or vector. Transpose flips rows and columns: the element that was in row i, column j of A becomes row j, column i of \(A^T\).

### Definition and notation

- **Matrix**: If \( A \in R^{m×n} \) then \( A^T \in R^{n×m} \) and \( (A^T)_{j,i} = A_{i,j} \).  
- **Vector**: A column vector \( x ∈ R^n \) has transpose \( x^T \) which is a row vector \( 1×n \). The row vector \( x^T \) is the same elements in one row instead of one column.

### Key algebraic rules (useful formulas)

- \( (A^T)^T = A \).  
- \( (A + B)^T = A^T + B^T \).  
- \( (cA)^T = c A^T \) for scalar \( c \).  
- \( (A B)^T = B^T A^T \) (reverse order).  
- \( (A^{-1})^T = (A^T)^{-1} \) if \( A \) is invertible.  
- \( x^T y = y^T x = \) scalar (dot product) when \( x,y ∈ R^n \).  
- For matrices of compatible size, \( \text{tr}(A B) = \text{tr}(B A) \) and uses transposes in proofs.

### Intuition

- Transpose changes perspective: rows become columns and columns become rows.  
- Use transpose to express dot products as matrix products:  \( x·y = x^T y \).  
- It converts a linear map expressed by rows into an equivalent map expressed by columns and vice versa; this is why it reverses multiplication order.

### Small numeric example

\( A = [[1,2,3],[4,5,6]] \) is \( 2×3 \)  
\( A^T = [[1,4],[2,5],[3,6]] \) is \( 3×2 \)  

Vector \( x = [7,8,9]^T \) is

\[
\begin{bmatrix}
7 \\
8 \\
9
\end{bmatrix}
\] (3×1)

---

## Matrix and Vector Cheat Sheet for ML

### Basic objects and notation
- **Scalar**: single number, denoted a, b ∈ R.  
- **Vector**: column by default, an ordered list of numbers, \( x ∈ R^n \) written \( x = [x_1, x_2, …, x_n]^T \). Row vectors are \( x^T \).  
- **Matrix**: rectangular array \( A ∈ R^{m×n} \) has m rows and n columns. Entry in row i, column j is \( A_{ij} \).  
- **Shapes**: always keep track as (rows × columns). Valid operations require compatible shapes.

### Vector operations
- **Addition / subtraction**  
  \( x + y \) is defined only when \( x,y \in R^n \). Add elementwise: \( (x + y)_i = x_i + y_i \).

- **Scalar multiplication**  
  \( a \cdot x \) multiplies every element: \( (a \cdot x)_i = a x_i \). Intuition: scale the length/direction of \( x \).

- **Dot product (inner product)**  
  \( x \cdot y = x^T y = \sum_{i=1}^n x_i y_i \) (scalar).  
  Intuition: measures alignment; if normalized, equals cosine similarity times lengths.

  - if \( x,y \) are orthogonal (perpendicular), then \( x \cdot y = 0 \).
  - if the dot product is positive, the vectors point roughly in the same direction; if negative, they point in opposite directions.
  - if the dot product is 1 (or -1), the vectors are perfectly aligned (or anti-aligned).
  - The value of the dot product can be interpreted as a measure of similarity between the two vectors. The larger the dot product, the more similar the vectors are in terms of direction.

- **Norms**  
  - Euclidean (L2): \( ||x||_2 = \sqrt{\sum x_i^2} \).  
  - L1: \( ||x||_1 = \sum |x_i| \).  
  Norms measure vector size; used in regularization (||β||_2^2, ||β||_1).

- **Outer product**  
  \( x y^T \in R^{n×n} \) with \( (x y^T)_{ij} = x_i y_j \).  
  Intuition: builds a matrix from two vectors; rank‑1 matrix that projects along \( x \) and scales by \( y \).

### Matrix operations
- **Addition / subtraction**  
  \( A + B \) defined when \( A,B \in R^{m×n} \); add elementwise.

- **Scalar multiplication**  
  \( a A \) multiplies every entry of \( A \) by \( a \).

- **Matrix multiplication (A B)**  
  If \( A \in R^{m×k} \) and \( B \in R^{k×n} \) then \( C = A B \in R^{m×n} \) with  
  \( C_{ij} = \sum_{r=1}^k A_{ir} B_{rj} \).  
  Intuition: each entry is a dot product between row \( i \) of \( A \) and column \( j \) of \( B \); combine linear effects across the shared dimension \( k \).

- **Matrix-vector multiplication**  
  If \( A \in R^{m×n} \) and \( x \in R^n \) then \( y = A x \in R^m \) with  
  \( y_i = \sum_{j=1}^n A_{ij} x_j \).  
  Intuition: each entry of \( y \) is a weighted sum of the columns of \( A \) weighted by \( x \).

- **Identity matrix**  
  \( I_n \in R^{n×n} \) has 1 on diagonal, 0 off-diagonal. Property: \( I A = A I = A \).

- **Determinant**  
  For square \( A \in R^{n×n} \), \( \text{det}(A) \) is a scalar measuring volume scaling and invertibility. If \( \text{det}(A) = 0 \), \( A \) is singular (noninvertible).
  - Intuition: absolute value of determinant gives scaling factor of volume when applying linear transformation \( A \).
  - Sign of determinant indicates orientation (positive preserves, negative reverses).
  - Zero determinant means the transformation squashes space into a lower dimension (e.g., a plane or line), losing information.

- **Inverse**  
  For square \( A \in R^{n×n} \), \( A^{-1} \) exists if \( \text{det}(A) \neq 0 \) and satisfies \( A^{-1} A = I \). Not all matrices are invertible.

### Important algebraic rules and properties
- Associativity: \( (A B) C = A (B C) \) when shapes match.  
- Distributivity: \( A(B + C) = A B + A C \) and \( (A + B)C = AC + BC \).  
- Scalar commutes: \( a (A B) = (a A) B = A (a B) \).  
- Transpose rules: \( (A^T)^T = A \); \( (A B)^T = B^T A^T \); \( (A + B)^T = A^T + B^T \).  
- Inverse rules: \( (A B)^{-1} = B^{-1} A^{-1} \) if \( A \) and \( B \) invertible.  
- Noncommutative: generally \( A B \neq B A \).  
- Trace and cyclic property: \( \text{tr}(A B) = \text{tr}(B A) \) when dimensions allow; useful in derivations and matrix calculus.

### Determinant, rank, and positive definiteness
- **Determinant** \( \text{det}(A) \) scalar for square \( A \); \( \text{det}=0 \Leftrightarrow A \) singular (noninvertible).  
- **Rank** \( \text{rank}(A) \) = dimension of column (or row) space. Rank deficiency indicates collinear features and causes \( (X^T X) \) to be singular in linear regression.  
- **Positive definite (PD)**: A symmetric matrix \( M \) is PD if \( x^T M x > 0 \) for all nonzero \( x \). Covariance and Hessian matrices often need PD property. PD ⇒ invertible.

### Useful matrix decompositions and eigen stuff
- **Eigen-decomposition (square, symmetric)**: \( A = Q Λ Q^T \) with orthogonal \( Q \) and diagonal \( Λ \) (eigenvalues). Interprets \( A \) as scaling along orthogonal directions.  
- **Singular Value Decomposition (SVD)**: \( A = U Σ V^T \) with \( U∈R^{m×m} \), \( Σ∈R^{m×n} \) diagonal nonnegative singular values, \( V∈R^{n×n} \). SVD is robust and used for PCA, low-rank approximations, pseudo-inverse.  
- **Pseudo-inverse (Moore-Penrose)**: \( A^+ = V Σ^+ U^T \) from SVD; used when \( A \) not full rank to compute least-squares solutions.