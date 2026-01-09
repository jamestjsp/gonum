# LAPACK Additions Specification for Gonum

This document specifies the missing LAPACK routines needed to support control systems applications (SLICOT-style algorithms).

## Overview

**Goal**: Add routines for Sylvester/Lyapunov equations and generalized eigenvalue problems.

**Priority Order**:
1. **Critical** - Sylvester equation solvers (enables Lyapunov)
2. **High** - Generalized Schur decomposition (QZ algorithm)
3. **Medium** - Supporting routines for generalized eigenvalue problems

## File Structure

Each routine follows gonum's pattern:
```
lapack/gonum/dXXXXX.go      # Implementation
lapack/testlapack/dXXXXX.go # Tests
```

---

## Design Decisions

### General Philosophy
- **Match LAPACK behavior**: Prefer LAPACK-compatible behavior for numerical algorithms, scaling, and edge cases
- **Follow gonum patterns**: For API design, error handling, input validation, types, and code organization
- **Study existing routines**: Use Dgeev, Dhseqr as style references
- **Fork first**: Maintain in separate package initially, upstream to gonum when mature

### Error Handling
- **Input validation**: Panic on invalid inputs (negative dimensions, nil slices, invalid parameters) - follow gonum pattern
- **Numerical failures**: Follow gonum's existing pattern for similar routines (e.g., QZ non-convergence)
- **Singular Sylvester**: Return failure with diagnostic info following gonum's pattern for similar cases
- **Callback panics**: Follow gonum pattern for handling panics in user callbacks

### Numerical Choices
- **Scaling**: Simple scaling only in Dtrsyl (no iterative refinement), return scale factor for caller
- **Block detection**: Exact zero check for quasi-triangular block boundaries (a[i+1,i] == 0)
- **QZ shifts**: Match LAPACK heuristics exactly for compatibility
- **Balancing**: Powers of 2 scaling (exact arithmetic, matches LAPACK)
- **Block sizes**: Use gonum's Ilaenv infrastructure for runtime selection
- **LAPACK bugs**: Implement corrected behavior from latest LAPACK, not historical bugs

### API Design
- **Workspace queries**: LAPACK style - return optimal size in work[0], document minimum separately
- **Eigenvalue ties**: Undefined ordering (matches LAPACK), document this
- **Internal routines**: Export Dtgsy2 as public API (useful for advanced users)
- **Dtgsen modes**: Single unified function with ijob parameter (matches LAPACK)
- **n=0 handling**: Follow gonum's existing pattern
- **Infinite eigenvalues**: Pass raw values (beta=0) to SchurSelect callback, document clearly
- **In-place modification**: Document that inputs are overwritten, no defensive copies
- **Block swaps**: Leave blocks as-is after Dtgexc/Dtgex2 (caller standardizes if needed)
- **Return style**: Follow gonum's existing pattern for returns and bool arrays
- **Invalid transpose**: Follow gonum's input validation pattern
- **Types location**: Define new types (SchurSelect, etc.) in main lapack/lapack.go

### Testing
- **Reference outputs**: Pre-generate golden files using netlib LAPACK, store in repo (no CGo needed to run tests)
- **Golden precision**: Tolerance-based comparison (not bit-exact)
- **Test matrices**: Pathological cases (Wilkinson, Kahan) + random matrices + fuzzing
- **RNG seeds**: Fixed seeds for reproducibility
- **Workspace tests**: Test both lwork=-1 query path and execution path explicitly
- **Tolerances**: Follow gonum's existing LAPACK test tolerance patterns
- **Concurrency**: Follow gonum's pattern for concurrent safety
- **No CGo benchmarks**: Benchmark Go implementation standalone only

### Development Process
- **Implementation order**: Complete each routine fully (impl + tests + docs) before starting next
- **Debug checks**: Use `//go:build debug` tag for internal assertions during development
- **Dependency bugs**: Fix locally in fork, document, upstream later
- **Helper functions**: Follow gonum's pattern for shared utilities
- **Documentation**: Prose only (no Unicode math notation), standard godoc

---

## Priority 1: Sylvester Equation Solvers (Critical)

### 1.1 Dtrsyl - Triangular Sylvester Equation Solver

**File**: `lapack/gonum/dtrsyl.go`

**Purpose**: Solve the real Sylvester matrix equation:
- `op(A)*X + X*op(B) = scale*C` (isgn=+1)
- `op(A)*X - X*op(B) = scale*C` (isgn=-1)

where A and B are upper quasi-triangular (Schur form), op(A) = A or A^T.

**LAPACK Signature**:
```fortran
SUBROUTINE DTRSYL( TRANA, TRANB, ISGN, M, N, A, LDA, B, LDB, C, LDC, SCALE, INFO )
```

**Go Signature**:
```go
func (impl Implementation) Dtrsyl(
    trana, tranb blas.Transpose,
    isgn int,           // +1 or -1
    m, n int,
    a []float64, lda int,
    b []float64, ldb int,
    c []float64, ldc int,
) (scale float64, ok bool)
```

**Dependencies** (all exist in gonum):
- `Dlasy2` - solve 1x1 or 1x2 or 2x1 or 2x2 Sylvester equations
- `Dscal`, `Dgemm`, `Dgemv` (BLAS)
- `Dlange`, `Dlamch` (LAPACK utilities)

**Algorithm**:
1. Partition A and B into 1x1 and 2x2 diagonal blocks (quasi-triangular)
2. Use exact zero check for block boundary detection
3. Solve block-by-block using back-substitution
4. Use `Dlasy2` for small (1-2)x(1-2) systems
5. Handle scaling to avoid overflow (simple scaling, return scale factor)

**Test Cases**:
- Random quasi-triangular A, B with known solution
- Pathological matrices (Wilkinson, Kahan-style)
- Edge cases: m=0, n=0, 1x1 blocks only, 2x2 blocks
- Near-singular cases (shared eigenvalues) - verify failure with diagnostics
- Verify: ||op(A)*X + X*op(B) - scale*C|| / (||A||*||X|| + ||X||*||B|| + ||C||) < tol

---

### 1.2 Dgees - Schur Factorization (Moved from Priority 4)

**File**: `lapack/gonum/dgees.go`

**Note**: Moved to Priority 1 to enable better Dtrsyl testing and serve as implementation warmup.

**Go Signature**:
```go
func (impl Implementation) Dgees(
    jobvs lapack.SchurComp,
    sort lapack.Sort,
    selctg func(wr, wi float64) bool,
    n int,
    a []float64, lda int,
    wr, wi []float64,
    vs []float64, ldvs int,
    work []float64, lwork int,
    bwork []bool,
) (sdim int, ok bool)
```

**Dependencies** (all exist):
- `Dgebal`, `Dgehrd`, `Dorghr`, `Dhseqr`, `Dtrsen`, `Dgebak`

---

### 1.3 Dtgsyl - Generalized Sylvester Equation Solver

**File**: `lapack/gonum/dtgsyl.go`

**Purpose**: Solve the generalized Sylvester equations:
```
A*R - L*B = scale*C
D*R - L*E = scale*F
```
where (A,D), (B,E) are in generalized Schur form.

**Go Signature**:
```go
func (impl Implementation) Dtgsyl(
    trans blas.Transpose,
    ijob int,           // 0: solve only, 1-4: compute Dif estimates
    m, n int,
    a []float64, lda int,
    b []float64, ldb int,
    c []float64, ldc int,
    d []float64, ldd int,
    e []float64, lde int,
    f []float64, ldf int,
    work []float64, lwork int,
    iwork []int,
) (scale, dif float64, ok bool)
```

**Dependencies**:
- `Dtgsy2` (needs implementation, will be exported) - level 2 solver
- `Dscal`, `Dgemm`, `Dcopy`, `Daxpy` (BLAS)
- `Dlacpy`, `Dlaset`, `Dlange` (LAPACK utilities)

---

### 1.4 Dtgsy2 - Level 2 Generalized Sylvester Solver

**File**: `lapack/gonum/dtgsy2.go`

**Note**: Exported as public API for advanced users building custom solvers.

**Go Signature**:
```go
func (impl Implementation) Dtgsy2(
    trans blas.Transpose,
    ijob int,
    m, n int,
    a []float64, lda int,
    b []float64, ldb int,
    c []float64, ldc int,
    d []float64, ldd int,
    e []float64, lde int,
    f []float64, ldf int,
    rdsum, rdscal float64,
) (scale, rdsum2, rdscal2 float64, pq int, ok bool)
```

---

## Priority 2: Generalized Schur Decomposition (High)

### 2.1 Dgges - Generalized Schur Factorization

**File**: `lapack/gonum/dgges.go`

**Purpose**: Compute generalized Schur factorization of matrix pencil (A,B):
```
A = Q*S*Z^T,  B = Q*T*Z^T
```
where S is upper quasi-triangular, T is upper triangular, Q and Z are orthogonal.

**Go Signature**:
```go
func (impl Implementation) Dgges(
    jobvsl, jobvsr lapack.SchurComp,
    sort lapack.Sort,
    selctg SchurSelect,  // nil if sort == 'N'; receives raw values including beta=0
    n int,
    a []float64, lda int,
    b []float64, ldb int,
    alphar, alphai, beta []float64,
    vsl []float64, ldvsl int,
    vsr []float64, ldvsr int,
    work []float64, lwork int,
    bwork []bool,
) (sdim int, ok bool)
```

**Callback Documentation**: The selctg callback receives raw (alphar, alphai, beta) values. When beta=0, the eigenvalue is infinite. Caller must handle this case. Eigenvalue ties have undefined ordering.

**Dependencies**:
- `Dggbal` (needs implementation) - balance matrix pair
- `Dgghrd` (exists) - reduce to Hessenberg-triangular form
- `Dhgeqz` (needs implementation) - QZ iteration
- `Dtgsen` (needs implementation) - reorder Schur form
- `Dggbak` (needs implementation) - back-transform eigenvectors

**New Types** (add to `lapack/lapack.go`):
```go
// SchurSelect is a function type for eigenvalue selection in Dgges.
// alphar, alphai are the real and imaginary parts of the numerator.
// beta is the denominator. When beta=0, the eigenvalue is infinite.
type SchurSelect func(alphar, alphai, beta float64) bool

// SchurSort specifies eigenvalue sorting in Dgges.
type SchurSort byte

const (
    SortNone     SchurSort = 'N' // No sorting
    SortSelected SchurSort = 'S' // Sort selected eigenvalues to top-left
)
```

---

### 2.2 Dggbal - Balance Matrix Pair

**File**: `lapack/gonum/dggbal.go`

**Purpose**: Balance a pair of general real matrices (A,B) for generalized eigenvalue problem.

**Go Signature**:
```go
func (impl Implementation) Dggbal(
    job lapack.BalanceJob,
    n int,
    a []float64, lda int,
    b []float64, ldb int,
    lscale, rscale []float64,
    work []float64,
) (ilo, ihi int)
```

**Implementation Note**: Use powers of 2 for scaling (scale factors are 2^k) to ensure exact scaling arithmetic.

---

### 2.3 Dhgeqz - QZ Algorithm

**File**: `lapack/gonum/dhgeqz.go`

**Purpose**: Implement QZ iteration for Hessenberg-triangular matrix pair.

**Go Signature**:
```go
func (impl Implementation) Dhgeqz(
    job lapack.SchurJob,
    compq, compz lapack.SchurComp,
    n, ilo, ihi int,
    h []float64, ldh int,
    t []float64, ldt int,
    alphar, alphai, beta []float64,
    q []float64, ldq int,
    z []float64, ldz int,
    work []float64, lwork int,
) (ok bool)
```

**Algorithm**: Francis double-shift QZ iteration
- Match LAPACK's exceptional shift heuristics exactly for compatibility
- Reference: Moler & Stewart (1973), Watkins (2007)
- Use `//go:build debug` tag for internal assertions during development

---

### 2.4 Dggbak - Back-transform Eigenvectors

**File**: `lapack/gonum/dggbak.go`

**Purpose**: Back-transform eigenvectors after Dggbal.

**Go Signature**:
```go
func (impl Implementation) Dggbak(
    job lapack.BalanceJob,
    side blas.Side,
    n, ilo, ihi int,
    lscale, rscale []float64,
    m int,
    v []float64, ldv int,
)
```

---

## Priority 3: Supporting Routines (Medium)

### 3.1 Dtgsen - Reorder Generalized Schur Form

**File**: `lapack/gonum/dtgsen.go`

**Purpose**: Reorder generalized Schur decomposition so selected eigenvalue cluster appears in leading diagonal blocks.

**Go Signature**:
```go
func (impl Implementation) Dtgsen(
    ijob int,
    wantq, wantz bool,
    selected []bool,
    n int,
    a []float64, lda int,
    b []float64, ldb int,
    alphar, alphai, beta []float64,
    q []float64, ldq int,
    z []float64, ldz int,
    work []float64, lwork int,
    iwork []int, liwork int,
) (m int, pl, pr, dif float64, ok bool)
```

**Note**: Single unified function with ijob parameter (0-4) matching LAPACK, not split into separate functions.

**Dependencies**:
- `Dtgexc` (needs implementation) - swap adjacent blocks
- `Dlacn2` - estimate 1-norm (exists)

---

### 3.2 Dtgevc - Generalized Eigenvectors

**File**: `lapack/gonum/dtgevc.go`

**Purpose**: Compute eigenvectors of generalized Schur form.

**Go Signature**:
```go
func (impl Implementation) Dtgevc(
    side lapack.EVSide,
    howmny lapack.EVHowMany,
    selected []bool,
    n int,
    s []float64, lds int,
    p []float64, ldp int,
    vl []float64, ldvl int,
    vr []float64, ldvr int,
    mm int,
    work []float64,
) (m int, ok bool)
```

---

### 3.3 Dtgexc - Swap Diagonal Blocks in Generalized Schur Form

**File**: `lapack/gonum/dtgexc.go`

**Go Signature**:
```go
func (impl Implementation) Dtgexc(
    wantq, wantz bool,
    n int,
    a []float64, lda int,
    b []float64, ldb int,
    q []float64, ldq int,
    z []float64, ldz int,
    ifst, ilst int,
    work []float64, lwork int,
) (ifstOut, ilstOut int, ok bool)
```

**Note**: Leaves blocks as-is after swap (does not auto-split 2x2 blocks that could be 1x1). Caller standardizes if needed.

---

### 3.4 Dtgex2 - Swap Adjacent 1x1 or 2x2 Blocks

**File**: `lapack/gonum/dtgex2.go`

**Go Signature**:
```go
func (impl Implementation) Dtgex2(
    wantq, wantz bool,
    n int,
    a []float64, lda int,
    b []float64, ldb int,
    q []float64, ldq int,
    z []float64, ldz int,
    j1, n1, n2 int,
    work []float64, lwork int,
) (ok bool)
```

---

### 3.5 Dggev - Generalized Eigenvalues (Convenience)

**File**: `lapack/gonum/dggev.go`

**Purpose**: Compute eigenvalues and optionally eigenvectors of (A,B).

**Go Signature**:
```go
func (impl Implementation) Dggev(
    jobvl lapack.LeftEVJob,
    jobvr lapack.RightEVJob,
    n int,
    a []float64, lda int,
    b []float64, ldb int,
    alphar, alphai, beta []float64,
    vl []float64, ldvl int,
    vr []float64, ldvr int,
    work []float64, lwork int,
) (ok bool)
```

---

## Implementation Order

| Phase | Routine | Est. Complexity | Dependencies |
|-------|---------|-----------------|--------------|
| 1a | Dgees | Medium | All exist |
| 1b | Dtrsyl | Medium | Dlasy2 (exists) |
| 1c | Dtgsy2 | Medium | None |
| 1d | Dtgsyl | Medium | Dtgsy2 |
| 2a | Dggbal | Low | None |
| 2b | Dggbak | Low | None |
| 2c | Dtgex2 | Medium | Dlagv2, Dlartg (exist) |
| 2d | Dtgexc | Medium | Dtgex2 |
| 2e | Dhgeqz | **High** | Core QZ algorithm |
| 2f | Dtgsen | Medium | Dtgexc, Dlacn2 |
| 2g | Dgges | Medium | All above |
| 3a | Dtgevc | Medium | Dlaln2 (exists) |
| 3b | Dggev | Low | Dgges, Dtgevc |

---

## Testing Strategy

### Unit Tests
Each routine gets a `testlapack/dXXXXX.go` file with:
1. **Trivial cases**: n=0, n=1 (follow gonum pattern)
2. **Pathological matrices**: Wilkinson, Kahan, and other numerically difficult cases
3. **Random matrices**: various sizes (small, medium), fixed RNG seeds for reproducibility
4. **Fuzz testing**: Edge case discovery
5. **Workspace queries**: Explicit tests for lwork=-1 path
6. **Accuracy checks**: residual norms, orthogonality (follow gonum tolerance patterns)

### Golden File Testing
- Generate reference outputs using netlib LAPACK (via CGo)
- Store golden files in repo with tolerance-based comparison
- Tests run without CGo dependency
- Regenerate golden files when needed using development tools

### Test Verification Formulas
- Dtrsyl: ||op(A)*X + X*op(B) - scale*C|| / (||A||*||X|| + ||X||*||B|| + ||C||) < tol
- Dgees: ||A - V*S*V^T|| / (||A||*n*eps) < tol, ||V*V^T - I|| < tol
- Dgges: ||A - VSL*S*VSR^T|| / (||A||*n*eps) < tol

---

## Interface Additions (lapack/lapack.go)

```go
// SchurSelect is a function type for eigenvalue selection in Dgges.
// alphar, alphai are the real and imaginary parts of the numerator.
// beta is the denominator. When beta=0, the eigenvalue is infinite.
type SchurSelect func(alphar, alphai, beta float64) bool

// Add to Float64 interface:
type Float64 interface {
    // ... existing methods ...

    // Sylvester equation
    Dtrsyl(trana, tranb blas.Transpose, isgn, m, n int, a []float64, lda int,
           b []float64, ldb int, c []float64, ldc int) (scale float64, ok bool)

    // Level 2 generalized Sylvester (exported for advanced users)
    Dtgsy2(trans blas.Transpose, ijob, m, n int, a []float64, lda int,
           b []float64, ldb int, c []float64, ldc int, d []float64, ldd int,
           e []float64, lde int, f []float64, ldf int,
           rdsum, rdscal float64) (scale, rdsum2, rdscal2 float64, pq int, ok bool)

    // Generalized Sylvester
    Dtgsyl(trans blas.Transpose, ijob, m, n int, a []float64, lda int,
           b []float64, ldb int, c []float64, ldc int, d []float64, ldd int,
           e []float64, lde int, f []float64, ldf int,
           work []float64, lwork int, iwork []int) (scale, dif float64, ok bool)

    // Generalized eigenvalue
    Dgges(jobvsl, jobvsr SchurComp, sort SchurSort, selctg SchurSelect,
          n int, a []float64, lda int, b []float64, ldb int,
          alphar, alphai, beta []float64, vsl []float64, ldvsl int,
          vsr []float64, ldvsr int, work []float64, lwork int, bwork []bool) (sdim int, ok bool)

    // Standard Schur
    Dgees(jobvs SchurComp, sort SchurSort, selctg func(wr, wi float64) bool,
          n int, a []float64, lda int, wr, wi []float64,
          vs []float64, ldvs int, work []float64, lwork int, bwork []bool) (sdim int, ok bool)
}
```

---

## References

1. LAPACK Users' Guide (3rd ed.) - Anderson et al.
2. "Matrix Computations" (4th ed.) - Golub & Van Loan
3. "The QZ Algorithm" - Moler & Stewart, SIAM J. Numer. Anal. (1973)
4. netlib LAPACK source: https://github.com/Reference-LAPACK/lapack
5. gonum LAPACK: https://github.com/gonum/gonum/tree/master/lapack

---

## Notes for Contributors

1. **Follow gonum style**:
   - Error checking via `panic()` for invalid inputs
   - Follow existing patterns for numerical failure returns
   - Use `blas64.Implementation()` for BLAS calls
   - Document with complete godoc (prose only, no Unicode math)
   - Study Dgeev, Dhseqr as style references

2. **Machine constants**: Use existing `dlamchX` constants from gonum

3. **Workspace queries**: Support `lwork == -1` for optimal workspace size (return in work[0], document minimum)

4. **No CGo**: All implementations must be pure Go

5. **Block sizes**: Use Ilaenv for runtime block size selection

6. **Debug builds**: Use `//go:build debug` tag for internal assertions

7. **Development workflow**: Complete each routine (impl + tests + docs) before starting next

8. **Dependency bugs**: If found in existing gonum code, fix locally, document, upstream later
