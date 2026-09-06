# BLAS-to-LAPACK hot-path performance pass

## Scope and reproducibility

Starting implementation: `856a28593860b8ecaac50537134852c1dc8a7a6d` on
`codex/arm64-simd-blas`. Common `mat` APIs are the workload proxy; this is not
usage telemetry or a claim that every LAPACK routine is optimized.

Native host: Apple M1 Pro, darwin/arm64. SIMD measurements use Go 1.27.1,
`GOEXPERIMENT=simd`, initially `GOMAXPROCS=1`. Default-path checks use Go 1.26.4.
Prebuild test binaries before timing; run baseline and candidate serially and
interleave repeated samples. Record allocation counts and distinguish public
API costs from reusable-workspace LAPACK kernels. No AMD64 performance is
inferred from this host.

Native comparisons use Homebrew Reference-LAPACK, not OpenBLAS or Accelerate.
Source review is pinned to v3.12.1 commit
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`. The installed formula is 3.12.1_1,
but ILAVER reports 3.12.0. Source identity and runtime identity are distinct.

## Consumer call ledger

| Public workload | LAPACK path | Shared lower-level work |
| --- | --- | --- |
| QR / tall solve | Dgeqrf, Dtrcon; Dormqr, Dtrtrs | Dlarf/Dlarfb, GEMV/GER, GEMM/TRMM/TRSM |
| LQ / wide solve | Dgelqf, Dorglq, Dtrcon; Dormlq, Dtrtrs | Reflectors and GEMM/TRMM/TRSM |
| LU / square solve | Dgetrf, Dgecon; Dgetrs | Panel GER, trailing GEMM, TRSM |
| Cholesky / SPD solve | Dpotrf, Dpocon; Dpotrs | SYRK, GEMM, TRSM; triangular vector solves |
| SVD | Dgesvd: QR/LQ or direct Dgebrd, then Dbdsqr | Reflectors, GEMM, GEMV/GER, rotations |
| Symmetric eigen | Dsyev: Dsytrd, then Dsterf or Dorgtr/Dsteqr | Symmetric matrix/vector updates, reflectors, rotations |
| General eigen | Dgeev: Dgebal, Dgehrd, Dhseqr, Dtrevc3 | Hessenberg/Schur updates, GEMM/GEMV, triangular solves |

Factorization APIs query preferred workspace. QR, LU, and Cholesky additionally
estimate condition numbers; these costs belong in public API measurements.
SVD solves use matrix multiplication and singular-value scaling rather than a
LAPACK solve routine. `Dense.Solve` chooses LU, QR, or LQ by matrix shape.

This ledger is a prioritization aid, not a full transitive numerical-parity audit.
Changes require persistent regression tests, affected conformance tests, and
independent reconstruction/residual or native checks appropriate to the path.

## Milestones

Persistent entry points:

- `mat`: `BenchmarkFactorization`, `BenchmarkFactorizationSolve` cover sizes
  32/128/256, shape and vector-job variants, and one/16 right-hand sides. PCG
  fixtures and one untimed warm-up make steady-state reuse explicit.
- `blas/gonum`: `BenchmarkDtrsmSizes`, `BenchmarkDsyrkSizes` cover small and
  block-boundary dimensions; `BenchmarkDgemmSharedBacking` and
  `BenchmarkSgemmSharedBacking` isolate LU/upper-Cholesky panel layouts.
- `lapack/gonum`, tag `netlib`: `BenchmarkDgeqrfNetlib`,
  `BenchmarkDgetrfNetlib`, `BenchmarkDpotrfNetlib`, and the existing
  `BenchmarkDgesvdNetlibKernels` separate reusable-workspace kernel costs from
  public API overhead. Input resets are included equally; layout conversion and
  workspace queries are excluded. Native allocation counts cover Go, not C.

The first measured target is GEMM eligibility for shared-backing panels. A
three-second CPU profile of public LU at 256x256 attributed 30.6% of samples
cumulatively to `dgemmSerialNotNot`; the existing SIMD guard rejected entire
overlapping slice suffixes even when their active matrix entries were disjoint.
The proposed repair checks only active row intervals, keeps genuine-overlap
rejection, and reuses the existing SIMD kernel without altering LAPACK algorithms.
Strided AXPY and triangular solves remain substantial independent costs.

1. Establish persistent factorization/solve and BLAS3 cohorts; profile the current
   implementation and compare representative kernels with Reference-LAPACK.
2. Change a measured shared bottleneck; validate kernel boundaries, supported
   layouts, numerical behavior, and end-to-end consumers against the same base.
3. Run fallback and full-suite gates, publish measured results and remaining
   priorities, and verify each milestone on the remote branch.

Unresolved questions: none.
