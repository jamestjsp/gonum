# BLAS-to-LAPACK hot-path performance pass

## Scope and reproducibility

Required workflow: `go-optimisation`, together with `gonum-simd` and the scoped
LAPACK numerical-review gates. The user added `go-optimisation` during this pass;
subsequent acceptance uses its prebuilt-binary comparison runner with recorded
binary hashes, sample order, runtime settings, and matching benchmark selection.
No repository PGO profile or GOFLAGS override was present when checked.

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

Longer representative baseline profiles (three-second benchmark targets,
`GOMAXPROCS=1`) give the next priorities:

| Workload | Cumulative hotspot share | Implication |
| --- | --- | --- |
| LU factorization, n=256 | 30.6% non-transposed GEMM | Enable disjoint shared-storage panels first |
| Tall QR, 512x256 | 43.0% GEMM with transposed B | Separate transpose-aware kernel opportunity |
| Cholesky, n=256 | 25.0% SYRK; 23.2% TRSV | Symmetric updates and condition estimation both matter |
| LU solve, n=256, 16 RHS | 99.0% TRSM | Test blocked triangular solves next |

These are sampled CPU shares for specific inputs, not universal workload weights.

## Shared-panel SIMD results

The benchmark-only checkpoint is `fefd7949` (same implementation as `856a2859`);
the shared-panel candidate is `dee9b2d4`. The initial manual six-sample screen
used 100 ms for broad native/public cohorts and 150 ms for shared-panel kernels.
After the skill audit, both committed trees were rebuilt with Go 1.27.1 SIMD
and explicit `-pgo=off`. The skill runner alternated ten rounds at 400 ms for
LU/Cholesky and 300 ms for shared GEMM, recording binary hashes, raw per-process
output, runtime environment and order. Its results supersede the initial public
and kernel percentages below. Raw local evidence is in
`/tmp/gonum-lapack-hotpaths.NcL3gt/`; the directory is not a permanent artifact.

| Workload, one worker | Before | Active-row guard | Time change |
| --- | ---: | ---: | ---: |
| Dgetrf, n=128, reusable workspace | 276.9 us | 251.8 us | -9.1% |
| Dgetrf, n=256, reusable workspace | 1.681 ms | 1.454 ms | -13.5% |
| Dpotrf Upper, n=256 | 783.7 us | 692.3 us | -11.7% |
| Public LU, n=128, audited | 360.4 us | 339.9 us | -5.7% |
| Public LU, n=256, audited | 2.013 ms | 1.813 ms | -10.0% |
| Public Cholesky, n=256, audited | 1.237 ms | 1.182 ms | -4.5% |
| Public Cholesky, n=128, audited | 256.1 us | 265.3 us | +3.6% |

All tabulated differences have p<0.05 (native rows: six manual samples; public
rows: ten runner samples). The n=128 Cholesky slowdown repeated and is
reported as a tradeoff; that size does not execute the changed GEMM path, so a
causal explanation has not been established. Larger LU/Cholesky gains outweigh
this measured small-size cost in this cohort, not in every possible workload.

Audited shared-panel DGEMM/SGEMM runtimes decreased 27.6-61.1% at one worker
(ten rounds, all p<0.001). Four-worker LU-shaped n=256 decreased 23.3% for DGEMM
and 36.7% for SGEMM (ten rounds, p<0.001). One-worker cases allocate nothing;
the parallel cases retain about 2 KiB and 20 allocations per operation in both
revisions. The initial screen's outliers and overbroad zero-allocation claim
are superseded by these recorded runner results. The
generated kernel still emits ARM64 vector FMLA instructions. Only dispatch
geometry changed, not its arithmetic loop or existing bounds checks.

QR and most SVD/public solve cases were statistically neutral. The broad screen
suggested general-eigen gains but had large baseline outliers, so this pass does
not claim those as established improvements. Lower Cholesky remains unchanged
and slower than this Reference-LAPACK runtime at n=256: 1.668 ms versus 1.147 ms.
Upper Cholesky and LU kernels at n=256 are faster than the native reference on
these fixtures (0.692 versus 1.783 ms; 1.454 versus 2.207 ms respectively).
These comparisons say nothing about optimized Accelerate/OpenBLAS performance.

Runner directories: `audit-mat`, `audit-gemm`, and `audit-gemm-p4` under the
evidence directory. For the public API comparison, SHA-256 identifies the
baseline binary as `d719facb5ec154b65d988d72a18787f93d458679e600fc6788013a9f8b739068`
and candidate as `55b80a875faabc6982087900cb9d8c9a3eb9d4eb5242b8339067d2e817d19966`.
Runtime controls: GOMAXPROCS as labeled; GODEBUG, GOGC and GOMEMLIMIT unset.

## Dedicated skill audit

Three independent Sol reviewers checked measurement quality, numerical behavior,
and code generation/dispatch. The audit corrected parallel-allocation reporting,
added self-contained comparison metadata and legacy-provenance notes, and
strengthened shared-panel exceptional-value and native solve tests.

The uncommitted blocked-TRSM prototype failed a persistent finite-to-infinity
regression in both precisions for Upper/NoTrans. Backward panel traversal also
reverses Lower/Trans accumulation within panels. The revised candidate blocks
only Lower/NoTrans and Upper/Trans/ConjTrans, preserving the original backward
paths. The original Lower/Trans test fixture was nondiscriminating and was
corrected to isolate descending-versus-ascending order within one panel.
No prototype timing is accepted as production or consumer performance evidence.

Native solve gates now compare Gonum and Netlib solutions directly as well as
checking independent residuals. Selected cases cover n=128/129/192/256, RHS
widths 1/16/17/64, both transpose/triangle choices, and RHS scales 1e-200/1e200.
Exponent-normalized residuals avoid overflowing the tolerance scale. These are
bounded numerical gates, not a full LAPACK parity audit.

## Reproduce

Build each tree before timing, with matching benchmark files:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -tags netlib ./lapack/gonum -o lapack.test
GOMAXPROCS=1 ./mat.test -test.run '^$' -test.bench '^BenchmarkFactorization' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./blas.test -test.run '^$' -test.bench '^Benchmark[DS]gemmSharedBacking$' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./lapack.test -test.run '^$' -test.bench '^BenchmarkD(geqrf|getrf|potrf)Netlib$' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./mat.test -test.run '^$' -test.bench '^BenchmarkFactorization$/LU$/n=256$' -test.benchtime=3s -test.cpuprofile=lu.cpu
GOTOOLCHAIN=go1.27.1 go tool pprof -top mat.test lu.cpu
```

Alternate baseline/candidate order for at least six samples and compare files
with `benchstat`. Repeat selected kernels with `GOMAXPROCS=4`. Native tags need
the optional darwin/cgo Homebrew Reference-LAPACK installation; the `mat` and
BLAS manifests are runnable without it, including on AMD64. AMD64 does not gain
new automatic GEMM dispatch from this ARM64-validated change.

1. Establish persistent factorization/solve and BLAS3 cohorts; profile the current
   implementation and compare representative kernels with Reference-LAPACK.
2. Change a measured shared bottleneck; validate kernel boundaries, supported
   layouts, numerical behavior, and end-to-end consumers against the same base.
3. Run fallback and full-suite gates, publish measured results and remaining
   priorities, and verify each milestone on the remote branch.

Unresolved questions: none.
