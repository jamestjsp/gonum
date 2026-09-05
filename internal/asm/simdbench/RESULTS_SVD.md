# Native SVD rotation follow-up

## Scope and method

Base: `150606198bafc4b32c207753acdfa6b43b2c288d`.
Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2.
Primary toolchain: Go 1.27.1, `GOEXPERIMENT=simd`, `GOMAXPROCS=1`.
Default comparison: installed Go 1.26.4 without the experiment.
No AMD64 measurements or cross-compilation were performed in this pass.

Both source revisions used the same persistent benchmark harness. Native runs
were serialized, with six samples per case and alternating base/change order.
Final rotation samples used 50ms and SVD samples used 100ms benchmark durations;
earlier screening runs used 100-200ms. Results are interpreted with benchstat,
not a single fastest sample. The desktop host was not CPU-isolated.

The fresh base profile of 256-by-256 thin-vector Dgesvd attributed 70.96% of
flat samples to Dlasr and 75.72% cumulatively to Dbdsqr. The dominant branch
was Right/Variable/Forward, accessing columns of row-major matrices.

## Accepted measurements

Median Dgesvd time with thin singular vectors, before and after this change:

| Toolchain | Matrix | Base | Changed | Time change | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| Go 1.27.1 SIMD | 64 × 64 | 932.7 µs | 927.5 µs | no significant change | 0.485 |
| Go 1.27.1 SIMD | 128 × 128 | 6.479 ms | 6.120 ms | -5.53% | 0.002 |
| Go 1.27.1 SIMD | 256 × 256 | 114.60 ms | 55.13 ms | -51.90% | 0.002 |
| Go 1.27.1 SIMD | 256 × 128 | 12.03 ms | 11.67 ms | -3.01% | 0.041 |
| Go 1.27.1 SIMD | 128 × 256 | 10.027 ms | 9.653 ms | -3.74% | 0.026 |
| Go 1.26.4 default | 64 × 64 | 1.042 ms | 1.042 ms | no significant change | 0.699 |
| Go 1.26.4 default | 256 × 256 | 120.48 ms | 60.90 ms | -49.46% | 0.002 |

Each comparison has six samples per revision. Benchstat's 95% interval for
the SIMD 256-by-256 result was ±4% on base and ±6% on changed; the default
comparison was ±1% and ±8%. Smaller default-toolchain gains did not reach
significance. Values-only SVD cases were unchanged within noise. No measured
SVD case significantly regressed; thin-vector cases reported zero allocations
per operation. These findings apply to this host and benchmark matrix, not
every SVD workload.

Dense Right/Variable/Forward Dlasr on compact 256-by-256 matrices improved
from 303.66 µs to 47.50 µs (-84.36%); Backward improved from 307.84 µs to
47.13 µs (-84.69%). Both have p=0.002. At padded stride 259, both directions
remained statistically unchanged, around 46 µs. The 63-by-65 fallback cases
also remained statistically unchanged.

The microbenchmark tradeoffs are real: some blocked boundary cases regressed
2-8%, and sparse Backward 64-by-256 at stride 259 regressed from 1.567 µs to
1.762 µs (+12.41%, p=0.002). Identity-only calls commonly added about 2-3 ns.
These costs are retained in the benchmark coverage and should constrain future
dispatch tuning; this is not a claim that every rotation layout is faster.

The separate same-input native comparison produced:

| Matrix | Changed Gonum | Reference-LAPACK | Lower elapsed time |
| --- | ---: | ---: | --- |
| 32 × 24 | 105.80 µs | 73.04 µs | Reference-LAPACK |
| 24 × 32 | 101.00 µs | 80.17 µs | Reference-LAPACK |
| 96 × 64 | 1.327 ms | 1.088 ms | Reference-LAPACK |
| 64 × 96 | 1.210 ms | 1.059 ms | Reference-LAPACK |
| 256 × 256 | 54.64 ms | 89.61 ms | Gonum |

All five paired differences have p=0.002, n=6 per implementation. Gonum takes
about 39% less time in the large case, but Reference-LAPACK takes 12-31% less
time in the smaller cases. This harness uses different inputs from the existing
Dgesvd benchmark above; compare implementations within each table only.
The reference backend and timing boundaries are documented below.

## Changes

- Corrected a pre-existing Left/Top/Backward duplicated loop that applied each
  rotation once per matrix column. This is a correctness fix, not a valid
  before/after speed comparison; it is not the profiled SVD branch.
- Repaired the shared Dlasr test's reversed copies, bottom-pivot reference
  indices, and missing right-side transpose. Nonzero structured and seeded
  random fixtures now exercise all 12 side/pivot/direction combinations.
- Right/Variable rotations use row blocks only when both dimensions are at
  least 64 and every rotation is active. Each row retains its original rotation
  order. Nominal blocks have 32 rows; a final remainder shorter than 16 rows is
  merged into its preceding block. Small and identity-containing cases retain
  the original traversal. An out-of-line helper keeps the fallback code small,
  and incrementing the matrix index by its stride avoids a per-row address
  multiplication.

This is architecture-neutral scalar cache blocking, not a new SIMD kernel.
Generated ARM64 code uses scalar fused arithmetic and retains bounds checks.
The existing SIMD BLAS configuration is otherwise unchanged.

A rejected row-at-a-time candidate was approximately five times slower on
large padded rotation cases, consistent with serialized dependent arithmetic.
Cutoff and tail sweeps also rejected a 33-row crossover and a one-row final
block. Compact and padded strides must both be measured: the original compact
256-by-256 rotation was much slower than its padded counterpart.

## Independent reference and correctness

Source review pin: [Reference-LAPACK v3.12.1,
6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca](https://github.com/Reference-LAPACK/lapack/tree/6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca),
`SRC/dlasr.f`; the SVD bridge also calls Dbdsqr and Dgesvd.
The installed Homebrew formula is `lapack 3.12.1_1`, but its runtime `ILAVER`
reports **3.12.0**. These identifiers are recorded separately. Its linked BLAS
is the Reference-LAPACK keg's `libblas`, not OpenBLAS or Accelerate.

All 12 Dlasr computational branches were source-reviewed and independently
exercised, including mixed identities and shapes around both the dispatch
and tail-merge boundaries. Row-major leading-dimension validation is a
deliberate adaptation of Netlib's column-major contract. The exact duplicated
rotation regression failed before the repair. Padding, non-finite identity
skips, and signed zero retain exact checks.

The larger oracle cases exposed cancellation-sensitive failures in a newly
added fixed output-relative comparison, including an unchanged branch. The
new test instead bounds absolute error by the input vector norm and rotation
count, using `2*gamma_(3*r)` for the two evaluations, and explicitly rejects
non-finite results. Existing numerical tolerances were not relaxed.

Persistent Dbdsqr tests cover upper/lower matrices, optional transformed
matrices, and values-only cases. Dgesvd tests cover square/tall/wide shapes,
five vector-job combinations, tiny/ordinary/huge scales, and a rank-deficient
case. Comparisons check singular values, reconstruction, and orthogonality;
they do not require identical singular vectors.

This is not a full transitive SVD parity audit. Overwrite and mixed All/Store
jobs, minimum-workspace paths, difficult non-convergence, and extreme or
clustered bidiagonal cases remain outside the new differential suite.

## Reproduction

The optional bridge uses the existing `netlib && darwin && cgo` configuration
and Homebrew LAPACK paths. It does not add a production CGo dependency.

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -tags netlib ./lapack/gonum \
  -run 'Test(Dlasr|DlasrNetlibDifferential|DbdsqrNetlib.*|DgesvdNetlib.*|NetlibRuntimeVersion)$' -count=3

GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd GOMAXPROCS=1 \
  go test -tags netlib ./lapack/gonum -run '^$' \
  -bench '^BenchmarkDgesvdNetlibKernels$' -benchmem -count=6 > native-svd.txt
benchstat -col /implementation native-svd.txt
```

The paired native benchmark computes the same logical random matrix and thin
vectors with Dgesvd on both sides. Each implementation queries and reuses its
own preferred workspace. Input restoration is included on both sides; layout
conversion and workspace allocation are excluded. The native CGo call is
included, and Go allocation counters do not account for native allocations.
This compares Reference-LAPACK plus reference BLAS, not an optimized vendor
library or the different Dgesdd algorithm.

Validation passed: full default and SIMD suites; affected LAPACK/mat `safe`
and `noasm` suites; LAPACK `bounds`; focused Dlasr race tests; the persistent
Netlib suite repeated three times; formatting, imports, copyright and diff
checks. A broader Dlasr/Dbdsqr/Dgesvd race run also passed during integration.

## Remaining work

Measure the shared traversal change on AMD64 before claiming a speedup there.
Compare optimized OpenBLAS or Accelerate separately, and profile the remaining
SVD cost before another kernel pass. Evaluate Dgesdd as a separate algorithmic
project; do not attribute a Dgesvd-versus-Dgesdd difference to SIMD.
