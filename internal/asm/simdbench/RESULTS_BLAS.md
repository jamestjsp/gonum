# Production BLAS SIMD performance pass

Measured 2026-09-05 on Apple M1 Pro, macOS 26.6.2 (25G83), darwin/arm64,
Go 1.27.1, `GOEXPERIMENT=simd`, `GODEBUG=simd=128`. The independent baseline
is `4cbe23600778fdb3b89f8ddfccf5b50e10c494f8`, including the previous AMD64
patch import. The measured implementation is
`d09a2b7bb77af78f0ad695d875e7fa4ebcd5495f` on `codex/arm64-simd-blas`
(GEMV/norm milestone `27ce2051`, GEMM milestone `d09a2b7b`). These are native
ARM64 measurements, not AMD64 predictions.

## Implementation and dispatch

- Portable Dgemm/Sgemm kernels retain a four-row, four-vector output tile in
  registers across the inner dimension. Single-vector and scalar column tails
  use the same accumulation strategy; incomplete rows use existing helpers.
  This avoids repeatedly loading/storing C and reuses B loads and broadcasts.
  The float32 source is generated from float64 by the existing Gonum generator.
- Production GEMM selection is ARM64-only with the SIMD experiment enabled.
  B must be untransposed; A may be transposed. There must be at least four rows,
  four inner elements, and four vectors of columns (8 doubles or 16 singles on
  this host). C must not overlap A or B. Existing parallel cutoffs are unchanged.
- Float64 GEMV uses shared portable candidates for disjoint unit-stride inputs
  with both dimensions at least eight. Transposed GEMV additionally requires
  at least 32 rows or at most 16 columns. Short-wide shapes retain the fallback.
  Four-vector unrolling improves the candidate's residual-row path without
  changing its multiply/add order. Ger was not promoted: its measured crossover
  did not justify production selection.
- The strided transposed GEMV fallback uses indexed, incrementally advanced
  matrix rows from 32 columns onward. Short rows retain the original AXPY loop.
  Value-range and unconditional indexed rewrites were rejected: fewer bounds
  checks did not compensate for severe short-row regressions on this CPU.
- Float64 norms and distances use compensated portable sums of squares from
  length 32. Zero-leading or extreme first values select the scaled recurrence
  immediately; unsafe sums detected later retry that recurrence. Tiny calls,
  NaNs, infinities, overflow/underflow, and zero-increment behavior are covered.
- Prefix candidates peel their first lane step, removing a redundant loop
  condition without changing the existing grouped scan order. Invariant AXPY
  broadcasts are hoisted. Neither change promotes AMD64 production dispatch.

Older Go releases, unsupported targets, `safe`, `noasm`, and GCCGo retain their
non-SIMD production paths. AMD64 assembly is still the production baseline;
the candidate comparison harness remains available there. These changes do
not make an experimental Go API GA or establish a portable performance cutoff.

## Measurement method

Prebuilt test binaries were run sequentially, with baseline/change order
reversed on alternate rounds. Each reported before/after comparison uses six
samples and `benchstat`. Public BLAS uses 75 ms per sample; GEMM shape/worker
and LAPACK consumer comparisons use 50 ms. `GOMAXPROCS=1` is the default;
explicit GEMM worker sub-benchmarks also set `GOMAXPROCS=4`. Compilation,
tests, and other agents' timing jobs were stopped before measurement.

Both checkouts contain identical benchmark fixtures. In particular, Dgemv's
repeated-call benchmark uses beta=1 instead of beta=3 so its output stays
finite. SVD and bidiagonal benchmarks restore their input on each iteration,
reuse queried workspace, and include the input copy in timing. SVD's vector
case computes thin U and VT. Fixtures use deterministic ordinary dense inputs,
not every possible matrix distribution or conditioning.

## Public BLAS results

Times are medians; negative percentages mean less time. Every changed timing
in the following tables has `p=0.002`, six samples per revision. B-transposed
Dgemm controls were statistically unchanged (`p=0.818`).

| Routine and shape | Baseline | Changed | Time change |
|---|---:|---:|---:|
| Dgemm NN, 10³ | 572.8 ns | 297.7 ns | -48.0% |
| Dgemm NN, 100³ | 217.3 µs | 145.6 µs | -33.0% |
| Dgemm NN, 1000³ | 190.1 ms | 147.5 ms | -22.4% |
| Dgemm TN, 100³ | 217.4 µs | 145.6 µs | -33.0% |
| Dgemv N, 10×10, unit increments | 72.98 ns | 57.47 ns | -21.3% |
| Dgemv T, 10×10, unit increments | 63.98 ns | 48.48 ns | -24.2% |
| Dgemv N, 100×100, unit increments | 5.577 µs | 1.934 µs | -65.3% |
| Dgemv N, 1000×1000, unit increments | 551.2 µs | 181.9 µs | -67.0% |
| Dgemv T, 1000×1000, unit increments | 211.7 µs | 194.3 µs | -8.2% |
| Dgemv T, 1000×10, unit increments | 5.316 µs | 2.758 µs | -48.1% |
| Dgemv T, 100×100, increments 2/3 | 5.639 µs | 4.479 µs | -20.6% |
| Dgemv T, 1000×1000, increments 2/3 | 514.2 µs | 400.3 µs | -22.2% |
| Dgemv T, 1000×10, increments 2/3 | 6.238 µs | 6.902 µs | +10.6% |
| Dnrm2, n=1000, unit increment | 2.456 µs | 781.0 ns | -68.2% |
| Dnrm2, n=100000, unit increment | 250.49 µs | 76.74 µs | -69.4% |
| Dnrm2, n=100000, increment 5 | 250.0 µs | 124.4 µs | -50.3% |

The short-row strided regression is unresolved, not a claimed improvement.
Its inner instruction count, loads, checks and call/spill behavior match the
baseline; code placement and register assignment differ. A frontend or operand
scheduling explanation remains a hypothesis. A separate f64 fixture measured
only +3% for that shape, reinforcing that call context and binary layout matter.
Tiny strided public GEMV and norms add roughly 0.7–2 ns. Internal length-three
strided norms add about 1.5 ns (27%); the production gate avoids the larger cost
of entering the compensated vector kernel at these lengths.

| GEMM dispatch, GOMAXPROCS=4 | Baseline | Changed | Time change |
|---|---:|---:|---:|
| Dgemm, 60³ | 52.39 µs | 33.58 µs | -35.9% |
| Dgemm, 100³ | 163.7 µs | 106.6 µs | -34.9% |
| Dgemm, 160³ | 332.5 µs | 219.3 µs | -34.0% |
| Sgemm, 60³ | 32.18 µs | 19.64 µs | -39.0% |
| Sgemm, 100³ | 114.21 µs | 61.78 µs | -45.9% |
| Sgemm, 160³ | 220.7 µs | 127.9 µs | -42.1% |

Both precisions also improved in every measured 1000×100×10, 100×1000×10,
and 128×128×8 dispatch case at `GOMAXPROCS=1` and `4` (7.7–36.0% less time).
Direct neighboring-column tests at n=56,59,60,61,64 rejected an early tail
implementation and confirmed the final full-column approach. With
`GOMAXPROCS=4`, the forced blocked helper is about 8% slower than serial at 60³
(a single output tile), but wins
at 100³ and 160³. Skinny shapes still deserve separate policy calibration:
forced parallel Sgemm at 128×128×8 is about 10% slower than its serial helper.
Forced helpers omit public validation/beta setup, so these are not identical
end-to-end policy comparisons. No parallel threshold was changed here.

## Consumer results

| Shape | Dgebrd time change | SVD values-only, baseline → changed | SVD thin vectors, baseline → changed |
|---|---:|---:|---:|
| 16×16 | -7.1% | 17.17 → 16.54 µs (-3.7%) | 39.03 → 37.06 µs (-5.1%) |
| 64×64 | -28.6% | 336.8 → 285.0 µs (-15.4%) | 1.0231 → 0.9398 ms (-8.1%) |
| 128×128 | -28.5% | 1.991 → 1.602 ms (-19.5%) | 7.183 → 6.410 ms (-10.8%) |
| 256×256 | -3.0% | 19.18 → 18.67 ms (-2.7%) | 116.4 → 114.7 ms (-1.5%) |
| 512×64 | -23.9% | 1.850 → 1.672 ms (-9.6%) | 4.321 → 3.867 ms (-10.5%) |
| 64×512 | -28.2% | 1.4511 → 0.9917 ms (-31.7%) | 3.445 → 2.576 ms (-25.2%) |
| 256×128 | -25.5% | 4.229 → 3.607 ms (-14.7%) | 13.24 → 12.04 ms (-9.1%) |
| 128×256 | -27.4% | 3.584 → 2.549 ms (-28.9%) | 11.696 → 9.993 ms (-14.6%) |

Allocation counts did not increase. Single-worker public BLAS, Dgebrd, and
thin-vector SVD measured zero allocations; values-only SVD retains two small
allocations (40 bytes). Parallel GEMM with `GOMAXPROCS=4` retains its existing task
allocations (10–66 per call for these shapes); the SIMD tiles themselves do
not allocate.

## Candidate and cutoff checks

The final prefix comparison uses a fresh baseline build at the same base commit,
75 ms samples, and the unchanged candidate harness. CumSum takes 40.3% less
time at n=31 and 42.8% less at n=4096 (4.463 → 2.554 µs); CumProd takes 16.2%
and 12.0% less (3.115 → 2.742 µs at n=4096). All four have `p=0.002`, n=6,
zero allocations. This resolves the prefix timing regression recorded in the
[previous import report](RESULTS_IMPORT.md), without claiming a new scan order.

At the norm cutoff, n=32 improves 32.7% contiguous and about 5.8% with increments
2/17; n=33 also improves. At n=4096 the gains are 67.2% contiguous and about 47%
strided. Zero-leading length-4096 inputs remain effectively unchanged; length
32 adds about 1–1.5 ns. These cutoff samples use 50 ms and six interleaved rounds.

The strided GEMV boundary n=31 stays on the old loop; n=32/33 improves about
19–22% at both 10 and 1000 rows. Same-binary transposed SIMD dispatch at
15×16, 31×16, 32×16, 32×17, and 32×255 improves 5.7–36.3%. Excluded 15×17
and 31×17 are statistically unchanged; excluded 31×255 still adds 3.4% in
that fixture. These are host-specific measured tradeoffs, not universal wins.

## Numerical and performance guardrails

An uncompensated norm promotion initially failed the existing Dgesvd test for
a 300-by-150, very-large-magnitude matrix when combined with GEMV promotion.
The independent baseline passed, and disabling either promotion removed the
failure. No SVD tolerance was relaxed. Compensating summation error, and product
error where SIMD MulAdd is fused, repaired the regression. The method follows
[TwoProductFMA and Dot2](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
Portable MulAdd need not be fused on every backend: fixture-level 1-ULP checks
are not a claim of universally correctly rounded norms.

Persistent tests cover a 256-bit norm oracle, cancellation and exponent
extremes, vector/tile/dispatch boundaries, all GEMM column remainders, transpose
and beta combinations, padding, overlapping slices, positive/negative/zero
increments, zero coefficients with NaN/Inf data, and prefix carry/alias order.
An existing zero-beta transposed GEMV bug also surfaced: the scalar path
cleared beyond the logical output length. It now clears only the n outputs,
with internal and public BLAS regressions.

Generated M1 code uses vector FMA for GEMM and norm product-error recovery.
The full GEMM tile and norm arithmetic helpers have no hot-loop vector spills;
norm helpers inline. Bounds/address bookkeeping still exists, so this is not
a claim that the compiler generates an ideal hand-written microkernel.
Scalar GEMM tails remain in their declared precision, including float32 FMA.

Validation passed: full Go 1.27.1 SIMD suite; full supported Go 1.26.4 suite;
affected internal-assembly, BLAS, LAPACK and mat tests under SIMD emulation,
`safe`, and `noasm`; internal-assembly and BLAS race/bounds tests; exact float32
regeneration; formatting, import policy, copyright, and diff checks. No new
cross-compilation or native AMD64 timing was performed in this pass.

## What the Go development branches taught us

The [upstream investigation](UPSTREAM.md) records exact revisions and open
compiler changes, including master and Gerrit as well as dev.simd. The
[experimental SVE SGEMM](https://go-review.googlesource.com/c/go/+/827812)
provided the useful register-tiling and broadcast-reuse pattern; no SVE API or
capability is assumed on the M1. Current loop-invariant-code-motion work also
supports explicitly hoisting invariant broadcasts today.

Recheck VEX spill/move encoding, FMA accumulation forms, NEON broadcasts, float
reductions, and missing portable widening/permutation operations with future
Go releases. Rebenchmark native AMD64 assembly versus candidates before
changing its dispatch. Open changes and accepted experiments are not a GA
schedule or proof that the compiler will choose every profitable algorithm.

## Remaining SVD bottleneck

A separate baseline CPU profile of 256-by-256 SVD with thin vectors attributed
about 71% of flat samples to Dlasr and 76% cumulative samples to Dbdsqr; Dgemv
and Dgemm accounted for about 12% and 9% cumulative respectively. These are
sampled profile proportions, not predicted speedups. BLAS improvements help,
but rotation application is the next substantial SVD target, especially
strided column rotations and reuse across successive rotations.

Read-only inspection also found a pre-existing duplicated column loop in
Dlasr's left/top/backward case and a reversed fixture-copy direction in its
shared test. That case is not the variable-pivot path in the profile. Both
require a separate LAPACK correctness repair and independent reference tests;
neither was silently changed as part of this BLAS pass.

## Reproduce

Build the baseline in a separate worktree, applying only the identical
benchmark changes from this pass: `blas/testblas/level2bench.go`, the two
LAPACK benchmark helpers and their wrappers, and the f64 norm/strided-GEMV
benchmark files. Keep the baseline implementation unchanged. Build each once:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./lapack/gonum -o lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./internal/asm/f64 -o f64.test
```

Alternate the two binaries for at least six rounds with identical flags:

```sh
GOMAXPROCS=1 GODEBUG=simd=128 ./blas.test -test.run '^$' \
  -test.bench '^Benchmark(Dgemm|Dgemv|Dnrm2)' -test.benchtime=75ms -test.benchmem
GOMAXPROCS=1 GODEBUG=simd=128 ./lapack.test -test.run '^$' \
  -test.bench '^Benchmark(Dgesvd|Dgebrd)$' -test.benchtime=50ms -test.benchmem
benchstat baseline.txt changed.txt
```

For AMD64, use the native comparison procedure in [README.md](README.md),
record CPU/vector width and affinity, and retain the default assembly run.
Do not interpret ARM64 `current`-versus-candidate results as ASM comparisons:
`current` now includes the selected production SIMD kernels on ARM64.
