# Bottom-up BLAS optimization

## Scope and evidence

This pass starts at the production BLAS Level 1 AXPY leaf, then checks BLAS
triangular solves, LAPACK solves and public `mat` consumers. It does not change
LAPACK algorithms. Required workflow: `go-optimisation` and `gonum-simd`, with
dedicated Sol numerical/harness and code-generation reviews.

Starting implementation: `3eb28d78306f776f4d2496e81db9b6f53026c938`.
The clean benchmark checkpoint is `dd91d903edd614c0506fad8174e47dfcbba49b44`,
which adds only persistent tests and short-RHS benchmark cases. Baseline binaries
are built from its detached clean worktree. Host: Apple M1 Pro, darwin/arm64;
Go 1.27.1 with `GOEXPERIMENT=simd`, `-pgo=off`, `GOMAXPROCS=1`. No repository
PGO profile or GOFLAGS override was present. Default tests use installed Go 1.26.4.
AMD64 production assembly and opt-out implementations are outside the change.

Go's [release history](https://go.dev/doc/devel/release) and the installed
Go 1.27.1 sources were refreshed before editing. Portable SIMD remains an
[experiment](https://github.com/golang/go/issues/78902); this pass uses the
existing native ARM64 leaf and adds no new experimental API dependency.

A separate baseline profile of public LU SolveTo at n=256 with one RHS puts
50.16% of sampled CPU time cumulatively in f64.AxpyUnitary, including 12.13%
in its overlap check. DTRSM calls this leaf once per coefficient even when
the slice contains just one value. The old leaf cannot inline (compiler cost
417 versus budget 80) and enters alias checks and vector setup before its
one-element scalar tail.

## Rejected leaf candidates

The first candidate separated the original vector body into an unexported helper
and handles only `len(x)==1` in the exported wrapper, in both precisions.
All other lengths keep the original arithmetic, bounds and overlap behavior.
An initial below-vector-width loop wrapper cost 83 and did not inline;
the one-element wrapper cost 77 and did inline into BLAS callers.
DTRSM and leaf benchmark code inspection showed direct scalar fused arithmetic and stores
for that branch, with the native helper retained for other lengths.

Ten interleaved rounds at 150 ms rejected that wrapper: float64 n=3/4/8/16
regressed 8.96/18.93/8.34/5.85%, and float32 n=7/8/16 regressed
5.17/15.99/6.04% (all p<0.001). The one-element leaf benchmark remained
about 3.42 ns in both precisions, statistically inconclusive. This repeated
update fixture has a loop-carried data dependency; it measures latency, not
independent-call throughput. Inlining was confirmed, so the larger-input
regressions cannot be attributed to a second nested function call.

A separate in-function singleton return and a sub-vector threshold folded into
the existing fallback were also screened (three rounds at 100 ms). Neither
consistently avoided short-vector costs. Screens are diagnostic, not accepted
performance claims. All AXPY production changes were reverted. The recorded
leaf comparisons remain useful rejection evidence, not promoted optimizations.

## Single-RHS BLAS specialization

The replacement candidate handles one RHS once at the DTRSM/STRSM entry point, keeping
the sequential scalar work inside the solve instead of dispatching AXPY for
every coefficient. This leaves AXPY and all multi-RHS arithmetic unchanged.
It preserves the existing transpose-specific update order, zero-coefficient
skip, alpha timing and reciprocal multiplication. Calling TRSV directly would
not preserve those details: its NoTrans path sums products before subtraction
and divides by the diagonal rather than multiplying by its reciprocal.

The scalar running value stays in a local variable. NoTrans rows apply alpha,
perform the original ordered updates, multiply by the reciprocal, then store
once. Transposed rows apply the reciprocal, update the remaining rows with the
pre-alpha value, then apply alpha and store. This removes both repeated AXPY
calls and conservative reload/store traffic without regrouping sums. Active
A/B entries must be disjoint, as in standard TRSM use; logically disjoint
regions sharing a backing allocation are explicitly tested.

Dispatch remains limited to the existing experimental ARM64 BLAS configuration,
left-sided solves with exactly one RHS, after ordinary validation and alpha-zero
handling. Multi-RHS and AMD64/default/opt-out arithmetic are unchanged. The
helper itself uses ordinary Go scalar operations, not new SIMD instructions;
the improvement addresses composition overhead exposed by the SIMD backend.

Persistent D/S solve tests compare to the old two-RHS path and check all real
transpose/triangle combinations, Unit/NonUnit, alpha=0/1/-0.75, stride/padding,
extreme scales, zero coefficients with NaN/Inf sources, backward-order overflow,
and shared backing with B in A row padding. Residual checks use existing precision
tolerances; native DGETRS/DPOTRS gates provide another independent reference.

Candidate implementation: `c284b8ac7f5e12543f4764a4345e6aede1f65d7a`.
The final BLAS, LAPACK and mat binaries were built at this commit with PGO off.
Additional one-RHS native extreme-scale test cases were added afterward; they
do not change production or benchmark source. Full standard/SIMD suites,
safe/noasm/bounds BLAS/LAPACK/mat suites, focused race and emulation checks,
and strengthened native solve tests pass. Float32 generation is synchronized.
Machine-code inspection shows scalar fused updates and no AXPY, mallocgc or
makeslice calls in the specialized helpers. Existing bounds checks remain.

## Benchmark and numerical contracts

`BenchmarkAxpyUnitaryShort` calls the production operation directly at lengths
0/1/2/3/4/7/8/15/16/17/64/256/4096. Exact-binary nonzero fixtures and alternating
alpha=+/-0.25 keep the input lifecycle bounded. Timing includes one call and
the alpha sign update per iteration. A b.N loop permits normal inlining; a
result sink after the timer retains the result. On this toolchain B.Loop uses
KeepAlive transformations, so the older blanket inlining-ban explanation does
not apply. Timing is in ns/op; SetBytes describes the logical x-vector size,
not total memory traffic.

Persistent tests cover ordinary and guarded short lengths, exact alias,
directional overlap where supported, immediate and partial-update panic cases,
and NaN/Inf/signed zero. They do not impose scalar overlap or bounds-check
contracts on AMD64 assembly. Leaf tests passed before and after the rejected
wrapper and remain valid with the original implementation restored.

Acceptance uses the skill runner with prebuilt binaries, ten interleaved rounds,
matching harnesses and recorded binary hashes/runtime controls. No builds,
tests or profiles overlap timing. A three-round public solve screen is diagnostic
only: it showed one-RHS gains and a small multi-RHS cost requiring confirmation.
Local evidence is under `/tmp/gonum-bottom-up.In6f6t`; it is temporary, not a
permanent artifact. Persistent benchmark entry points support reproduction.

## Accepted BLAS measurements

Ten interleaved rounds, 100 ms per benchmark, GOMAXPROCS=1. All 24 single-RHS
cases improve with p<0.001; each row below is the range across U/N, U/T, L/N
and L/T, not an aggregate throughput score.

| Routine | m | Time reduction, one RHS |
| --- | ---: | ---: |
| DTRSM | 32 | 72.57–84.52% |
| DTRSM | 128 | 71.40–84.77% |
| DTRSM | 256 | 70.68–85.17% |
| STRSM | 32 | 73.90–85.88% |
| STRSM | 128 | 72.48–86.13% |
| STRSM | 256 | 72.47–86.16% |

For example, DTRSM m=256 Upper/NoTrans falls from 132.60 to 38.88 us;
Lower/Trans falls from 127.05 to 18.84 us. All 24 two-/sixteen-RHS controls
have no significant regression. One STRSM m=128, two-RHS Upper/Trans control
improves 1.69% (p=0.023); the remaining controls are statistically inconclusive.
Every measured BLAS case remains at zero B/op and zero allocs/op.

BLAS SHA-256 identities recorded by the runner:

```text
baseline  32bcaebf0f1c11640c85e8b23ac0ef9cc85071e88d2d7de965b94d4e1665a388
candidate 754ffc59b871cdcca17631eacb5860bebf9a23157d716d328c0782316470197a
```

## Public solve consumers

Ten interleaved rounds at 150 ms, GOMAXPROCS=1. These are steady-state
`mat.SolveTo` operations on existing factorizations, not factorization time.
QR/SVD fixtures are tall, LQ is wide, and LU/Cholesky are square.
All gains below have p<0.001, n=10 per version.

| Public solve, one RHS | Size | Baseline us/op | Candidate us/op | Change |
| --- | ---: | ---: | ---: | ---: |
| QR | 128 | 439.2 | 412.4 | -6.09% |
| QR | 256 | 1734 | 1640 | -5.41% |
| LQ | 128 | 505.2 | 478.9 | -5.22% |
| LQ | 256 | 2140 | 2041 | -4.61% |
| LU | 128 | 69.68 | 18.30 | -73.73% |
| LU | 256 | 274.20 | 76.80 | -71.99% |
| Cholesky | 128 | 69.82 | 16.32 | -76.62% |
| Cholesky | 256 | 274.35 | 62.64 | -77.17% |

Both one-RHS SVD controls and all ten sixteen-RHS controls are statistically
inconclusive. No public solve has a significant regression. Allocations are
unchanged: LU/Cholesky remain allocation-free; QR/LQ retain two allocations,
and SVD seven. The SVD path does not use this triangular specialization.

## Reference-LAPACK solve comparison

Ten interleaved rounds at 100 ms, GOMAXPROCS=1, using Homebrew Reference-LAPACK
3.12.1_1 (ILAVER reports 3.12.0), not Accelerate or OpenBLAS. The existing
native harness times matching RHS copies and solves; factors, layout conversion
and pivot conversion are outside timing. LP64 native pivots are prepared once.
Native correctness tests compare solutions and scaled residuals, including
one-RHS n=192 cases at RHS scales 1e-200 and 1e200.

All eight Gonum single-RHS changes below have p<0.001, n=10 per version.
Reference times are medians from the candidate binary; comparison to Netlib
describes these specific fixtures, not general backend superiority.

| Solve, one RHS | n | Gonum before us | Gonum after us | Reference us | Gonum change |
| --- | ---: | ---: | ---: | ---: | ---: |
| DGETRS N | 128 | 69.85 | 18.38 | 7.691 | -73.69% |
| DGETRS T | 128 | 66.32 | 11.19 | 16.02 | -83.13% |
| DGETRS N | 256 | 275.22 | 76.37 | 30.35 | -72.25% |
| DGETRS T | 256 | 262.05 | 43.17 | 70.67 | -83.52% |
| DPOTRS U | 128 | 69.65 | 16.51 | 9.436 | -76.29% |
| DPOTRS L | 128 | 65.63 | 12.35 | 14.51 | -81.18% |
| DPOTRS U | 256 | 275.00 | 64.22 | 41.12 | -76.65% |
| DPOTRS L | 256 | 260.41 | 52.93 | 59.28 | -79.67% |

Normal LU remains 2.39–2.52 times the Reference-LAPACK median, and upper
Cholesky 1.56–1.75 times. Transposed LU and lower Cholesky are below the
reference medians. All native Reference-LAPACK before/after controls are
statistically inconclusive. Gonum sixteen-/sixty-four-RHS controls are also
inconclusive except DPOTRS n=128, nrhs=16, Upper: 94.71 to 94.96 us,
+0.27% (p=0.023). This small measured regression is retained in the record.
All native cases remain allocation-free.

A separate ten-round, 300 ms recheck of that exact Upper/n=128/nrhs=16
case does not reproduce the slowdown: Gonum 93.66 to 93.51 us (p=0.739),
Reference-LAPACK 153.5 to 153.4 us (p=0.353). The original result is not
discarded or pooled with the follow-up.

LU and Cholesky factorization-only controls at n=128/256, ten rounds at 200 ms,
show no significant change and unchanged allocation counts. This pass claims
solve improvements, not faster factorization.

## Runtime controls and binary identities

With GOMAXPROCS=4, ten 100 ms rounds at m=256 confirm DTRSM one-RHS
reductions of 70.69–85.41% and STRSM reductions of 71.94–86.41% (p<0.001).
All eight sixteen-RHS controls remain statistically inconclusive; all cases
remain allocation-free. This is a runtime-setting check, not a claim that
the single-RHS helper uses four workers.

With GOMAXPROCS=1 and GODEBUG=simd=0, ten 150 ms rounds show one-RHS public
LU reductions of 72.41–73.78% and Cholesky reductions of 76.97–77.52%,
at n=128/256 (p<0.001), without allocations. The new helper uses scalar Go
and does not enter a software-emulated vector loop. This does not claim that
all existing native architecture-specific leaves are disabled by GODEBUG.

Runner SHA-256 identities for the remaining prebuilt packages:

```text
mat baseline     5f68dd7346290b90f620cc3bfffd63dde5cb90e255a02606e6efa4ddfd2b814d
mat candidate    c017bb65394f281143d8f514ed4ea39902a0936b481b8d26c75c64b082508354
lapack baseline  33e72dd38460ae55148a70a05e09dd41526d0f82f895463d9be5777440abc1bc
lapack candidate d394037d65f6f056ca2d1667c28bd4480f7fa9ce1912c6bc2163a77436de8891
```

GOGC and GOMEMLIMIT are unset throughout; GODEBUG is unset except in the
explicit emulation run. These are repeated local M1 Pro measurements, not
cross-platform predictions. No AMD64 binary was cross-compiled or benchmarked.

## Next bottom-up targets

Separate three-second post-change profiles were captured only after all timed
acceptance runs. LU n=256/one-RHS now spends 97.82% of sampled CPU flat in
`dtrsmLeftVector`, with no AXPY entry among the reported hot frames. Further
work on the remaining normal-LU reference gap must improve the scalar
triangular arithmetic itself, while preserving numerical edge behavior.

The public QR n=256/one-RHS profile points to a different leaf: `AxpyInc`
accounts for 83.70% flat, beneath `gemvT` (90.22% cumulative), `Dlarft` and
`Dormqr`. Profile capture includes process setup, but setup factorization
accounts for only 0.82% of samples. The next priority is therefore a bounded
strided-AXPY and transposed-GEMV fixture matching those real strides, followed
by the same Dlarft/Dormqr and public QR/LQ consumer gates. These profiles
identify candidates; they are not evidence that another proposed change wins.

Keep two-/three-/four-RHS TRSM as a separate measured workload, and keep
SVD and factorization-only controls. Do not generalize this one-RHS result
into a claim that all Level 1, GEMM, LAPACK or AMD64 paths are optimized.

## Reproduction

Build the same benchmark packages from clean baseline and candidate worktrees,
using the same installed toolchain, tags and explicit PGO setting:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o lapack.test
```

The native build requires the existing Darwin/cgo Homebrew Reference-LAPACK
bridge. Run prebuilt binaries serially with the `go-optimisation` skill's
`scripts/compare_benchmarks.py`, `GOMAXPROCS=1`, `--rounds 10`, matching
`--baseline` and `--candidate` paths, and a new `--output` directory. Compare
its `baseline.txt` and `candidate.txt` with benchstat. Preserve metadata and
raw output; never run builds, tests or profiles concurrently with timing.

The BLAS selector is
`^Benchmark[DS]trsmBlockedSizes$/m(32|128|256)$/n(1|2|16)$`, with
`--benchtime 100ms`. It includes one-, two- and sixteen-RHS controls, all four
triangle/transpose orientations, and both precisions. These are only the
explicitly available size pairs in the persistent benchmark, not a full
Cartesian product of the selector's sizes.

Public solves use
`^BenchmarkFactorizationSolve$/(QR|LQ|LU|Cholesky|SVD)$/n=(128|256)$/nrhs=(1|16)$`
with `--benchtime 150ms`.

Native solves use `^BenchmarkD(getrs|potrs)Netlib$` with `--benchtime 100ms`.
Factorization controls use `^BenchmarkFactorization$/(LU|Cholesky)$/n=(128|256)$`
with `--benchtime 200ms`.

The GOMAXPROCS=4 BLAS selector is
`^Benchmark[DS]trsmBlockedSizes$/m256$/n(1|16)$`, at 100 ms. The
GODEBUG=simd=0 public selector is
`^BenchmarkFactorizationSolve$/(LU|Cholesky)$/n=(128|256)$/nrhs=1$`, at 150 ms.

The native regression recheck selector is
`^BenchmarkDpotrsNetlib$/n=128$/nrhs=16$/uplo=U$`, at 300 ms.
For diagnostic profiles, run the candidate mat binary (replace QR with LU for
its profile):

```sh
GOMAXPROCS=1 ./mat.test -test.run='^$' -test.bench='^BenchmarkFactorizationSolve$/QR$/n=256$/nrhs=1$' -test.benchtime=3s -test.cpuprofile=qr.cpu
GOTOOLCHAIN=go1.27.1 go tool pprof -top ./mat.test qr.cpu
```

Never overlap profiling with acceptance timing.
