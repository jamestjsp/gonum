# Bottom-up strided GEMV and QR

## Scope

Continue from `63e967af0426bdb8acd58a2f7bc2c7a6f871b6f6`, using
`go-optimisation`, `gonum-simd` and dedicated Sol code/numerical reviewers.
The preceding QR SolveTo n=256/one-RHS profile placed 83.70% flat CPU in
`AxpyInc`, beneath transposed GEMV, Dlarft and Dormqr. This pass changes BLAS
composition, not a LAPACK algorithm or public API.

Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83). Experimental builds
use Go 1.27.1, GOEXPERIMENT=simd and explicit -pgo=off; default tests use
installed Go 1.26.4. No GOFLAGS override or repository PGO profile is present.
The native comparison uses Homebrew Reference-LAPACK 3.12.1, not Accelerate
or OpenBLAS.
Official [release history](https://go.dev/doc/devel/release), the installed
package documentation and [portable SIMD proposal](https://github.com/golang/go/issues/78902)
were refreshed on 2026-09-06. No new SIMD API is required by this pass.

Dlarft Forward/ColumnWise calls Dgemv(Trans) with input stride ldv and output
stride ldt. Its matrix-row AXPY has unit input stride and strided output.
Dormqr uses ldt=64 even when its reflector block width is 32. Benchmark both
that actual workspace and packed ldt=32; they are not interchangeable fixtures.

## Rejected leaf hypothesis

A unit-input-stride AxpyInc range loop removed repeated x index bounds checks
behind an overflow-safe valid-span gate, retaining the original invalid-input
fallback. Both float64 and float32 bodies remained inlineable at cost 68/80,
including in transposed GEMV, with scalar fused arithmetic and no per-row call.

Three-round, 100 ms diagnostic screens rejected it. Float64 length 17 at
output stride 32 rose from 13.65 to 22.08 ns; GEMV 128x16 with reflector strides
rose from 2.061 to 2.829 us. Public QR one-RHS n=128 rose from 415.0 to
511.5 us and n=256 from 1.632 to 2.018 ms. Three samples are not enough for
the acceptance significance test; these are diagnostic rejection medians,
not accepted performance claims. Both production AXPY edits were fully reverted.
Removing a bounds check did not establish better generated-code performance.

## GEMV candidate and contracts

An ARM64 experimental dispatch gate selects m>=8, 1<=n<=32, positive incX
and incY>1, preserving the existing contiguous dispatch and all fallback
implementations. It validates complete active spans without integer overflow
before reading or writing data. Invalid, overlapping and unsupported-stride
calls retain the old helper, including partial-write panic behavior. Read-only
A and x may share storage, as they do in Dlarft.

The helper keeps four independent output values in scalar registers across
matrix rows, with two-/one-output tails. Every output retains its original
ordered updates, including a separate alpha*x scale. Beta zero starts with
positive zero without reading old y. The
change reduces strided output load/store traffic; it does not add wider SIMD
instructions or regroup a dot-product reduction.

Persistent leaf/GEMV tests cover real strides and boundaries; new Dlarft
conformance coverage reconstructs the reflector at block width 32 with ldt=64.
The reusable Dlarft benchmark lives in lapack/testlapack/dlarft_bench.go,
with the implementation wrapper in lapack/gonum/bench_test.go.

## Measurement contracts

AxpyInc benchmarks use exact-binary bounded inputs and alternating alpha.
New GEMV benchmarks use beta=0, producing the same finite result each time;
the existing strided GEMV control was corrected from unbounded beta=1 to
beta=-1, which keeps repeated updates bounded. Dlarft benchmarks generate valid
Householder vectors before timing and overwrite T each iteration. Public solves
reuse factorizations and include their ordinary API costs, not factorization.

Compare identical prebuilt harnesses serially with the go-optimisation skill
runner. Acceptance uses ten interleaved rounds, GOMAXPROCS=1 unless explicitly
varied, unset GOGC/GOMEMLIMIT, and no concurrent builds, tests or profiles.
Local raw output and metadata are under `/tmp/gonum-strided-gemv.CXnTAM`;
that directory is temporary. Persistent benchmark entry points permit reruns.

Clean baseline checkpoint: `3e328a6074755a6e49798a94f2087bcf79b93aab`,
which adds only the common benchmark/test harness to the starting implementation.
The candidate is implementation commit `c6416f1e`; benchmark source is identical
between these revisions. Candidate-only tests exercise the new span validator.
The host is on AC power; no CPU affinity or frequency pinning is imposed.

Full default/SIMD tests pass, along with affected safe/noasm/bounds suites,
focused race and GODEBUG=simd=0 checks, independent reflector reconstruction,
and the existing native QR reconstruction/orthogonality tests. No tolerance was
relaxed. The short-A panic fixture explicitly caps cloned slices because spare
capacity otherwise permits the original two-index row reslice; both baseline
and candidate run that same corrected fixture.

Final machine code retains four FP accumulators, one scale multiplication and
four scalar fused updates per row. There are no FP stack spills or AXPY calls
in the row loop. Bounds checks remain. The non-inlined validator performs
three overflow-check divisions per qualifying call; cheap shape/stride tests
are outside it so nonqualifying fallbacks do not pay the extra call. No claim
is made that bounds checks or all setup overhead were eliminated.

## Accepted GEMV measurements

Ten interleaved rounds at 150 ms, GOMAXPROCS=1. Every reflector-shaped
case improves with p<0.001, n=10 per version; all cases remain at zero B/op
and allocs/op. Representative exact medians:

| m x n | incX / incY | Baseline ns/op | Candidate ns/op | Change |
| --- | --- | ---: | ---: | ---: |
| 128 x 1 | 128 / 32 | 423.8 | 160.7 | -62.09% |
| 128 x 7 | 128 / 32 | 1620.0 | 516.7 | -68.10% |
| 128 x 16 | 128 / 32 | 2139.0 | 715.1 | -66.57% |
| 128 x 31 | 128 / 32 | 3240 | 1485 | -54.18% |
| 128 x 16 | 128 / 64 | 2180.5 | 712.0 | -67.35% |
| 128 x 31 | 128 / 64 | 3305 | 1490 | -54.92% |
| 256 x 1 | 256 / 32 | 860.2 | 321.3 | -62.64% |
| 256 x 16 | 256 / 32 | 4402 | 1532 | -65.21% |
| 256 x 31 | 256 / 32 | 6650 | 3288 | -50.56% |
| 256 x 16 | 256 / 64 | 4594 | 1534 | -66.60% |
| 256 x 31 | 256 / 64 | 6884 | 3292 | -52.18% |

The two contiguous reflector controls are inconclusive. The existing bounded
incX=2/incY=3 sweep improves 23.61–68.96% at m=10, n=8..32, and
25.30–81.93% at m=1000, n=8..32 (all p<0.001). The n=1000 controls and
m=1000/n=33 are inconclusive. The m=10/n=33 fallback rises from 159.1 to
159.8 ns, +0.44% (p<0.001); retain this small dispatch-boundary regression.

The six admitted boundary cases at m=8/9 improve 43.82–72.39% (p<0.001).
The m=7 control is inconclusive; the m=8/n=33 control improves 0.19%
(p=0.001), too small to motivate an algorithmic claim. These ranges are
per-case observations, not a workload-throughput aggregate.

## Public solve measurements

Ten interleaved rounds at 150 ms, GOMAXPROCS=1. The four QR gains have
p<0.001, n=10 per version. These are SolveTo timings on existing factorizations.

| QR size | RHS | Baseline us/op | Candidate us/op | Change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 414.9 | 205.7 | -50.44% |
| 256 | 1 | 1631.3 | 824.3 | -49.47% |
| 128 | 16 | 757.7 | 548.1 | -27.66% |
| 256 | 16 | 2948 | 2133 | -27.65% |

All six LQ controls are statistically inconclusive. Small significant increases
in the initial control matrix are retained below rather than hidden by an
aggregate. All other LU/Cholesky/SVD controls are inconclusive.

| Control | Baseline us/op | Candidate us/op | Change | p |
| --- | ---: | ---: | ---: | ---: |
| QR 32, one RHS | 9.138 | 9.149 | +0.12% | 0.024 |
| QR 32, sixteen RHS | 33.53 | 33.59 | +0.21% | 0.010 |
| Cholesky 128, one RHS | 16.22 | 16.27 | +0.33% | 0.001 |
| SVD 128, one RHS | 203.1 | 203.8 | +0.34% | 0.029 |
| LU 256, one RHS | 76.33 | 76.42 | +0.12% | 0.011 |
| Cholesky 256, one RHS | 62.54 | 62.75 | +0.33% | 0.002 |

Allocations do not change in any public solve: LU/Cholesky remain at zero,
QR/LQ retain two, and SVD seven. Many simultaneous per-case significance tests
can produce small false positives; focused follow-up evidence is kept separate
from this initial matrix.

## LAPACK and runtime measurements

All Dlarft cases use reflector block width 32. Ten 150 ms rounds, p<0.001
for every improvement, zero allocations:

| n | ldv | ldt | Baseline us/op | Candidate us/op | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 32 | 32 | 21.18 | 11.64 | -45.05% |
| 64 | 32 | 64 | 21.08 | 11.63 | -44.85% |
| 128 | 64 | 32 | 45.96 | 22.90 | -50.18% |
| 128 | 64 | 64 | 45.99 | 22.91 | -50.18% |
| 256 | 128 | 32 | 116.98 | 51.73 | -55.77% |
| 256 | 128 | 64 | 117.85 | 52.54 | -55.42% |
| 512 | 256 | 32 | 239.5 | 107.3 | -55.21% |
| 512 | 256 | 64 | 240.5 | 107.4 | -55.35% |

The native DGEQRF comparison has seven matrix shapes, ten rounds at 150 ms.
Gonum 256x256 improves from 5.339 to 5.137 ms (-3.78%, p<0.001);
Reference-LAPACK is 6.891 ms in the candidate run. Gonum 128x128 is
744.7 us versus reference 769.3 us, but its before/after change is inconclusive.
Small cases still have a reference gap: 32x32 is 15.84 versus 10.32 us,
and 64x32 is 36.34 versus 29.91 us. These are case-specific comparisons,
not claims about Accelerate or OpenBLAS.

Gonum 128x256 initially rises 0.25% (1.723 to 1.727 ms, p=0.015);
the untouched native 32x64 control also rises 0.26% (23.09 to 23.15 us,
p=0.009). Other native before/after cases are inconclusive. Every native
case remains allocation-free; these small changes receive a separate recheck.

The initial public LU/Cholesky factorization controls at n=128/256 improve
1.22–3.39% (p<0.001 except Cholesky 256, p=0.002), retaining 72 B/op.
These are observed public-call effects, not a claim that their factorization
algorithms were rewritten. Shape-qualified QR/LQ/SVD factorization results are
collected separately from these four controls.

With GOMAXPROCS=4, QR one-RHS improves 51.75% at n=128 and 49.57% at
n=256; sixteen-RHS improves 27.84% and 30.06% respectively. All four gains
have p<0.001, n=10. LQ controls are inconclusive and allocations unchanged.
The helper itself is serial; this is a caller runtime-setting check.

With GODEBUG=simd=0 and GOMAXPROCS=1, QR n=256 improves 49.66%
for one RHS (1628.2 to 819.6 us) and 16.23% for sixteen RHS
(4.973 to 4.166 ms), both p<0.001. LQ controls remain inconclusive,
with unchanged allocations. The smaller relative sixteen-RHS gain reflects
other work still performed through the emulated portable backend.

## Focused rechecks and factorization controls

Separate ten-round 300 ms rechecks do not confirm significant slowdowns in
the initial GEMV or public solve controls. GEMV m=10/n=33 remains 159.2
versus 159.9 ns (p=0.126), so its small median difference is not erased.
LU/Cholesky/SVD one-RHS cases at n=128/256 are all inconclusive; QR n=32,
one RHS is inconclusive and sixteen RHS improves 0.61% (p=0.029). The
original samples remain separately reported, not pooled with these rechecks.

A ten-round 200 ms native recheck finds Gonum 128x256 inconclusive
(1.724 versus 1.727 ms, p=0.853). The unchanged Reference-LAPACK 32x64
control still rises 0.27% (23.08 to 23.14 us, p=0.007); retain that measurement
rather than attributing it to a changed native algorithm.

Shape-qualified public factorization benchmarks, ten 150 ms rounds, show QR
square n=256 improving 3.87% (5.587 to 5.371 ms, p<0.001), and tall n=256
3.74% (13.39 to 12.89 ms, p=0.001). Both QR n=128 cases and LQ n=128
are inconclusive. Wide LQ n=256 rises 0.36% (35.70 to 35.83 ms, p=0.029).
A separate ten-round 300 ms recheck is inconclusive (35.77 to 35.72 ms,
p=0.218). Both measurements are retained separately from the QR gains.

Thin SVD factorization at n=128 is inconclusive for both square and tall
matrices. At n=256, square improves 1.45% (57.16 to 56.33 ms) and tall
2.51% (89.51 to 87.26 ms), both p=0.002, n=10. These modest measured
factorization changes do not mean that SVD solves or all SVD modes improved.
Allocation counts remain unchanged in the factorization cohorts.

## Remaining measured work

Separate three-second post-change profiles, collected after the main acceptance
runs, put 68.88% flat QR one-RHS n=256 CPU in the new scalar GEMV helper.
Its remaining bounds/address work and possible wider output tiles are future
hypotheses, not accepted changes. This pass deliberately stops short of another
ISA-specific implementation without comparison evidence.

LQ one-RHS n=256 now gives a clear adjacent bottom-up target: DotInc is
80.23% flat, beneath non-transposed gemvN, Dlarft and Dormlq. A/X are
contiguous in relevant RowWise reflector products while the output column is
strided; verify the exact strides before constructing the next fixture.
Profiles include setup (0.91% QR factorization, 2.62% LQ factorization),
which is excluded from steady-state benchmark timing.

AMD64 assembly, float32 AXPY and default/safe/noasm implementations remain
unchanged. There is no native AMD64 performance claim and no cross-compilation.
The new scalar helper remains behind the existing ARM64 experiment boundary;
promoting it to other backends requires native baseline comparisons there.

## Binary identities

SHA-256 values recorded by the runner:

```text
f64 baseline     9346bb4c1909afbd9932ebf1437363010ba80e7020a39c324e9e4e20bfce46d9
f64 candidate    c274900c370f6e1ad7cce4fb8427dd9267f461ba967898f31260198e43d3a638
mat baseline     a5848112db5fd4dd796468fa31b3448e626885c82b9f4fc9061b8f9576ccb70f
mat candidate    965ce82b72b4655eba2c259ab3801c478b0c9eaf9cbdd9a8c81d578a35347570
lapack baseline  e98e8244ec331289d868a4cad1c0da3cb558bd92758aa0352ea09e6d54f30aaf
lapack candidate 7fab494d04308103e794026a04c726c4b5e702173f37cfca16d582f012ecb56b
native baseline  975f7211aef148857d6b675b1101ece7b97ebf1ef870536194cd0dbc779935e2
native candidate c40c18f0df274350b7fa45c5a5a87e80f92313e3aa3527e7f797ca01cdeeb132
```

## Reproduction

Build each package in the clean baseline and candidate worktrees using the same
installed toolchain and explicit PGO setting:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./internal/asm/f64 -o f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./lapack/gonum -o lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o native.test
```

The native build requires the existing Darwin/cgo Homebrew Reference-LAPACK
bridge. Dynamic links resolve to /opt/homebrew/opt/lapack/lib, not Accelerate
or OpenBLAS. Native QR timing includes an equal matrix copy and factorization
on both sides; layout conversion, workspace queries and allocation are outside
the timer. Both backends are checked by independent reconstruction/orthogonality
tests, not by requiring identical Householder factors.

Run the go-optimisation skill's scripts/compare_benchmarks.py with matching
prebuilt --baseline/--candidate paths, a new --output directory, --rounds 10,
--benchtime 150ms and GOMAXPROCS=1. Preserve its raw output and metadata.
Benchstat version: golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0.

| Package | Exact --bench selector |
| --- | --- |
| f64 kernel/control | `^BenchmarkGemvT(Dlarft\|Arm64StridedBoundary\|Strided)$` |
| public solves | `^BenchmarkFactorizationSolve$/(QR\|LQ\|LU\|Cholesky\|SVD)$/n=(32\|128\|256)$/nrhs=(1\|16)$` |
| reflector construction | `^BenchmarkDlarft$` |
| native QR | `^BenchmarkDgeqrfNetlib$` |
| LU/Cholesky factors | `^BenchmarkFactorization$/(LU\|Cholesky)$/n=(128\|256)$` |
| QR/LQ factors | `^BenchmarkFactorization$/(QR\|LQ)$/shape=(square\|tall\|wide)$/n=(128\|256)$` |
| thin SVD factors | `^BenchmarkFactorization$/SVD$/kind=thin$/shape=(square\|tall)$/n=(128\|256)$` |

The GOMAXPROCS=4 solve selector is
`^BenchmarkFactorizationSolve$/(QR|LQ)$/n=(128|256)$/nrhs=(1|16)$`.
The GODEBUG=simd=0, GOMAXPROCS=1 selector is
`^BenchmarkFactorizationSolve$/(QR|LQ)$/n=256$/nrhs=(1|16)$`.
Both also use ten rounds at 150 ms. The latter selects portable emulation;
it does not disable every architecture-specific native leaf. This scalar helper
does not invoke emulated vector operations.

Focused rechecks use ten rounds with the same binaries and environment:

- 300 ms: `^BenchmarkGemvTStrided$/m=10$/n=33$`.
- 300 ms: `^BenchmarkFactorizationSolve$/(Cholesky|SVD|LU)$/n=(128|256)$/nrhs=1$`.
- 300 ms: `^BenchmarkFactorizationSolve$/QR$/n=32$/nrhs=(1|16)$`.
- 200 ms: `^BenchmarkDgeqrfNetlib$/m=(32|128)$/n=(64|256)$`.
- 300 ms: `^BenchmarkFactorization$/LQ$/shape=wide$/n=256$`.

For diagnostic profiles, run the candidate mat binary with GOMAXPROCS=1,
`-test.run='^$' -test.bench='^BenchmarkFactorizationSolve$/QR$/n=256$/nrhs=1$'
and `-test.benchtime=3s -test.cpuprofile=qr.cpu`; replace QR with LQ for its
profile. Inspect using Go 1.27.1 `go tool pprof -top`. Do not overlap profiles
or validation builds with benchmark timing.
