# Bottom-up RowWise GEMV and LQ

## Scope and mechanism

Continue from `bc9a80b7e5d9133d2be149e8e38c0a72c05791ad`, following
`go-optimisation` and `gonum-simd` with dedicated Sol implementation,
code-generation and numerical reviews. A refreshed three-second LQ SolveTo
profile at n=256/one RHS puts 80.63% flat CPU in DotInc, 83.19% cumulative
in gemvN. Profiling is separate from acceptance timing.

The actual call chain is LQ.SolveTo -> Dormlq -> Dlarft(Forward, RowWise)
-> Dgemv(NoTrans) -> GemvN -> gemvN -> DotInc. Dlarft forms a column of T
using m=i preceding reflectors (normally 1..31 for block width 32), a long
contiguous matrix-row tail and contiguous x, but strided y. Dormlq's T
workspace has ldt=64. The existing contiguous SIMD gate requires incY=1,
so these products previously used one sequential DotInc per output row.

An additional ARM64-experimental gate selects 4<=m<=32, n>=8, incX=1,
positive incY>1, valid lda and complete overflow-safe slice spans. Output's
bounding span must be disjoint from active A and x spans. This is conservative:
overlap only in output stride gaps still falls back. Read-only A/x sharing
is allowed. Invalid slices, unsupported strides and overlapping outputs
retain the existing fallback, including its partial-write panic behavior.

The helper processes four rows together, with two-/one-row tails. It shares
each x load while retaining a separate, sequentially ordered sum per output.
It uses scalar arithmetic, not a horizontal SIMD reduction. The existing
portable GemvNSIMD reduction was not promoted to this path because it
reassociates terms; that is not automatically the same numerical contract.
Beta zero does not read old y; nonzero beta retains y*beta + alpha*sum.
DotInc itself, AMD64, float32 and default/safe/noasm implementations are unchanged.

Generated code has four register-resident scalar FMA accumulators, no
hot-loop helper calls, divisions, allocations or FP stack spills. Bounds
branches remain for the four matrix loads per column. The non-inlined
validator performs two overflow-check divisions once per eligible call;
cheap shape/stride predicates avoid it on nonqualifying shapes. There is
no claim that all bounds checks or dispatch overhead were eliminated.

## Measurement contract

Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power. No CPU
affinity or frequency pinning. Native builds use Go 1.27.1,
GOEXPERIMENT=simd and explicit -pgo=off; GOFLAGS is empty and no repository
PGO profile is present. Default compatibility uses Go 1.26.4.
[Go 1.27 release notes](https://go.dev/doc/go1.27#simd), the installed
simd.Emulated documentation and the
[portable SIMD proposal](https://github.com/golang/go/issues/78902) were
refreshed on 2026-09-06. This pass introduces no new experimental API.

Clean comparison baseline: `684ac2fa7f76389fafc682e5da1c66c14b931b3a`,
which adds only the common test/benchmark harness to the starting revision.
Candidate implementation: `f8d215fb0f8a369373e114b64e9cd35cec9d339a`.
Benchmark source is identical between these checkpoints; the candidate has
an additional validator-only test.

The direct GEMV fixture uses finite exact-binary inputs and beta=0, so every
iteration overwrites y with the same bounded result. It includes boundary,
long-tail, contiguous and unsupported-shape/stride controls. RowWise Dlarft
benchmarks generate valid LQ reflectors with Dgelq2 before timing and
overwrite T each iteration. Existing ColumnWise benchmark names remain
unchanged. Public solves reuse factorizations and include normal SolveTo
costs, not factorization; separate factorization benchmarks include that work.

The skill's compare_benchmarks.py runner alternates prebuilt baseline/candidate
order, rejects failed or mismatched runs, and records hashes and runtime
settings. Acceptance uses ten rounds on a quiet host, GOMAXPROCS=1 unless
specified, with GOGC/GOMEMLIMIT unset. No builds, tests or profiles overlap
timed cohorts. Three-round 100 ms screens were diagnostic only.
benchstat is golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0.

Raw logs, profiles, disassembly and runner metadata are local temporary
evidence under `/tmp/gonum-rowwise-gemv.JW8uih`; persistent benchmark entry
points and the source checkpoints permit reruns after that directory expires.

## Kernel results

Ten interleaved 150 ms rounds, GOMAXPROCS=1. All ten eligible cases improve
30.93–73.68% with p<0.001, n=10 per version. Every case remains at zero
B/op and allocs/op. Representative medians:

| m x n | incY | Baseline ns/op | Candidate ns/op | Change |
| --- | ---: | ---: | ---: | ---: |
| 4 x 8 | 32 | 23.01 | 15.53 | -32.49% |
| 5 x 9 | 64 | 29.25 | 20.20 | -30.93% |
| 31 x 16 | 64 | 297.5 | 133.2 | -55.23% |
| 4 x 128 | 64 | 509.9 | 149.0 | -70.79% |
| 31 x 128 | 32 | 3895 | 1245 | -68.04% |
| 31 x 256 | 64 | 8858 | 2678 | -69.77% |
| 32 x 256 | 64 | 9151 | 2408 | -73.68% |
| 31 x 512 | 64 | 18736 | 5565 | -70.30% |

Retained small fallback regressions: m=3/n=8 goes from 18.48 to 18.78 ns
(+1.62%); m=4/n=7 from 21.79 to 22.07 ns (+1.31%), both p<0.001.
These extra 0.28–0.30 ns are not hidden by the useful-size gains. The other
five contiguous or unsupported-shape/stride controls are statistically
inconclusive, not proven equivalent. Per-case changes are not an application
throughput aggregate.

## Public solve results

Ten interleaved 150 ms rounds, GOMAXPROCS=1. All four useful-size LQ
improvements have p<0.001, n=10 per version:

| LQ size | RHS | Baseline us/op | Candidate us/op | Change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 478.5 | 207.3 | -56.67% |
| 256 | 1 | 2036.8 | 825.8 | -59.45% |
| 128 | 16 | 893.3 | 622.1 | -30.35% |
| 256 | 16 | 3784 | 2576 | -31.93% |

Both LQ n=32 cases and all QR/LU/SVD solve controls are inconclusive.
Cholesky n=128/one RHS shows a small -0.18% change (16.27 to 16.24 us,
p=0.022); other Cholesky controls are inconclusive. No solve control has
a statistically significant slowdown in this cohort, which does not prove
exact equivalence. The unrelated small Cholesky movement is not attributed
to the new helper. Multiple per-case tests increase false-positive risk.

Allocations remain unchanged: QR/LQ two, LU/Cholesky zero, SVD seven per
solve. These are steady-state SolveTo results, not factorization timings.

## Reflector construction

Ten interleaved 150 ms rounds, GOMAXPROCS=1, k=32. All eight RowWise
Dlarft cases improve 50.14–68.69%, p<0.001, n=10, with zero allocations:

| n / ldv | ldt | Baseline us/op | Candidate us/op | Change |
| --- | ---: | ---: | ---: | ---: |
| 64 / 64 | 32 | 21.26 | 10.60 | -50.14% |
| 64 / 64 | 64 | 21.27 | 10.59 | -50.22% |
| 128 / 128 | 32 | 54.00 | 21.54 | -60.11% |
| 128 / 128 | 64 | 53.99 | 21.57 | -60.06% |
| 256 / 256 | 32 | 133.46 | 44.80 | -66.43% |
| 256 / 256 | 64 | 133.54 | 44.78 | -66.47% |
| 512 / 512 | 32 | 291.50 | 91.35 | -68.66% |
| 512 / 512 | 64 | 291.50 | 91.27 | -68.69% |

Two unchanged ColumnWise n=64 controls show small increases: ldt=32,
11.63 to 11.64 us (+0.07%, p=0.004); ldt=64, 11.61 to 11.64 us
(+0.21%, p=0.001). These observations are retained rather than attributed
to the new RowWise algorithm. The other six ColumnWise controls are
inconclusive. All ColumnWise cases remain allocation-free. Separate ten-round
300 ms rechecks do not confirm either small slowdown: ldt=32 is 11.62 to
11.63 us (p=0.839), ldt=64 is 11.63 to 11.63 us (p=0.616). These are
separate measurements, not pooled with or substituted for the original cohort.

## Factorization and runtime controls

Ten 150 ms rounds of separate factorization measurements find LQ wide
n=256 improving 3.82% (35.71 to 34.35 ms, p=0.019), with a wider 6%
candidate interval. LQ n=128 and all four QR square/tall n=128/256 cases
are inconclusive. A separate ten-round 300 ms LQ n=256 recheck confirms
the gain: 35.72 to 34.35 ms (-3.84%, p<0.001), with unchanged allocations.

Thin SVD square n=256 improves 1.02% (56.18 to 55.61 ms, p=0.011),
and wide n=256 improves 2.39% (82.29 to 80.32 ms, p=0.001). Tall n=256
and all three n=128 shapes are inconclusive. These small measured effects
do not establish improvements for all SVD modes or sizes. Allocation counts
and byte medians are unchanged across both factorization cohorts.

With GOMAXPROCS=4, ten 150 ms rounds retain LQ solve gains: one RHS
n=128/256 improves 56.69%/59.62%, and sixteen RHS improves 30.42%/33.84%.
All four have p<0.001; all four QR controls are inconclusive. Allocations
are unchanged.

With GOMAXPROCS=1 and GODEBUG=simd=0, ten 150 ms rounds give LQ n=256
one RHS 2039.3 to 825.1 us (-59.54%) and sixteen RHS 3.893 to 2.680 ms
(-31.16%), both p<0.001. Both QR controls are inconclusive and allocations
unchanged. The new helper is scalar; portable emulation does not disable
all architecture-specific leaves in the rest of Gonum.

## Numerical validation

Persistent tests cover the m=3/4/5 and 31/32/33 boundaries, n=7/8/9 and
longer tails, offsets, ldt=32/64, beta-zero NaN outputs, exact signed zero,
ordered overflow/cancellation, subnormals and bitwise comparisons with the
retained scalar path. NaNs are compared by classification, not payload.
Tests preserve output gaps and guards, verify read-only A/x storage, exercise
true active A/x overlap, output overlap, negative/zero/nonunit increments,
short slices with capped capacities, and overflow-safe span rejection.

Existing independent Dlarft reflector reconstruction, Dgelqf versus
unblocked Dgelq2, Dormlq versus Dorml2, and public LQ reconstruction/solve
checks cover the consumers. No numerical tolerance is relaxed. This pass
does not add a Netlib LQ bridge or claim to beat optimized vendor BLAS.

Validation passed: full `go test -pgo=off ./...` on default Go 1.26.4 and
experimental Go 1.27.1; affected f64/BLAS/LAPACK/mat suites with safe, noasm
and bounds tags; focused f64 race; and GODEBUG=simd=0 GEMV/LQ/QR/SVD and
reflector tests. Common RowWise tests also pass on the clean baseline.
Formatting, import-policy, copyright and diff checks pass.

## Remaining measured work

Post-change three-second profiles were collected only after timing finished.
For LQ n=256/one RHS, DotInc is now 6.10% flat, while the new GemvN helper
is 59.59% flat / 61.05% cumulative. Its residual bounds/address work is a
future tuning hypothesis. Transposed/transposed DGEMM is 22.97% cumulative,
including strided AXPY work. LQ setup appears in this profile (Factorize
2.03% cumulative); it remains excluded from steady-state SolveTo timing.

For thin wide SVD n=256, Dbdsqr is 38.05% cumulative and Dlasr 34.22%
cumulative; these nested costs must not be added. Dlasr itself is 19.47%
flat and its blocked right-variable helper 13.86% flat. DotUnitary contributes
21.83% cumulative, and SIMD DGEMM 12.39%. This provides a measured next
LAPACK target: rotation application beneath bidiagonal SVD, with GEMV/dot
consumers retained as controls. These profiles identify hypotheses, not
accepted additional optimizations or a vendor-library speed claim.

## Reproduction

Build the same package on each checkpoint with:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./internal/asm/f64 -o /tmp/f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./lapack/gonum -o /tmp/lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
```

Use separate output paths for the two builds. No cross-compilation or native
AMD64 timing is performed; other-backend promotion still needs native evidence.

Acceptance selectors (ten 150 ms rounds, default GOMAXPROCS=1):

```text
f64:    ^BenchmarkGemvNRowWiseDlarft$
lapack: ^BenchmarkDlarft$
mat:    ^BenchmarkFactorizationSolve$/(LQ|QR|LU|Cholesky|SVD)$/n=(32|128|256)$/nrhs=(1|16)$
mat:    ^BenchmarkFactorization$/(QR|LQ)$/shape=(square|tall|wide)$/n=(128|256)$
mat:    ^BenchmarkFactorization$/SVD$/kind=thin$/shape=(square|tall|wide)$/n=(128|256)$
```

The P4 cohort uses `^BenchmarkFactorizationSolve$/(QR|LQ)$/n=(128|256)$/nrhs=(1|16)$`.
The emulation cohort uses `^BenchmarkFactorizationSolve$/(QR|LQ)$/n=256$/nrhs=(1|16)$`.
Focused rechecks use ten 300 ms rounds and the same binaries:
`^BenchmarkDlarft$/n=64$/ldv=32$/ldt=(32|64)$` and
`^BenchmarkFactorization$/LQ$/shape=wide$/n=256$`.

For the skill runner, pass `--baseline`, `--candidate`, `--bench`,
`--rounds 10`, `--benchtime 150ms` and a fresh `--output` directory; run
benchstat on its baseline.txt and candidate.txt. Record the same runtime
environment on both sides. Profile separately with the selected benchmark,
`-test.run '^$' -test.benchtime=3s -test.cpuprofile=/tmp/profile.cpu`.

Final binary SHA-256 identities:

```text
f64 baseline     32eea6cfd4c64a05f1a63d58d96d9cb7eb457aac9580db676640769fb2a3a58e
f64 candidate    94191a6eab57c1280e3632ccc18301f8de6de7cacb290c9fceb7bb768adf2fd3
lapack baseline  41892c8a6df42a69a13cdbaa4b1c5d61b0595461ac33d6ca9ddd5982c32d9d46
lapack candidate 3f9360653a429ccc6ebfedf0ee45a43292d82045ddfcf506bb85f195438373ad
mat baseline     8852efa8a6b8c0ba8ee7621d1db38bbb135d1c835f86dd2cd01649140acba319
mat candidate    d9b0b89e01fa6f859f523ac3def7c5f54508c50933a9e06b32958c0e0fd06374
```
