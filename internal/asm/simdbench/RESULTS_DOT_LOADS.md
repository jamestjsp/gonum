# ARM64 transposed-B DGEMM and SVD consumers

Status: accepted paired-output ARM64 DGEMM implementation, committed as
`937183b7fe1b8db17c0cb0458f9483530dbc1d56`. The broad DotUnitary replacement
described below was rejected and removed after repeatable consumer regressions.
The shared f64 production code remains unchanged.

## Scope and environment

Measured on 2026-09-06 against `8b8e472d4399bc7ec94526c4e071dfacdfbadf65`,
also the fetched `origin/codex/arm64-simd-blas` tip before this work.
Baseline implementation is unchanged in an independent worktree; both builds
receive the same persistent benchmark and dot regression harness, published in
`486b6a88`. The implementation baseline predates that harness-only commit.

Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power.
Builds use Go 1.27.1, `GOEXPERIMENT=simd`, `-pgo=off`, empty GOFLAGS;
GOGC, GOMEMLIMIT and GODEBUG are unset. Timing uses prebuilt binaries, alternating
baseline/candidate then candidate/baseline, with no concurrent compilation,
tests or profiling. No CPU affinity or frequency pinning was used. Normal
desktop background activity remains; indexing bursts observed during preparation
are excluded from timing. Compatibility uses default Go 1.26.4.

The original target was `internal/asm/f64/dot_simd_arm64.go`; the accepted
change instead reuses its arithmetic within an isolated transposed-B DGEMM
kernel using ARM64 NEON `simd/archsimd`. Default, AMD64
assembly, safe, noasm and GCCGo selection are unchanged. No cross-compilation
or native AMD64 measurements were performed. Go SIMD remains experimental;
the [Go 1.27 notes](https://go.dev/doc/go1.27#simd) and installed Go 1.27.1
`src/simd/archsimd/slice_gen_arm64.go` were checked before editing.

## Why this leaf

The previous final thin-wide n=256 public SVD profile had 3.30 seconds of CPU
samples: DotUnitary accounted for 21.21% flat and 26.67% cumulative; 0.79 of
its 0.88 cumulative seconds came through `dgemmSerialNotTrans`.

The dominant DGEMM source is `Dlarfb` (0.99 of 1.32 sampled DGEMM seconds),
principally reached from `Dorglq` and `Dgelqf`. Wide SVD first LQ-reduces a
256-by-512 matrix. Applying a right-side rowwise Householder block calls
`Dgemm(NoTrans, Trans, ...)`, initially producing 224-by-32 outputs with a
dot reduction length of 480, then lengths 448 and below. The portable DGEMM
tile declines transposed B, so this path calls the unitary dot leaf.

`Dgebrd` supplies a smaller DGEMM contribution (0.17 sampled seconds). It
operates on the reduced 256-by-256 matrix here; its first trailing update has
224-by-224 outputs and reduction length 32. An initial interpretation that
length-32 dots dominated this workload was corrected before timing.

## Rejected leaf prototype: change and contract

Only load windows change. Each eight-product iteration forms fixed,
capacity-bounded eight-element operand slices, then loads four fixed pairs.
The second operand is first capped at its logical length so a short slice
cannot silently read through spare capacity. The four accumulators, fused
multiply-add operations, reduction tree, paired remainder and scalar tail
retain their existing order. No unsafe pointers, new dispatch or tuning
threshold are introduced.

Go 1.27.1 disassembly reduces the main eight-product loop from roughly 90
instructions to roughly 24: eight vector loads, four vector FMAs and the
remaining block bounds/address/loop bookkeeping. Per-vector bounds branches
and capacity masking disappear. Block bounds checks remain; this is not a
claim of a bounds-check-free loop. There are no hot-loop calls or spills;
escape diagnostics keep both operands nonescaping.

Persistent tests exercise all lengths 0 through 80, larger tails through 1025,
offsets, longer second operands, overlapping read-only slices, unchanged input
storage, short operands with spare capacity, cancellation, overflow,
subnormals, signed zero, infinity and NaN. The reference deliberately models
this selected ARM64 kernel's accumulation order, not universal bitwise
agreement between architectures. These tests pass on the unchanged baseline.

## Rejected leaf prototype: measurements

All results below are native interleaved comparisons, ten samples per binary
per case. Percentages describe time reductions, not application throughput.
Per-case significance uses benchstat; many comparisons increase the chance
of small false positives. All DGEMM cases allocate 0 B/op and 0 allocs/op.

### Public transposed-B DGEMM, P1

100 ms per benchmark per round, including public validation, dispatch and
beta-zero output handling. Inputs stay finite and unchanged between calls.

| Output m by n; reduction k | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| 96 by 32; 224 | 378.08 us | 95.94 us | -74.62% |
| 224 by 32; 480 | 1.8698 ms | 491.8 us | -73.70% |
| 224 by 32; 480, padded | 1.8731 ms | 504.8 us | -73.05% |
| 96 by 224; 32 | 411.1 us | 163.4 us | -60.24% |
| 224 by 224; 32 | 958.0 us | 381.4 us | -60.18% |
| 224 by 480; 32 | 2.0484 ms | 822.5 us | -59.85% |
| 224 by 480; 32, padded | 2.0593 ms | 893.8 us | -56.60% |

All these differences have p<0.001. The first two Householder-block shapes
and the square bidiagonal update represent the traced consumer; other shapes
and padding are coverage, not claims about sampled call frequencies.
Larger square/padded controls improve 69.66-71.53%; k=16 through 33 boundary
cases improve 40.78-57.42%. No SIMD crossover is changed.

Tiny unchanged-scalar cases show two significant regressions: 1-by-1/k=7
24.63 to 24.92 ns (+0.29 ns, +1.18%, p=0.001), and 4-by-4/k=7
128.0 to 128.2 ns (+0.2 ns, +0.20%, p=0.029). Padded k=8 is inconclusive;
k=15 is -0.52%. These subnanosecond changes are retained, not hidden by an
aggregate. They do not justify a new short-vector dispatch or arithmetic change.

### Dot and public factorization

Clean `dot-accept` timings (100 ms) show length 16 at 9.705 to 4.973 ns
(-48.76%), 32 at 18.290 to 6.213 ns (-66.03%), 480 at 259.80 to 63.98 ns
(-75.37%), and 4096 at 2219.5 to 621.1 ns (-72.02%). All have p<0.001,
with zero allocations. Offset-one results agree; all lengths below 16 are
inconclusive. These leaf improvements did not satisfy the consumer gate.

Public n=256 SVD improves 8.37-19.80%, but smaller cases regress. A separate
ten-round, 300 ms confirmation reproduces square/tall n=32/128 regressions:
2.67-9.25% for singular values only and 2.81-4.72% for thin vectors. Controls
also expose QR n=32/128 regressions of 7.50-13.04% and LU regressions of
11.49-16.89% at every measured size. No geometric mean is used to accept this.

Independent disassembly finds unchanged arithmetic instructions in Ger,
Axpy, GEMV, rotations and inspected LAPACK consumers, but changed code
placement after the smaller DotUnitary symbol. Ger changes cache-line entry
offset from zero to 32. This is strong evidence of a text-layout effect,
not proof that every cycle of regression is explained by entry alignment.
The replacement was removed rather than adding padding or alignment hacks.

## Accepted isolated DGEMM implementation

The implementation keeps the original shared f64 leaf unchanged and pairs
two output columns in `blas/gonum/dgemm_trans_simd_arm64.go`. Each eight-element
step reuses four A vectors across two B rows: twelve loads instead of sixteen,
with eight vector accumulators retaining each output's original dot order.
There are no accumulator spills, hot-loop calls or heap allocations. Three
fixed-window bounds groups remain. The existing active-row disjointness check
allows safe shared LAPACK panels and rejects actual C/A or C/B overlap before
any output write, including overlap only with the odd final B row.

Only NoTrans/Trans or NoTrans/ConjTrans enters the tile. The guard checks n>=2
and k>=16 before calling it; k=16 retains the original dot's transition to
four-accumulator arithmetic. Odd output columns use the original DotUnitary.
No alpha-zero behavior is changed. Non-ARM64 and float32 helpers return false;
the single-precision generator explicitly maps the new helper name. No
experimental types escape the implementation package or public API.

The first paired-output prototype still called the helper on ineligible tiny
matrices, costing about 4-7 ns. Moving its cheap eligibility checks ahead of
the call removed most of that cost. Final code preserves the original f64
Dot/Ger/GEMV addresses. Inspected LAPACK rotation bodies retain their original
instruction sequences and cache-line alignment, with a uniform +0x6c0 shift.
No padding, alignment directives or architecture-independent leaf replacement
are used to force those measurements.

### Final public DGEMM, P1

Ten rounds, 100 ms per case, public-call timing, all 0 B/op and 0 allocs/op.

| Output m by n; reduction k | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| 96 by 32; 224 | 377.33 us | 72.06 us | -80.90% |
| 224 by 32; 480 | 1.8703 ms | 362.6 us | -80.61% |
| 224 by 32; 480, padded | 1.8703 ms | 371.6 us | -80.13% |
| 224 by 224; 32 | 957.6 us | 224.5 us | -76.56% |
| 1 by 2; 16 | 41.44 ns | 30.12 ns | -27.33% |
| 1 by 3; 480 | 809.6 ns | 377.8 ns | -53.33% |

All tabulated differences have p<0.001. Other length-32 panel shapes improve
74.99-76.63%; long-reduction n=31/32/33 controls improve 77.55-80.67%,
including odd output tails. Larger 128/129/256 reduction controls improve
76.56-79.80%.

Five ineligible tiny cases show significant overhead: 1-by-1/k=7 +0.28 ns,
1-by-1/k=16 +0.62 ns, 1-by-1/k=17 +0.57 ns, 4-by-4/k=7 about +0.3 ns,
and padded 4-by-4/k=8 about +0.3 ns. The largest is +2.05%; k=15 cases
slightly improve. This bounded subnanosecond dispatch tradeoff is recorded,
not described as universal improvement.

### Final public factorizations and Netlib

All final cohorts use ten interleaved rounds at 100 ms per case. There are no
significant timing regressions among 45 P1 public factorization cases or the
33 P4 SVD/QR/LQ/LU/Cholesky cases. Smaller n=32/128 SVD/QR/LQ results are
inconclusive, not proof of exact equivalence. Changes below are time reductions
at n=256; all listed improvements have p<0.001.

| Public workload | P1 baseline → candidate | P1 change | P4 change |
| --- | ---: | ---: | ---: |
| SVD values, square | 17.61 → 15.75 ms | -10.56% | -3.33% |
| SVD thin, square | 49.06 → 43.47 ms | -11.40% | -5.33% |
| SVD values, tall | 31.09 → 24.72 ms | -20.49% | -7.16% |
| SVD thin, tall | 80.08 → 65.37 ms | -18.37% | -7.73% |
| SVD values, wide | 28.03 → 21.83 ms | -22.12% | -15.38% |
| SVD thin, wide | 73.66 → 59.58 ms | -19.12% | -14.16% |
| QR, square | 5.366 → 3.492 ms | -34.92% | -15.13% |
| QR, tall | 12.904 → 8.372 ms | -35.12% | -14.03% |
| LQ, wide | 34.29 → 19.67 ms | -42.63% | -28.08% |

P1 n=256 EigenSym with vectors improves 6.26% (30.11 → 28.22 ms), and
Eigen with right vectors improves 10.00% (62.56 → 56.31 ms). LU, Cholesky
and eigenvalues-only controls are inconclusive at this size. Sub-percent
favorable changes in small P4 controls are not attributed to this kernel.
Public B/op is effectively unchanged. Isolated allocation-count shifts
(including P1 LQ 9 → 10 and P4 LQ 1218 → 1215) do not establish a memory
improvement or regression; the kernel itself allocates zero.

The separate native reference-BLAS/LAPACK comparison gives n=256 square thin
SVD at 44.46 → 38.89 ms for Gonum (-12.53%, p<0.001); Netlib is unchanged
at 89.61 → 89.52 ms. Other four smaller Gonum cases are inconclusive
before/after and remain about 1–41% slower than reference Netlib. No Netlib
before/after case changes significantly. These benchmarks include timed input
restoration and cgo calls, use each backend's native layout, and exclude
conversion and workspace setup. They are not Accelerate or OpenBLAS results.

### Post-change profile

A separate P1 thin-wide n=256 SVD profile, taken after timing completed,
contains 3.34 seconds of CPU samples. Original DotUnitary falls from 21.21%
flat / 26.67% cumulative to 2.10% / 2.69%. The new paired kernel accounts
for 3.29% flat / 4.19% cumulative. The enclosing SIMD DGEMM dispatch accounts
for 12.28% flat / 21.26% cumulative; cumulative percentages overlap and must
not be added.

Remaining costs include Dlasr at 36.53% cumulative, and DotInc at 10.18%
flat, all sampled DotInc calls coming through gemvN. Strided dot beneath
non-transposed GEMV is a measured follow-up, not an improvement delivered here.
An independent alpha-zero GEMM semantics concern is tracked separately as
Ergo A7PB45; this patch does not change that behavior.

## Validation

Passed full suites with Go 1.27.1 SIMD enabled and default Go 1.26.4, plus
native minimum Go 1.24.0 tests for affected f64, BLAS, LAPACK and mat packages.
Affected BLAS/LAPACK/mat tests also pass under safe, noasm, race instrumentation
and `GODEBUG=simd=0`. The latter changes portable SIMD emulation, not archsimd.
The final affected race run uses GOMAXPROCS=4 and `-p=2`; it completed without
failures. Native Netlib Dgesvd/Dbdsqr differential tests pass three consecutive
runs. New dot tests also pass against the unchanged baseline.

Persistent DGEMM regressions cover block boundaries through k=481, odd output
tails, both real transpose enums, alpha/beta combinations, offset and padded
storage, exact existing per-output dot arithmetic, cancellation, overflow,
subnormals, signed zero, infinities, NaNs and a scalar-tail FMA witness.
Alias tests reject active C/A or C/B overlap before mutation, including only
the odd final B row, and accept safe disjoint panels in shared backing storage.
Focused new tests pass normally and with race instrumentation.

Scoped generation changes only the expected single-precision dispatch file;
formatting, diff whitespace, import policy and copyright checks pass. Final
escape diagnostics confirm A/B/C do not escape. No tolerances were relaxed.

The first dot cohort was stopped after a background CPU burst; a replacement
may have briefly overlapped the old child process during shutdown. Both
`dot/` and `dot-final/` are diagnostic only. The clean `dot-accept/` run
confirmed the rejected prototype's leaf improvement, not production acceptance.
First paired-tile three-round screens and a stopped superseded race run are
also diagnostic only; final acceptance uses the release cohorts below.

## Reproduction and evidence

Local raw evidence is under `/tmp/gonum-dot-followup.BVSDAE` (temporary,
not checked in). Build separate binaries from the baseline with the common
harness and the candidate:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./internal/asm/f64 -o /tmp/f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./blas/gonum -o /tmp/blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o /tmp/lapack.test
```

Use the go-optimisation skill's `compare_benchmarks.py` with separate
`--baseline` and `--candidate`, a fresh `--output`, `--rounds 10`, and matching
`GOMAXPROCS` and `--cpu`. Analyze raw samples with benchstat
`golang.org/x/perf@v0.0.0-20260312031701-16a31bc5fbd0`.

Final evidence directories (each includes metadata, raw samples and summaries):

| Directory | Selector | P |
| --- | --- | ---: |
| release-dgemm | `^BenchmarkDgemmTransposedB$` | 1 |
| release-mat-p1 | `^BenchmarkFactorization$` | 1 |
| release-mat-p4 | `^BenchmarkFactorization$/(SVD\|QR\|LQ\|LU\|Cholesky)$/.*` | 4 |
| release-netlib | `^BenchmarkDgesvdNetlibKernels$` | 1 |

All use 100 ms per case and ten rounds. The profile is `release-svd.cpu`,
with selector `^BenchmarkFactorization$/SVD/kind=thin/shape=wide/n=256$`,
P1 and 3 s benchtime. It is not mixed into timing samples.

SHA256 of the exact measured binaries, built from the frozen source before
commit metadata was added:

```text
tile-base-blas.test d97eda38da3e3c40d43918a07660b8852f9d1bdbd06cc6ea3677a4aee5bbe458
release-blas.test   ee6cd27e61f6a2169d5c7560ef1d2fc422dcca053d44c3909eb9545575fe6683
base-mat.test      010b168791999b566b569b148789f8de0d87f6422d2015d5a18570e1f58d638d
tile2-mat.test     01ce97bb9dcf605ceae0060546001eedd18ce28a0c4bc2d706e7cb84209fabd0
base-lapack.test   57d5b60c468c1329277327ccf22a8668ca5f6b4ed5298e41999bfdf8654fb694
tile2-lapack.test  12dde1fca486352887b315fab12f384079f28578ae7f1348bef571e0b770dc24
```

The identical DGEMM benchmark source in both trees has SHA256
`26d72066a96b670aa95272d663d89a714fc2191d53916560b9c7882a1fd8d48a`.

Native Netlib differential checks:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib ./lapack/gonum -run '^Test(DgesvdNetlib.*|DbdsqrNetlib.*)$' -count=3
```

The native oracle is Homebrew reference LAPACK/BLAS, not Accelerate/OpenBLAS.
Homebrew resolves to `Cellar/lapack/3.12.1`; the runtime version test reports
ILAVER=3.12.0, with source review pinned to Reference-LAPACK v3.12.1.
