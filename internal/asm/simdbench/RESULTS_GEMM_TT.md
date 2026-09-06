# TT GEMM register blocking

## Scope and mechanism

Baseline: `fec3d70fd8b8d6025a10571da104773da32b5e06`.
Native host: Apple M1 Pro, darwin/arm64, Go 1.27.1,
GOEXPERIMENT=simd, -pgo=off. Production activation remains behind the existing
ARM64 experiment gate; there is no AMD64 performance claim or dispatch change.

Left-side Dormlq calls row-wise Dlarfb, whose two TT products have shapes
`(n,ib,m-i-ib)` and `(m-i-ib,n,ib)`. Ordinary QR factorization and Dgelqf
are not the relevant TT consumers. A three-second baseline profile of
Dormlq(Left), m=256/n=64/k=160, attributed 86.16% of sampled CPU cumulatively
to dgemmSerialTransTrans, with 70.93% flat in its inlined AxpyInc.

The candidate retains a 2x4 output tile across the inner dimension. It is
scalar register blocking, not a new SIMD intrinsic implementation. Both real
precisions are generated from the same source. Each output keeps increasing-k
accumulation order, alpha scaling before each term, and the exact-zero skip.
Active C/A or C/B overlap rejects the helper; tails use the previous kernel.
The screening boundary is m>=2, n>=4, k>=16, checked before calling the helper.

Native disassembly shows eight accumulators in F1-F8 and scalar FMADDD updates,
with no accumulator spills or calls in the full-tile inner loop. Bounds checks
remain for A and B accesses: their removal is a future measured hypothesis,
not a claimed result here.

## Measurement

Independent detached baseline worktree with identical persistent benchmark
harnesses; prebuilt binaries; ten alternating baseline/candidate samples.
No concurrent builds, tests or profiles ran during acceptance timing. Normal
desktop activity remained. Main runs use GOMAXPROCS=1; public TT dispatch was
also checked at four workers. All accepted timing cohorts reported zero
allocations. Durations: leaf 100ms/sample, caller and non-TT controls 200ms/sample.

Final caller results (time, not throughput; p<.001, n=10 each):

| Left Dormlq m/n/k | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 128/16/95 | 206.3 us | 149.1 us | -27.71% |
| 256/32/127 | 1081.6 us | 664.8 us | -38.54% |
| 256/64/160 | 2.531 ms | 1.356 ms | -46.45% |

The caller fixture factors its input outside timing, alternates Q/Q-transpose
applications, resets every 16 calls outside timing, and checks finite scaled
Frobenius norm preservation. It includes full and partial reflector blocks.

Final leaf controls include both precisions and padding 0/7. The 2x4x16
activation case improves 53-55%; 16x32x224 and 224x16x32 improve 53-55%;
224x64x32 improves 54-62%. At four workers, 224x64x32 improves 35.6-37.2%.
Before hoisting the guard, the broader shape cohort also showed 33-71% gains
in square, rectangular and tail-heavy cases; those numbers are exploratory,
not a substitute for the final guarded-candidate controls.

Costs and limitations are retained:

- The first version regressed tiny fallback calls about 3%; the hoisted guard
  removed most of that. Final 4x4x4 controls were inconclusive in both
  precisions. A separate longer recheck found a 0.47% Sgemm cost in one case.
- The awkward 3x5x17 tile is 1.4-2.4% slower (roughly 3-6 ns), in both
  precisions. It spends most work in the retained tails. This small absolute
  cost is retained alongside the much larger measured caller gains.
- One float32 k=15 padded fallback control is 0.93% slower; other k=15
  controls were inconclusive. This is not a claim of universal improvement.
- Existing medium NN, TN and NT controls were inconclusive: 147.1->147.2 us,
  147.0->147.2 us, and 115.8->115.8 us. Those legacy controls use random
  finite inputs and repeated additive updates, unlike the reset caller fixture.
- No application-wide speedup, new parallel cutoff, or native AMD64 result
  is inferred from these cohorts.

Reproduction entry points: BenchmarkDgemmTT, BenchmarkSgemmTT, and
BenchmarkDormlqLeft. Runner: go-optimisation/scripts/compare_benchmarks.py;
analysis: benchstat. Local raw samples, binary hashes, codegen, profile and
metadata are under `/tmp/gonum-tt.yWX4lf/`. Final cohorts: `final-leaf`,
`final-consumer`, `four-workers`, `non-tt-controls`; `tiny-recheck` is the
longer tiny-call check. These temporary artifacts are not repository fixtures.

## Correctness and validation

Persistent tests cover both precisions, tile boundaries and tails, offset
slices and padding, shared disjoint backing, active overlap rejection, exact
scalar-order results, exceptional arithmetic, overflow/cancellation, signed
zero, and public dispatch. Independent native Netlib tests exercise active
tiles and transpose/conjugate-transpose combinations with exact binary inputs.
An exact-cancellation signed-zero difference against Netlib was reproduced on
the unchanged baseline: Netlib scales after reduction while Gonum scales each
term. The oracle allows only this zero-sign difference in active outputs;
nonzero outputs and padding remain exact. Scalar-order tests still check zero
signs against Gonum's established behavior.

Passed full Go 1.27.1 SIMD suite, native Netlib BLAS suite, focused GEMM race
tests, default-Go affected packages, combined safe/noasm/bounds affected
packages, BLAS/testlapack vet, formatting and diff checks. Generated Sgemm
source was reviewed together with its generator mappings.
Minimum Go 1.24.0 affected-package tests and focused portable-emulation
tests (`GODEBUG=simd=0`) also passed. No cross-compilation was requested.

Next: profile SYRK under Cholesky from the committed TT revision. Remaining
TT opportunities include tail cost and bounds-check overhead; do not combine
them with this milestone without a new independent comparison.
