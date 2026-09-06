# SYRK and Cholesky

## Baseline and scope

Baseline: `ba4d7ed2d3bd8725817e02cebc8d1534e0a7d761`, after the TT GEMM
milestone. Apple M1 Pro, darwin/arm64, Go 1.27.1, GOEXPERIMENT=simd,
-pgo=off. Production routing remains under the existing ARM64 experiment
gate; no AMD64 promotion or native AMD64 performance claim is made.

Public Cholesky calls Dpotrf. Its upper path uses transposed SYRK on diagonal
blocks; the lower path uses NoTrans and is not redirected by this change.
A three-second profile of mat Cholesky n=256 attributed 16.72% cumulative
sampled CPU to Dsyrk. Triangular solves in condition estimation accounted for
more CPU, so a large SYRK leaf gain cannot imply a large factorization gain.

## Retained implementation

For disjoint Trans/ConjTrans calls with n,k>=16, keep four portable SIMD
vectors of each active triangular row in registers, then one-vector and scalar
tails. Traverse k in increasing 64-term blocks across output rows. Beta is
applied exactly once, in the first block; each output retains increasing-k
accumulation, alpha-before-term scaling, and the exact-zero skip.

The existing row-interval disjoint check runs before mutation. Real C/A
overlap falls back to the original path. A shares storage with C in LAPACK
panels, so disjoint active row ranges remain eligible. A conservative check
against the full C rectangle may reject overlap confined to its inactive
triangle; that is a performance limitation, not a change in results.

The double-precision source generates single precision. Native simd128
disassembly confirms four accumulators in V3-V6, four VFMLA operations per
nonzero coefficient, and no inner-loop calls or accumulator spills. Bounds
checks remain. Stores/reloads between k blocks are intentional. The native
frame is 208 bytes; the generic emulated variant is not native codegen evidence.

Also repaired transposed-Lower beta=0 behavior in both precisions: the old
path multiplied C by zero, contaminating output with unused NaN/Inf C values.
It now clears the active triangle. Persistent tests fail on the unchanged
baseline and pass against Reference BLAS and the repair. Scalar argument and
slice-validation order, alpha-zero handling, and NoTrans arithmetic remain
unchanged.

## Rejected experiments

1. Reusing public GEMM for rectangular off-diagonal blocks: smaller D cases
   regressed 11-33%; S cases regressed 16-67%. Removed this candidate despite
   its modest Cholesky improvement.
2. Direct SIMD without k blocking: compact leaves improved 8-45%, but actual
   lda=ldc=256 panels with k=128/192 regressed 5-7%, and Cholesky256 was
   inconclusive. The final 64-term blocking removes those measured regressions.

Better A-panel reuse motivated k blocking. Cache misses were not measured
with hardware counters; timing and loop structure support this explanation,
not a claim of measured cache-hit rates.

## Final measurements

Independent detached baseline with identical persistent harnesses, prebuilt
binaries and ten alternating samples per case. No concurrent builds, tests
or profiles during timing; ordinary desktop activity was not eliminated.
SYRK leaves use 100ms/sample; actual-panel controls use 200ms. The final
single-worker caller confirmation uses 500ms/sample; four-worker callers
use 200ms. All SYRK cases allocate zero bytes. Cholesky remains at 72 B/op,
three allocations, on both revisions.

| Public Cholesky | Workers | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: | ---: |
| n=32 | 1 | 16.58 us | 16.56 us | inconclusive, p=.075 |
| n=128 | 1 | 252.4 us | 250.1 us | -0.92%, p<.001 |
| n=256 | 1 | 1.129 ms | 1.112 ms | -1.50%, p<.001 |
| n=32 | 4 | 16.56 us | 16.55 us | inconclusive, p=.271 |
| n=128 | 4 | 252.4 us | 250.1 us | -0.94%, p<.001 |
| n=256 | 4 | 1.129 ms | 1.112 ms | -1.44%, p<.001 |

These longer final results supersede the noisier initial 1.4-2.5% caller gains.
The actual shared-storage n=64, lda=ldc=256, alpha=-1, beta=1 panel cases
improve 8.10%, 7.31%, and 7.49% at k=64/128/192. Matching independent-storage
controls improve 6.86-7.71%.

Final compact/padded Trans leaves, both triangles and precisions:

- n=16/17,k=16: about 41-44% less time.
- n=64,k=16: D improves 5.5-10%; S improves 16.8-22.2%.
- n=64,k=256: D improves 7.5-10.8%; S improves 12.9-16.8%.
- n=4/15 fallback controls are mostly inconclusive. D Lower n=4 costs
  0.84-1.44% more (about 3-5 ns), retained with the beta-zero correctness fix.
- NoTrans n=64,k=256 controls are inconclusive in both precisions and triangles.

Do not interpret a leaf geomean as application throughput or claim universal
improvement. The tested dimensions and worker counts are the evidence boundary.

## Validation and reproduction

Persistent tests cover triangle/transpose combinations, offset guards,
padding, real overlap rejection, shared backing, vector tails, cache boundaries
63/64/65/127/128/129, signed zero, skipped NaN/Inf operands, exceptional beta,
alpha Inf with zero operands, and one-time beta scaling. Sequential-reference
comparisons are exact apart from NaN classification. Dyadic native differential
tests require exact finite numerical equality; signed-zero semantics also have
dedicated tests. Zero-beta tests include exceptional C across vector/cache blocks.

Reference: Reference-LAPACK v3.12.1 commit
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, BLAS/SRC/dsyrk.f.
The opt-in Darwin bridge calls Homebrew LP64 Reference BLAS, not Accelerate;
row-major adaptation flips the triangle and transpose. Native SYRK plus
Dpotrf/Dpotrs differential tests passed three runs.

Passed full Go 1.27.1 SIMD suite; default Go 1.26.4 and minimum Go 1.24.0
affected packages; combined safe/noasm/bounds affected packages; focused SYRK
race tests; GODEBUG=simd=0 correctness; BLAS/LAPACK vet; formatting and diff
checks. No cross-compilation was requested.

Benchmarks: BenchmarkD/SsyrkSIMDShapes, BenchmarkDsyrkCholeskyPanel, and
BenchmarkFactorization/Cholesky. Runner: go-optimisation/scripts/compare_benchmarks.py;
analysis: benchstat. Local evidence is under `/tmp/gonum-syrk.7Cnqnn/`:
`final-boundaries`, `final-wide`, `panel-blocked`, `final-consumer`,
`four-worker-consumer`, profiles, final-codegen.txt, binary hashes and metadata.
Rejected cohorts are retained there too. Temporary evidence is not a repository
fixture; the benchmarks/tests above are persistent and runnable.

## Remaining opportunities

This closes the measured TT-GEMM/SYRK goal, not every possible BLAS optimization.
Native AMD64 measurement is still required before changing its production
dispatch. Bounds-check cost, conservative inactive-triangle alias rejection,
and Cholesky condition-estimation triangular solves remain possible future
targets. Recheck vector widths, APIs, numerical gates and native crossovers on
future Go/SIMD releases before changing the experiment guards or cache cutoff.
