# AMD64 patch integration, 2026-09-04

The two Samsung-drive patches matched base
`e925697fb16c99cbbe52cf4a489f1e6f503923db`; every bundle SHA256 checksum passed.
They are preserved separately as `d224925a` (AMD64 assembly correctness) and
`33a826fc` (SIMD tuning), followed by the review corrections described here.

The [supplied AMD64 report](RESULTS_AMD64.md) measures the imported tuning,
not these subsequent corrections. Its raw samples were not supplied.
Production dispatch is unchanged: the assembly correctness repair is active on
AMD64, but candidate speedups do not imply BLAS, SVD, or application speedups.

## Review corrections

- Added runnable complex64/complex128 dot-product regressions: separate component
  sums overflowed before cancellation, returning a NaN component instead of zero.
  Non-finite unitary results now retry sequential complex multiplication and
  summation, including contiguous increment calls.
- Retained direct portable widening/staging for targets without the AMD64 widening
  leaf. The imported fallback slowed ARM64 Ddot by 52–157%; final Ddot timings
  are statistically inconclusive versus base.
- Reused the portable strided SIMD core for complex128 unitary dot and scale when
  the AMD64 leaf is unavailable. This removes the measured shuffle-fallback
  regressions without duplicating an ARM64 candidate implementation.
- Named matrix benchmark baselines `current`, since ARM64 is not using AMD64 ASM.

## Native ARM64 measurements

Apple M1 Pro, macOS 26.6.2, darwin/arm64, Go 1.27.1,
`GOEXPERIMENT=simd`, 128-bit vectors, `GOMAXPROCS=1`.
Baseline and final candidate binaries were built before timing; the same unchanged
comparison harness ran six 50ms samples per case, reversing binary order each
round. No concurrent test or compilation workload ran during these measurements.
CPU affinity was not pinned. All 1,368 measurements across 114 cases and two
revisions reported zero allocations.

Selected medians compare final SIMD candidates with base SIMD, not assembly.
Every table improvement has benchstat p=0.002, n=6.

| Kernel/size | Base ns/op | Final ns/op | Time change |
|---|---:|---:|---:|
| f64/L2NormUnitary/4096 | 29271 | 1024 | -96.50% |
| f64/L2DistanceUnitary/4096 | 29239 | 1462 | -95.00% |
| f64/L2NormInc/4096 | 29317 | 3826 | -86.95% |
| f64/GemvN/64 | 1598 | 844.3 | -47.17% |
| f64/GemvT/64 | 2057 | 843.5 | -58.99% |
| f64/Ger/64 | 2056 | 921.3 | -55.19% |
| f32/Ger/64 | 1222 | 495.8 | -59.42% |
| c64/AxpyUnitary/4096 | 8608 | 2792 | -67.56% |
| c64/DotcUnitary/4096 | 5740 | 2796 | -51.30% |
| c128/DotcUnitary/4096 | 8915 | 6700 | -24.85% |

The equal-weight kernel geomean fell 25.77%, not an application speedup.
Per-case significance uses alpha=0.05 without multiple-comparison correction.
Remaining short-call differences include complex AXPY dependency-guard costs
of roughly 2–3% and smaller changes in unchanged functions.

An unresolved regression remains in the unchanged f64 CumSum candidate:
26.71 to 37.61 ns/op at n=31 (+40.76%) and 3110 to 4461 ns/op at n=4096
(+43.45%), confirmed in an isolated six-round 200ms rerun (p=0.002).
Its source and normalized SIMD128 instruction stream are unchanged. Binary
layout is a hypothesis, not an established cause; this must be resolved before
promoting that candidate. The current production routine is unaffected.

## Validation

- Full default Go 1.26.4 and SIMD-enabled Go 1.27.1 suites passed.
- Affected native ARM64 packages passed with hardware and emulated SIMD.
- Race, safe, and noasm tests passed for internal ASM and BLAS.
- AMD64 numeric and comparison binaries passed under Rosetta with
  `GODEBUG=simd=0` and `simd=128`, including the assembly alignment regressions.
  This is translated correctness evidence, not native AMD64 performance evidence.
- Generated code contains ARM64 vector FMUL/FADD in the norm fast path and AMD64
  VCVTPS2PD with vector arithmetic in the widened dot leaf. Bounds checks remain
  in generated loops; further compiler optimization is still possible.
- Formatting/import grouping, repository import/copyright policies, and diff
  checks passed.

Local integration logs and raw samples are in
`/tmp/gonum-patches.C7GiDm/`: `checked-base.txt`, `checked-final.txt`,
`arm64-final.txt`, `scan-{base,final}.txt`, and test logs.
These temporary files are not committed. The [comparison instructions](README.md)
provide the runnable kernel harness. Remeasure the integrated revision on native
AMD64, including AVX256 as well as AVX512, before changing dispatch.
