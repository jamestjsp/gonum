# GEMM zero-alpha prerequisite

## Change and correctness

Baseline: `f10531ed333d2d50a643f04732e28691648bee1d` on
`codex/arm64-simd-blas`. Dgemm and generated Sgemm previously scaled C but
continued multiplication when alpha was zero. Unused NaN/Inf operands could
therefore contaminate C. Return after scaling C when alpha == 0. Argument
validation, empty-output handling, and nonzero-alpha/zero-k arithmetic are
unchanged. This is a correctness repair, not a new SIMD kernel.

Reference: Reference-LAPACK v3.12.1, commit
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, BLAS/SRC/dgemm.f,
zero-alpha branch. The persistent tests failed before the repair and pass
after it. Coverage includes both real precisions, all nine transpose pairs,
k=0/1/17, signed-zero alpha, beta=0/1/-.5/NaN/+Inf, exceptional inputs,
padding, input preservation, and slice-validation order. Non-NaNs are checked
exactly including signed zero; NaNs are compared by classification.

The opt-in native oracle uses Homebrew's LP64 reference BLAS, not Accelerate.
Its row-major Fortran bridge is independently calibrated with exact finite
products for all transpose pairs and both precisions. No production cgo
dependency is introduced.

## Native measurement

Apple M1 Pro, darwin/arm64, Go 1.27.1, GOEXPERIMENT=simd, -pgo=off,
GOMAXPROCS=1. An independent detached baseline worktree received the identical
benchmark harness. Prebuilt binaries were run in alternating baseline/candidate
order, ten samples each, 100ms per sample. No concurrent builds, tests or
profiling ran during timing; ordinary desktop activity was not eliminated.
All cases had zero allocations. These are finite, beta-zero public GEMM
controls with A transposed; B's transpose is listed below.

| Routine / n / B | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| D / 4 / N | 43.10 ns | 20.77 ns | -51.82% |
| D / 4 / T | 40.46 ns | 20.70 ns | -48.85% |
| D / 32 / N | 3233.5 ns | 174.8 ns | -94.59% |
| D / 32 / T | 547.0 ns | 175.1 ns | -67.99% |
| D / 128 / N | 172.359 us | 1.234 us | -99.28% |
| D / 128 / T | 8.247 us | 1.291 us | -84.35% |
| S / 4 / N | 38.43 ns | 15.53 ns | -59.58% |
| S / 4 / T | 34.98 ns | 15.81 ns | -54.80% |
| S / 32 / N | 1726.0 ns | 119.0 ns | -93.11% |
| S / 32 / T | 493.8 ns | 117.5 ns | -76.20% |
| S / 128 / N | 87.2155 us | 0.8168 us | -99.06% |
| S / 128 / T | 7.845 us | 0.8273 us | -89.45% |

The table is alpha=0 only (all p<.001, n=10); it is not an application-speedup
claim. Ordinary alpha=1 controls showed a possible tiny-call cost and noisy
large TT results. A separate ten-sample, 300ms recheck measured D/n=4/B=N
at 111.3 -> 112.0 ns (+0.63%, p=.024) and D/n=4/B=T at 85.14 -> 85.39 ns
(+0.29%, p=.004). D/n=128 N and T were inconclusive (p=.481 and .631).
The small cost is retained for correct alpha-zero semantics. No AMD64 or
LAPACK end-to-end performance improvement is claimed.

Reproduce using BenchmarkDgemmZeroAlphaControl and
BenchmarkSgemmZeroAlphaControl in blas/gonum/gemm_zero_test.go. Benchmark
runner: go-optimisation/scripts/compare_benchmarks.py; analysis: benchstat.
Local raw runs and metadata: /tmp/gonum-gemm-zero.s94E7l/{compare,recheck}.
These temporary paths are evidence from this machine, not repository fixtures.
Baseline binary SHA256:
`cc8598aaa5c146355fca413bb693892e60b17573fe3441999a418dab46de999a`;
candidate:
`343fab1224f429644132929693a23cabd96c536c2c2b6bb5cb182774878567d2`.

## Validation

Passed full Go 1.27.1 SIMD `go test ./...`; native netlib-tagged BLAS suite;
combined safe/noasm/bounds BLAS, LAPACK and mat tests; default Go 1.26.4 and
minimum Go 1.24.0 affected-package tests; targeted GEMM race tests; BLAS vet;
and git diff --check. Sgemm was regenerated from Dgemm. Code inspection
confirms the new return precedes multiplication dispatch without changing
the numerical kernels.

## Next target

Trace left-side Dormlq through row-wise Dlarfb before optimizing TT GEMM.
Ordinary QR factorization and Dgelqf are not TT consumers. Measure the actual
rectangular update shapes, tails, alias fallback and caller-level impact;
then assess SYRK under Cholesky. Do not infer AMD64 promotion from ARM64 data.
