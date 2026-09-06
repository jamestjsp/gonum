# Level 1 follow-up: strided index scans and single-precision norms

Base: `792803f0ed5ebe046f74620a7d95840b7ff1c592`.
Candidate: `856a28593860b8ecaac50537134852c1dc8a7a6d`.
Native host: Apple M1 Pro, darwin/arm64.
Primary toolchain: Go 1.27.1, `GOEXPERIMENT=simd`, `GOMAXPROCS=1`.
Fallback toolchain: installed Go 1.26.4, without experimental SIMD.

This follows [the complete Level 1 reference comparison](RESULTS_LEVEL1_NETLIB.md).
It targets strided Idamax/Isamax, Snrm2, and Scnrm2, including the prior
n=256 Scnrm2 regression. It does not add an Accelerate backend or claim
AMD64 performance. There was no cross-compilation.

## Implementation and numerical constraints

The new norm algorithms are shared scalar Go, not new SIMD intrinsics.
They widen each float32 component to float64 before squaring and accumulate
four independent sums. Every finite float32 square is exact in float64:
the exponent range of nonzero squares is approximately -298 through 256.
Even the maximum addressable slice length cannot overflow the float64 sum.
This removes per-element scaling divisions and prevents intermediate
underflow without a magnitude-specific fast-path gate.

Summation still rounds; this is not a guarantee of correctly rounded output
for every possible vector. Tests compare against a 256-bit reference over
the actual rounded float32 inputs. A finite-input norm may legitimately
overflow when converted back to float32.

Snrm2 keeps real norm's NaN-over-Inf behavior through ordinary floating-point
propagation. Scnrm2 uses widened accumulation at n>=32 for both contiguous
and strided vectors. If the sum is NaN it retries the original scaled
recurrence, preserving complex norm's Inf-over-NaN behavior. Smaller complex
vectors keep their original recurrence.

Dznrm2's existing ARM64 SIMD path retains its original contiguous dispatch.
A separate generated strided hook enables the complex64 fast path; its
complex128 counterpart is an inlineable false-returning stub. This avoids
calling a rejected fast path in strided Dznrm2. The single-precision BLAS
sources remain generated from their owning masters.

Strided Idamax/Isamax screen four magnitudes at a time. Blocks that cannot
improve the current maximum skip index updates; other blocks retain ordered
strict comparisons, earliest ties, and existing first/later-NaN behavior.
Small strided scans retain a separate original loop. Sharing that loop with
the unrolled tail initially regressed small Isamax; repeated boundary
benchmarks caught it, and the separate loop removed that regression.

Native code inspection confirms scalar float32-to-float64 conversions,
independent double-precision multiply-add chains, and a final square root.
The contiguous real norm has no bounds-check calls or per-element divisions.
Strided kernels retain bounds checks. No assembly or unsafe memory access
is introduced by these new norm algorithms.

## Measurement method

All binaries were built before timing. Benchmarks are serialized, with six
samples per case and alternating base/change execution order. The native
backend remains Homebrew Reference-LAPACK libblas, not optimized vendor BLAS.
The existing batched harness amortizes CGo and reports time per BLAS call;
residual CGo and public argument-check costs are included.

The revision/native suite covers the four target routines plus Dznrm2 as a
dispatch guard, n=4/16/31/32/33/256/4096/65536, and increments 1 and 2.
Pattern tests additionally cover rare and frequent index updates and wider
strides. The persistent pattern harness also supports random data and
strides 3 and 257. Default-toolchain and real LAPACK consumer checks are
separate within-toolchain comparisons.

Final native/revision and index-pattern samples use 40ms; default-toolchain
samples use 30ms; Dgeev consumer samples use 60ms. The native comparison uses
the final complex dispatch recheck's 40ms samples in both toolchains. All use
the same inputs and batched-call normalization on each side. This desktop
machine is not CPU-isolated; medians and benchstat significance are reported,
not fastest observations.

These are legacy manually orchestrated measurements. The retained report does
not include self-contained runner metadata such as binary hashes, per-process
sample order, PGO state, or the complete runtime environment. The historical
numbers and qualifiers below are preserved as recorded; new acceptance work
should use the `go-optimisation` prebuilt-binary comparison runner and retain
its metadata and raw per-run output.

## Results

The primary sweep has 160 cases and six samples per case (960 timings
per revision), including both Gonum and native backends. After restoring
the original Dznrm2 dispatch, the two complex routines were retested with
six samples in both primary and default toolchains. The tables use those
final complex measurements, and the unchanged real/index kernels' primary
measurements; the two runs are not pooled.

Median Gonum time per call, Go 1.27.1 SIMD configuration:

| Routine | n | inc | Base | Change | Time reduction |
|---|---:|---:|---:|---:|---:|
| Snrm2 | 256 | 1 | 604.55 ns | 64.39 ns | 89.35% |
| Snrm2 | 4096 | 1 | 10.281 µs | 1.254 µs | 87.80% |
| Scnrm2 | 256 | 1 | 1073.0 ns | 145.2 ns | 86.47% |
| Scnrm2 | 4096 | 1 | 17.888 µs | 2.536 µs | 85.82% |
| Idamax | 4096 | 2 | 5.093 µs | 1.509 µs | 70.38% |
| Isamax | 4096 | 2 | 5.100 µs | 1.837 µs | 63.98% |

All differences above have p=0.002, n=6. Snrm2 at n=4096, inc=2 improved
87.61%; Scnrm2 improved 85.95%. The old n=256 Scnrm2 regression is eliminated:
the new time is also far below the approximately 940 ns recorded before
the previous patch, though the statistical comparison here uses the
immediate base revision, not that older measurement.

### Compared with native Reference BLAS

At n=4096, the final norms take roughly one quarter of reference BLAS time.
This is not an Accelerate or OpenBLAS comparison.

| Routine | Gonum inc=1 (µs) | Netlib inc=1 (µs) | Go/native inc=1 | Go/native inc=2 |
|---|---:|---:|---:|---:|
| Snrm2 | 1.254 | 5.167 | 0.24 | 0.25 |
| Scnrm2 | 2.536 | 10.350 | 0.25 | 0.24 |
| Idamax | 1.454 | 1.386 | 1.05 | 1.07 |
| Isamax | 1.778 | 1.392 | 1.28 | 1.33 |
| Dznrm2 | 6.319 | 10.363 | 0.61 | 1.73 |

A ratio below 1 favors Gonum. The strided index gap is substantially smaller,
but reference BLAS still leads at n=4096. At n=256, inc=2, Idamax takes
100.2 ns versus reference's 108.1 ns. Thus crossing the reference is
size-dependent, not an across-the-board result.

### Patterns, fallback toolchain, and consumers

At n=4096, inc=2, rare-update index inputs improve 70.48% for float64 and
64.15% for float32; monotone inputs still improve 19.84% and 22.06%.
For inc=17, gains range from 6.40% to 48.82%; for inc=257, from 8.51% to
47.23%. These are specific measured strides, not a guarantee for arbitrary
memory layouts. Tiny Isamax's initial 25–43% regression was removed before
acceptance.

With Go 1.26.4 at n=4096, Snrm2 improves 75.86% (inc=1), Scnrm2 75.05%
(inc=1), strided Idamax 68.29%, and strided Isamax 59.34% (both inc=2).
The gains do not require experimental SIMD.

Dgeev on AntisymRandom50/100 and Circulant50/100 shows no practically
meaningful overall change: two cases are statistically unchanged, and two
improve only 0.24–0.36%. This is not evidence of a large eigenvalue or SVD
speedup. Gonum's LAPACK implementation is double precision, so these
single-precision norm gains should not be attributed to its real SVD.

### Remaining overheads and limits

The final code is not faster in every tiny case. Scnrm2 at n=4 adds
approximately 0.3–0.5 ns in the SIMD build and 0.3–0.7 ns in the default
build (up to 5.47%). Small default Dznrm2 strided cases add about 0.3–0.7 ns.
The initial 7.41% contiguous default Dznrm2 regression is resolved:
11.90 versus 11.94 ns at n=4 is statistically unchanged (p=0.725).
Other small index/unchanged unitary timing differences are below 1%.

No Go allocations were observed in these kernel benchmarks; counters are per
outer batch and do not measure native heap allocations. Larger
single-precision gains, supported fallback behavior, and numerical tests
justify retaining the changes with these small overheads documented.
Further work could target the remaining index gap or optimized vendor BLAS,
but neither is claimed complete here.

## Reproduction

Build matching binaries from the base and candidate with the same benchmark
files, then alternate their execution. For example:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -tags netlib \
  -o /tmp/level1-followup.test ./blas/gonum
GOMAXPROCS=1 /tmp/level1-followup.test -test.run '^$' \
  -test.bench '^BenchmarkLevel1Netlib$/routine=(Snrm2|Scnrm2|Idamax|Isamax|Dznrm2)$' \
  -test.benchtime=40ms -test.count=6 > level1-followup.txt
benchstat -col /implementation level1-followup.txt
```

The single-binary `-test.count=6` command above reproduces the within-binary
Gonum-versus-native sweep. It does not reproduce the base-versus-candidate
alternation used for the revision comparison. For a new revision comparison,
build matching binaries at both named commits with the same harness and use
`go-optimisation/scripts/compare_benchmarks.py` to alternate them and retain
binary hashes, run order, environment metadata, and raw samples.

Correctness checks include high-precision magnitude/overflow cases, zero,
NaN/Inf priority, n=31/32/33 and 255/256/257 boundaries, positive strides,
padding sentinels, index ties, offsets, and monotone/random inputs.

Validation passed: full Go 1.27.1 SIMD and installed Go 1.26.4 suites;
three native differential/ABI repetitions; affected safe/noasm packages;
BLAS/f32 bounds checks; focused race tests; generated-source inspection;
formatting, import policy, copyright, and whitespace checks. No tolerance
in an existing conformance test was relaxed.
