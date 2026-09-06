# Level 1 BLAS versus native Reference BLAS

## Scope and method

Base: `b9fb1114efd1f786fc0166a87fe6ea325aebe9e6`.
Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83).
Primary toolchain: Go 1.27.1 with `GOEXPERIMENT=simd` and `GOMAXPROCS=1`.
Fallback comparison: installed Go 1.26.4 without the experiment.
No AMD64 execution or cross-compilation was performed.

The comparison covers all 46 methods of Gonum's four Level 1 interfaces:
14 float32, 12 float64, 10 complex64, and 10 complex128 routines. The manifest
checks benchmark names against those interfaces, including mixed-precision dot
products and scalar rotation generators.

The native backend is Homebrew Reference-LAPACK's `libblas`, linked directly
through its Fortran ABI. `brew list --versions lapack` reports 3.12.1; the
linked dylib reports current version 3.12.0. This is reference BLAS, **not**
OpenBLAS, Accelerate, or another optimized vendor implementation. The bridge
is optional under `netlib && darwin && cgo` and adds no production CGo dependency.

Vector cases use n=16, 256, 4096, and 65536 with increments 1 and 2. The five
directly changed dispatch routines additionally cover n=4, 31, 32, and 33.
Scalar rotation generators use one fixed finite input tuple, not a vector-size
sweep. Independent index-pattern benchmarks include increasing and decreasing
data so the scan is not tuned only for rare maximum updates.

Each native call batches 32 identical kernel invocations, or four for n=65536,
to amortize the CGo boundary. Gonum executes the same number of public BLAS
calls from Go. Reported ns/op is normalized per BLAS call; allocation counters
remain per batch, and native heap allocations are not captured by Go counters.
The residual CGo cost and Go/public-call validation costs are still included;
these are not raw instruction-throughput or single-CGo-call latency numbers.

Both implementations start with the same seeded, nonzero, bounded data.
Scaling uses -1 or i, rotations use quarter turns, and AXPY alternates alpha
and -alpha. No timed input restoration or allocation is needed. Finite-state
checks follow timing. Reductions retain checksums to keep calls observable.

Final measurements use six samples per case. The all-routine native sweep
uses 40ms samples; focused base/change comparisons use 50ms and alternate
revision order. Reflector consumer samples use 100ms, and default-toolchain
checks use 30ms. All timed processes are serialized, with no concurrent builds
or tests. This desktop host is not CPU-isolated; statistical results are
interpreted with benchstat rather than selecting a fastest sample.

## Measured changes against the base

Median time per call at n=4096, inc=1, comparing Gonum to Gonum with the
same Go 1.27.1 SIMD toolchain and harness:

| Routine | Base (µs) | Change (µs) | Time reduction |
|---|---:|---:|---:|
| Snrm2 | 12.06 | 10.27 | 14.80% |
| Isamax | 9.859 | 1.772 | 82.03% |
| Idamax | 5.071 | 1.454 | 71.34% |
| Scnrm2 | 26.41 | 17.87 | 32.32% |
| Scasum | 12.0465 | 0.4737 | 96.07% |
| Icamax | 26.572 | 5.065 | 80.94% |
| Dznrm2 | 18.023 | 6.302 | 65.04% |
| Dzasum | 3.777 | 0.9511 | 74.82% |

All listed differences have p=0.002 with six samples. Several float32 base
cases were variable: Scasum's base interval was ±45%, Icamax ±20%, Isamax
±17%, and Scnrm2 ±15%. Do not interpret their percentages as universal
speedups. At n=65536, contiguous Scasum was more stable (about ±1%):
544.032 to 7.666 µs, a 98.59% time reduction. Input length and branch prediction
strongly affect the old branch-heavy float32 absolute-value implementation.

Index-pattern checks guard against a rare-update-only optimization:
Idamax at n=4096 improved 71.49% for decreasing data and was statistically
unchanged for increasing data. Both patterns at n=15, 16, and 17 were
statistically unchanged after retaining the small scalar loop.

The complex LAPACK consumer Zlarfg improved from 1.3365 to 0.6209 µs
at n=256 (53.54%), and 20.737 to 9.102 µs at n=4096 (56.11%), both inc=1.
Its input-copy cost is included on both revisions, with zero allocations.
The n=16/32 cases and all tested inc=2 cases were statistically unchanged.

A real LAPACK consumer check used six interleaved 50ms samples of Dgeev on
AntisymRandom50/100 and Circulant50/100. All four were statistically unchanged
(p=0.132 to 0.589), providing no evidence of a real-eigensolver speedup or
regression from this patch on those cases.

The shared scalar changes also help without experimental SIMD. With Go
1.26.4 at n=4096, inc=1, Idamax improved 70.96%, Isamax 86.68%, Scnrm2
65.41%, and Icamax 80.14%. Dznrm2 and Dzasum were statistically unchanged
in that configuration, where their new SIMD paths are disabled. These are
separate within-toolchain comparisons, not evidence that one Go version
is generally faster than another.

### Regressions and remaining gaps

This patch is not uniformly faster. With Go 1.27.1, Scnrm2 at n=256
regressed from 939.7 to 1068.0 ns for inc=1 (+13.66%), and 940.2 to
1077.0 ns for inc=2 (+14.54%), both p=0.002. The shared absolute-value
change is retained for its substantial larger-vector gains; this medium-size
regression remains a follow-up target. Tiny Dzasum (n=4, inc=1) added
0.293 ns (+5.04%). Default-toolchain Dznrm2 at n=4/16 added about 0.3 ns.

Reference BLAS still leads on single-precision norms, complex index scans,
many strided update/copy routines, and some contiguous double/complex128
updates. In particular, strided Idamax/Isamax remain about 3.6 times the
reference time at n=4096. Scalar rotation-generator measurements use only
one input tuple: Drotg's roughly 10-times gap is a lead for broader
input-distribution and numerical-contract investigation, not sufficient
evidence to replace its robust algorithm.

Next investigations should prioritize strided index scans and updates,
a robust single-precision norm fast path that also resolves the n=256
regression, and complex128 updates. Optimized vendor BLAS and native AMD64
need separate measurements; this reference-library comparison establishes
neither vendor parity nor AMD64 performance.

## Complete native comparison

The final sweep contains 760 cases and six samples each (4560 timings).
The table below summarizes every routine at n=4096, or n=1 for scalar
generators. Times are medians. A Go/native ratio below 1 means lower Gonum
time; above 1 means lower reference time. Rounded ratios near 1 are not
claims of statistical superiority. Contiguous Saxpy, Dscal, and Cswap,
and strided Sdsdot, Ddot, and Zswap were statistically indistinguishable.
Zcopy measurements were notably noisy (up to ±58% for strided Gonum).

| Routine | Gonum, inc=1 (µs) | Netlib, inc=1 (µs) | Go/native, inc=1 | Go/native, inc=2 |
|---|---:|---:|---:|---:|
| Sdsdot | 1.1520 | 5.1090 | 0.23 | 1.00 |
| Dsdot | 1.1490 | 5.1020 | 0.23 | 0.99 |
| Sdot | 1.0850 | 5.0500 | 0.21 | 0.99 |
| Snrm2 | 10.2790 | 5.1620 | 1.99 | 1.99 |
| Sasum | 0.2415 | 3.8050 | 0.06 | 0.98 |
| Isamax | 1.7730 | 1.3880 | 1.28 | 3.64 |
| Sswap | 0.4508 | 1.0012 | 0.45 | 1.39 |
| Scopy | 0.2079 | 0.3847 | 0.54 | 1.44 |
| Saxpy | 0.3249 | 0.3329 | 0.98 | 1.40 |
| Srotg | 0.0127 | 0.0051 | 2.50 | — |
| Srotmg | 0.0078 | 0.0069 | 1.13 | — |
| Srot | 0.6512 | 1.8245 | 0.36 | 1.27 |
| Srotm | 0.6538 | 1.9870 | 0.33 | 1.53 |
| Sscal | 0.2425 | 0.3312 | 0.73 | 1.09 |
| Ddot | 2.2470 | 5.1310 | 0.44 | 1.00 |
| Dnrm2 | 3.1600 | 5.1810 | 0.61 | 0.99 |
| Dasum | 0.4736 | 3.7890 | 0.12 | 0.98 |
| Idamax | 1.4530 | 1.3920 | 1.04 | 3.58 |
| Dswap | 0.8885 | 1.4355 | 0.62 | 1.19 |
| Dcopy | 0.3994 | 0.5378 | 0.74 | 1.45 |
| Daxpy | 0.7307 | 0.4995 | 1.46 | 1.40 |
| Drotg | 0.0489 | 0.0049 | 9.92 | — |
| Drotmg | 0.0081 | 0.0073 | 1.12 | — |
| Drot | 1.2920 | 1.8220 | 0.71 | 1.26 |
| Drotm | 1.2880 | 1.9900 | 0.65 | 1.51 |
| Dscal | 0.5489 | 0.5112 | 1.07 | 1.10 |
| Cdotu | 1.7450 | 3.8990 | 0.45 | 1.10 |
| Cdotc | 1.8080 | 3.9020 | 0.46 | 1.14 |
| Scnrm2 | 17.8700 | 10.3300 | 1.73 | 1.73 |
| Scasum | 0.4737 | 7.7185 | 0.06 | 0.49 |
| Icamax | 5.0630 | 3.9370 | 1.29 | 1.29 |
| Cswap | 1.5060 | 1.5110 | 1.00 | 1.18 |
| Ccopy | 0.4034 | 1.3130 | 0.31 | 1.42 |
| Caxpy | 1.3680 | 2.3270 | 0.59 | 1.59 |
| Cscal | 0.9692 | 1.6895 | 0.57 | 1.53 |
| Csscal | 0.7212 | 1.3185 | 0.55 | 1.45 |
| Zdotu | 3.0330 | 3.9050 | 0.78 | 0.98 |
| Zdotc | 3.1670 | 3.9170 | 0.81 | 0.98 |
| Dznrm2 | 6.2940 | 10.3270 | 0.61 | 1.73 |
| Dzasum | 0.9515 | 3.9110 | 0.24 | 0.96 |
| Izamax | 5.0550 | 3.9210 | 1.29 | 1.30 |
| Zswap | 2.7210 | 1.8240 | 1.49 | 1.00 |
| Zcopy | 0.8049 | 1.3335 | 0.60 | 1.13 |
| Zaxpy | 2.8230 | 2.5340 | 1.11 | 1.28 |
| Zscal | 1.9190 | 1.6940 | 1.13 | 1.21 |
| Zdscal | 1.7190 | 1.3190 | 1.30 | 1.52 |

## Implementation

- `math32.Abs` now clears the floating-point sign bit instead of branching on
  sign and zero. It preserves finite magnitudes, returns +0 and +Inf, and keeps
  NaN payload bits while clearing the NaN sign. The documented NaN result does
  not promise a sign. This shared scalar change benefits several float32 BLAS
  routines without changing their reduction algorithms.
- `Idamax` and generated `Isamax` screen four magnitudes together for n>=32.
  A block whose values cannot exceed the current maximum skips index updates;
  other blocks retain ordered strict comparisons. Earliest ties and existing
  first/later-NaN behavior are preserved. Small and strided scans keep their
  original traversal. Generated ARM64 code contains explicit screening
  branches, not a serial conditional-select chain; bounds checks remain.
- Contiguous `Dzasum` and `Scasum` reuse the existing real SIMD absolute-sum
  kernels over interleaved real/imaginary components. Short vectors retain
  the original loop. Promotion is limited to the measured ARM64 SIMD build;
  other builds retain scalar implementations.
- Contiguous `Dznrm2` reuses the compensated real SIMD norm for ordinary
  magnitudes at n>=32. Short vectors, conservative first-value exclusions,
  and non-finite results use the original component-wise scaled recurrence.
  In particular, complex norm's existing infinity-over-NaN behavior is not
  replaced with the real helper's NaN-first behavior. `Scnrm2` retains its
  scaled algorithm and benefits only from the absolute-value helper change.

These are a mix of shared Go improvements and reuse of existing SIMD kernels,
not 46 new SIMD implementations. Single-precision sources are regenerated
from their owning masters; no public interface or numerical tolerance changed.

## Correctness and reproduction

Persistent tests cover all 46 native methods, complete mutator backing slices,
padding, empty and tail cases, supported positive/negative increments, index
ties, and rotation outputs. Native batch tests validate complex return ABI,
reduction checksums, AXPY sign alternation, and constant generator inputs.
Changed kernels additionally cover vector boundaries, finite high-precision
norm references, subnormal/huge magnitudes, signed zero, and non-finite priority.

One pre-existing difference is explicitly retained: with n=0, Gonum's Sdsdot
returns 0 while this Reference BLAS returns alpha. The differential test checks
each behavior separately. This pass is not a claim of full BLAS semantic parity,
and undefined native parameter slots are not compared as meaningful results.

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -tags netlib \
  ./blas/gonum ./blas/gonum/internal/netlib -run 'Test(Level1Netlib|ComplexDotABI|Batched|IAMAX)' -count=3

GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -tags netlib \
  -o /tmp/level1-netlib.test ./blas/gonum
GOMAXPROCS=1 /tmp/level1-netlib.test -test.run '^$' \
  -test.bench '^BenchmarkLevel1Netlib$' -test.benchtime=40ms -test.count=6 > level1.txt
benchstat -col /implementation level1.txt
```

For revision comparisons, apply the identical benchmark harness to the base
checkout, prebuild both binaries, and alternate their execution. Compare
Gonum to Gonum to measure this patch; compare Gonum to native within the same
harness to measure the remaining reference gap.

Validation includes full SIMD/default suites, affected `safe`/`noasm` suites,
BLAS/kernel `bounds`, focused race tests, native differential tests, generated
source inspection, formatting, imports, copyright, and diff checks.
