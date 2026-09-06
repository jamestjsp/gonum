# Bottom-up SVD rotation follow-up

## Scope and mechanism

Continue from `511627624ef848acf7faba7e253cb897471b47b5`, using
`go-optimisation`, `gonum-simd` and dedicated Sol numerical/code-generation
reviews. The earlier thin wide SVD n=256 profile attributes 34.22% cumulative
CPU to Dlasr beneath Dbdsqr (38.05% cumulative). These nested percentages
must not be added. Of Dlasr's 660 ms flat samples, about 450 ms are in
Left/Variable/Backward and 210 ms in the Right fallback. The existing blocked
Right helper contributes another 490 ms cumulative. This is the measured
reason to optimize rotations rather than rewrite SVD.

Public path: mat.SVD.Factorize -> Dgesvd -> Dbdsqr -> Dlasr. Dbdsqr applies
left/right variable-pivot rotations in both directions to shrinking active
blocks, often with a leading dimension inherited from the larger matrix.

ARM64-experimental changes stay inside Dlasr, after its original validation:

- Left/Variable uses the existing f64.RotUnitary NEON kernel for n>=16,
  except a single rotation (m=2) requires n>=32. All-identity calls return
  without matrix writes. Active A and coefficients must be disjoint, and
  coefficients must be finite in [-1,1]; rejection retains the old loop.
- Right/Variable retains the existing m,n>=64 and all-active gate, then
  carries four independent rows through the original rotation sequence.
  Two-/one-row tails handle awkward dimensions. Steady-state A traffic
  drops from two loads/two stores to one load/one store per row/rotation,
  plus endpoint loads/stores; coefficient traffic is not included.
  Overlap, nonfinite or out-of-range coefficients retain the existing blocked fallback,
  including its pre-existing alias behavior, not a new sequential alias contract.
- After the blocked path declines, m>=32/n>=16 sequential Right calls use a
  separate non-inlined copy of the original loop. This isolates its hot code
  from growth in Dlasr's dispatcher; smaller calls retain the in-function loop.

Alias checks conservatively cover the bounded A storage span, including
padding, not only logically active elements. These changes add no allocation,
public API, workspace, parallelism or new experimental API.
Top/Bottom pivots, AMD64, default, safe and noasm remain
on their existing implementations. Right uses scalar multiply-plus-FMA sequencing,
not vector SIMD; it is kept behind the measured experimental ARM64 gate.
The coefficient guard is not a test that c*c+s*s equals one.

Race-instrumented builds retain the original Right implementation because
instrumentation changes the compiler's scalar fusion choices. Left SIMD stays
enabled and the public numerical regressions still run under race; only
direct tests asserting selection of the optimized Right helper are excluded.

## Numerical review

The source-review boundary is Dlasr and its new helpers plus existing
f64.RotUnitary, not every translated routine reachable from Dgesvd.
Reference: [Reference-LAPACK v3.12.1, SRC/dlasr.f](https://github.com/Reference-LAPACK/lapack/blob/6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca/SRC/dlasr.f).
All 12 Side/Pivot/Direct branches were source-reviewed and exercised by the
existing finite/mixed-identity Netlib differential suite. Left/Right Variable
are regression-covered; Top/Bottom source expressions are unchanged. Existing Go
deviations include row-major storage/lda>=n, typed arguments and panics,
slice validation and the pre-existing all-active Right row-block schedule.
Dlasr has no workspace query or convergence status. Existing quick-return
and invalid-argument tests remain in force. This is not a claim of complete
transitive SVD parity or Netlib endorsement.

Persistent tests cover cutoff/tail boundaries, compact/padded storage, offsets,
exact guard/padding preservation, identity/signed-zero/nonfinite inputs,
coefficient/A aliases, normalized extreme values, and rejection before writes.
Independent Dlasr, Dbdsqr and Dgesvd Netlib tests run separately from timing.
The linked Homebrew package is lapack 3.12.1; live ILAVER reports 3.12.0.
This is reference Netlib, not Accelerate or optimized OpenBLAS.

Two rejected details matter:

1. Unguarded RotUnitary reuse can turn an unnormalized overflow result into
   NaN. Whole-call coefficient/alias guards preserve the previous path.
   An initial new bit-exact extreme reference also failed the untouched
   baseline because the compiler fused a different term. The new test was
   corrected to use scaled forward error and nonfinite classification;
   no existing numerical tolerance was relaxed.
2. The first Right carry prototype failed the existing 1e-14 block-boundary
   regression. Direction-specific explicit math.FMA expressions now preserve
   the fused term in the baseline ARM64 loops. The existing test is unchanged.
3. Initial ten-round timing caught a 10-13% m=63/n=65 fallback regression.
   The hot instructions/registers were unchanged, but the added dispatch
   shifted their placement. This is consistent with a code-layout effect.
   Isolating the sequential loop restored that case; broader screening caught
   64x31 as well and established the m>=32/n>=16 isolation boundary. No padding
   or instruction-alignment hacks were added. Both cases remain in benchmarks.
4. The baseline race build passed while the candidate failed the unchanged
   Right block-boundary test. Right-only race opt-out preserves the original
   instrumented arithmetic rather than weakening the test. Non-race
   instruction identity is checked separately after this build-tag change.

## Generated code

Native disassembly confirms Left's four-column NEON loop has four
vector loads/stores, eight multiplies and two adds/subtracts, with no hot-loop
calls, bounds checks or spills. Its 128-byte wrapper retains one RotUnitary
call per active rotation. Bulk arithmetic is unfused; the scalar tail uses
FMA, so bitwise equivalence is not claimed.

Right's direction-specific arithmetic helpers inline, four carries stay in
FP registers, and the 16-byte-frame kernel has no hot-loop calls, divisions,
allocations or spills. Bounds branches remain on row loads/stores; this pass
does not claim their elimination.

The isolated sequential fallback has a 16-byte frame. Its arithmetic/FMA
order, per-rotation coefficient reads, identity checks and bounds checks match
the original scalar Right loop, with no hot calls, divisions or spills.
It adds one call per qualifying Dlasr invocation, not per matrix element.

## Measurement contract

Final clean common-harness checkpoint: `3d49adb8ad9b2e71cc84c2d8769d1c67d1fbffec`.
It differs from the published starting point only in persistent tests and
benchmark coverage. All compared production baseline code is unchanged.
Measured implementation: `a261c3530e22875d324364d0653b06993ecbd2a5`;
race-only build selection repair: `c7d22d61c985e9459c42c4274fd62bf268090cf9`.
The complete non-race machine-code sections match before/after that repair.
Final benchmark source SHA-256 is
`05c9b253f6ef3fe012b852314569fec42b0b024d35d8e8ae5770fca39943fac5`.
The baseline mat binary reused from the earlier test-only checkpoint has
identical mat benchmark and production source. Initial acceptance used
`7f087432d485dff49c853c301a4d4b7b8f76e834` and pre-isolation candidate binaries;
those results are retained separately, not pooled with the final rerun.

Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power, no affinity or
frequency lock. Native builds use Go 1.27.1, GOEXPERIMENT=simd, explicit
-pgo=off, empty GOFLAGS; GOGC/GOMEMLIMIT are unset. LAPACK timing binaries
include the netlib tag; public mat binaries do not. Compatibility uses
Go 1.26.4. [Go 1.27 SIMD notes](https://go.dev/doc/go1.27#simd) and installed
API documentation were refreshed; experimental adoption remains reversible.

The go-optimisation compare_benchmarks.py runner alternates prebuilt B/C and
C/B, retains binary hashes/environment, and rejects failures or mismatched
benchmarks. Acceptance uses ten independent rounds, 150 ms per case,
GOMAXPROCS=1 unless stated. No compilation, test suites or profiles overlap
timing. Normal desktop background activity remains; the host is not isolated.
The final exhaustive Right/Variable cohort uses ten 100 ms rounds; other
final cohorts use ten 150 ms rounds unless specified.
Three-round 100 ms screens are diagnostic, not acceptance evidence.
benchstat: x/perf v0.0.0-20260312031701-16a31bc5fbd0.

Kernel fixtures repeatedly apply normalized rotations to finite, nonzero
matrices with post-run finite checks. Public SVD benchmarks include actual
Factorize costs, reusing the SVD receiver. Native Netlib comparisons use each
backend's native layout and include restoring A but not layout conversion.
The bridge includes cgo overhead. They are not wrapper-inclusive API parity
benchmarks and do not compare optimized vendor BLAS.

Raw logs, codegen, runner metadata and binaries are temporary local evidence
under `/tmp/gonum-dlasr-followup.CGDjvd`. Persistent benchmark/test entry
points and source checkpoints allow reruns after that directory expires.

## Final acceptance results

Percentages below are time changes, not throughput percentages. Intervals are
benchstat's 95% intervals. Per-case p-values are unadjusted; many comparisons
increase false-positive risk. No initial and final samples were pooled.

### Left variable rotations

Ten 150 ms rounds. All 12 dense eligible cases improve 14.10-51.72%, p<0.001.
All 26 cases retain zero B/op and allocations. Selected Forward/Backward medians:

| m x n / lda | Baseline ns/op | Candidate ns/op | Change |
| --- | ---: | ---: | ---: |
| 2 x 32 / 32 | 26.41 / 26.75 | 22.69 / 22.67 | -14.10% / -15.25% |
| 4 x 16 / 16 | 41.94 / 42.25 | 35.41 / 35.44 | -15.56% / -16.11% |
| 32 x 16 / 16 | 393.6 / 392.7 | 310.1 / 309.8 | -21.19% / -21.11% |
| 32 x 32 / 35 | 703.8 / 702.2 | 470.4 / 484.1 | -33.17% / -31.06% |
| 32 x 256 / 256 | 5497 / 5494 | 2654 / 2653 | -51.72% / -51.71% |

At 32x256/lda259, sparse cases improve 22.96%/22.71% and identity-only
calls improve 16.83%/16.51%, all p<0.001. Candidate intervals are 0-4%.

Retained regressions: single-rotation n=15/16/17/31 fallbacks add roughly
0.56-0.76 ns (2.14-4.10%, p<=0.017). Sparse m=4/n=17 adds 0.64-0.93 ns
(3.19-4.70%, p<0.001). This is not zero-cost dispatch.

### Right variable rotations and fallback repair

Ten 100 ms rounds cover all 142 persistent Right/Variable cases, including
tiny inputs, 31/32/33 and 63/64/65 boundaries, 79/80/81 row tails, compact/
padded storage, dense/sparse/identity patterns and both directions. All 32
carry-eligible dense cases improve: Forward 22.74-28.79% (p<=0.002),
Backward 2.10-12.27% (p<=0.022). Candidate intervals are 0-7%.
All cases retain zero B/op and allocations.

| m x n / lda | Forward baseline -> candidate us/op | Backward baseline -> candidate us/op |
| --- | ---: | ---: |
| 64 x 64 / 64 | 2.829 -> 2.152 | 2.827 -> 2.564 |
| 65 x 65 / 68 | 2.910 -> 2.240 | 2.910 -> 2.751 |
| 80 x 65 / 68 | 3.657 -> 2.673 | 3.653 -> 3.220 |
| 256 x 64 / 67 | 11.210 -> 8.213 | 11.211 -> 9.980 |
| 256 x 256 / 256 | 46.72 -> 33.27 | 46.49 -> 42.97 |

The initial 10-13% 63x65 regression is removed. Final Forward compact/padded
cases are inconclusive (2.735 -> 2.745/2.744 us, p=0.269/0.054); Backward
retains +0.20%/+0.33% (2.733 -> 2.739/2.742 us, p<=0.002). At 64x31,
Forward is -0.65% (1.315 -> 1.307 us, p=0.013); Backward retains +0.23%
(1.307 -> 1.310 us, p<0.001). These are not claims of exact equivalence.

Identity calls retain about 0.8-1.9 ns overhead (0.40-3.37%, p<=0.018).
Tiny n=1 quick returns add about 0.58-0.61 ns, which is 12-13% of a roughly
4.7 ns baseline. Other significant small dense fallback increases are
roughly 0.1-4 ns (up to 5.53% on the 7x2 case); medium fallbacks increase
at most 0.48% in this cohort. These costs are retained alongside the gains.
Sparse cases have no statistically significant slowdown; some improve by up
to 9%, but they use the sequential fallback, not the new carry algorithm.

### Public SVD

Ten 150 ms rounds. All nine thin-SVD cases improve, p<=0.019, with candidate
intervals of 0-5%. Shapes are square n x n, tall 2n x n and wide n x 2n.

| n | Shape | Baseline ms/op | Candidate ms/op | Change |
| ---: | --- | ---: | ---: | ---: |
| 32 | square | 0.1826 | 0.1738 | -4.81% |
| 32 | tall | 0.2744 | 0.2647 | -3.54% |
| 32 | wide | 0.2334 | 0.2253 | -3.45% |
| 128 | square | 6.372 | 5.487 | -13.89% |
| 128 | tall | 10.98 | 10.09 | -8.13% |
| 128 | wide | 9.040 | 8.178 | -9.54% |
| 256 | square | 55.56 | 49.00 | -11.80% |
| 256 | tall | 86.63 | 79.97 | -7.69% |
| 256 | wide | 80.34 | 73.72 | -8.24% |

Eight values-only controls are inconclusive. Wide n=128 improves 0.16%
(p=0.023), an unchanged-path movement not attributed to rotations. This is
not a general claim for all matrices, SVD jobs or architectures.

Public SVD bytes/op are effectively unchanged (reported differences round
to +/-0.00%). Allocation medians are unchanged except thin/wide n=256
reports four -> three. No allocation was removed by the new kernels, so
this is not claimed as a memory optimization; short benchmark iteration
counts can change setup amortization.

### Worker-count and unrelated controls

Ten 150 ms rounds at GOMAXPROCS=4 confirm all six thin-SVD n=128/256 gains,
p<0.001, with candidate intervals of 0-1%. Square/tall/wide time changes are
-13.85%/-8.30%/-9.95% at n=128 and -12.71%/-9.25%/-9.50% at n=256.
This is not an assumption that the same percentage holds at every worker count.

All 15 QR/LQ/LU/Cholesky factorization controls at GOMAXPROCS=1 are
inconclusive (p=0.075-0.971), including sizes 32/128/256 and applicable
square/tall/wide shapes. No statistically significant slowdown was found
in this final control cohort; exact equivalence is not established.

### Native reference Netlib

Ten 150 ms rounds. All five Gonum Dgesvd cases improve 2.88-15.94%, p<=0.002.
All five unchanged Netlib timing controls are inconclusive. Native-layout
thin-vector medians follow. All ten cases report zero B/op and allocations.

| m x n | Gonum before ms | Gonum after ms | Netlib after ms |
| --- | ---: | ---: | ---: |
| 32 x 24 | 0.1064 | 0.1034 | 0.07344 |
| 24 x 32 | 0.10104 | 0.09636 | 0.08071 |
| 96 x 64 | 1.315 | 1.224 | 1.090 |
| 64 x 96 | 1.219 | 1.077 | 1.060 |
| 256 x 256 | 52.90 | 44.47 | 89.66 |

The 256-square Gonum fixture takes about half Netlib's time; Gonum still
takes about 2-41% more time on the four smaller fixtures. This does not
establish a general SVD win or compare Accelerate/optimized OpenBLAS.

## Final validation and remaining work

Passed full default Go 1.26.4 and experimental Go 1.27.1 suites. Affected
f64/BLAS/LAPACK/mat suites pass with safe, noasm and bounds individually;
the final build-tag repair also passes their combined opt-out configuration.
Three repetitions of race-enabled Dlasr tests pass. The same common public
regressions pass on the clean baseline. Native Dlasr/Dbdsqr/Dgesvd Netlib
checks pass three times on both experimental and default release builds.
GODEBUG=simd=0 Dlasr/bidiagonal/SVD/public-mat checks pass; this setting
controls portable emulation, not all architecture-specific leaves. Existing
tolerances remain unchanged. Vet, formatting, import policy, copyright and
diff checks pass. Linker duplicate-library/rpath warnings were benign.

Default-build new helper symbols and calls are eliminated, and default
Right blocked/sequential instruction bodies match the baseline. Whole
default Dlasr instruction identity is not claimed: the compiler changed one
FMA association in the source-unchanged Left/Top/Backward loop. Default
all-layout Netlib invariants and existing tests pass; floating-point bitwise
identity across builds is not guaranteed.

For the race-only repair, full Mach-O __TEXT,__text dumps have identical
SHA-256 hashes before and after: LAPACK
`1e0a7e405e2c798da4d9de2458b7013686cd5daca52140c1e6ad5279969f75b0`, mat
`4825a001c38f6606b5b01c30651ca08854ecf2d143baca7870723962441f5c10`.
These hashes exclude headers from otool's text-section dump, unlike the
whole-file binary hashes below. Thus the race guard does not invalidate the
non-race timings.

A separate post-validation three-second thin-wide n=256 profile has 3.30 s
of CPU samples. Dlasr is 28.48% cumulative beneath Dbdsqr (34.24%); do not
add those nested values. Right carry is 12.42% flat, sequential Right 6.06%
flat, and RotUnitary 8.79% cumulative. DotUnitary is now 21.21% flat/26.67%
cumulative; 0.79 of its 0.88 sampled seconds comes through dgemmSerialNotTrans.
That dot-product/DGEMM path is the next measured bottom-up target, not an
already accepted optimization. Native AMD64, other CPUs, PGO and all SVD
jobs remain unmeasured. No cross-compilation was performed, as requested.

## Reproduction

Build separate baseline and candidate binaries at the checkpoints above:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o /tmp/lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
```

Use the skill's compare_benchmarks.py with separate --baseline/--candidate,
a fresh --output, --rounds 10, --cpu 1, the indicated --benchtime and
GOMAXPROCS=1. Selectors:

```text
Left, 150ms:   ^BenchmarkDlasrVariable$
Right, 100ms:  ^BenchmarkDlasr$/side=R/pivot=V/
SVD, 150ms:    ^BenchmarkFactorization$/SVD/
Netlib, 150ms: ^BenchmarkDgesvdNetlibKernels$
Controls:     ^BenchmarkFactorization$/(QR|LQ|LU|Cholesky)$/
P4 SVD:       ^BenchmarkFactorization$/SVD/kind=thin/shape=./n=(128|256)$
```

Use --cpu 4 and GOMAXPROCS=4 for P4. Run benchstat on baseline.txt and
candidate.txt; retain its per-case results and the runner's metadata. Oracle:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib ./lapack/gonum -run '^Test(Dlasr.*|DbdsqrNetlib.*|DgesvdNetlib.*)$' -count=3
```

Final binary SHA-256 identities (LAPACK baseline/candidate, mat baseline/candidate):

```text
2b0f2ffa4e7ce7f7c5f9d7bc21eef89ee5a82687f75299678ba41228f79b0a0c
201f56aa55c1b8b2e6745095c692c5625a13f822a6894074cfb27c84e59ca41d
8ac176b8b64fa5ed1807bba4fabe0a97625978a693d76c362aaad9332c6d1c78
5bfcc6555c5ba6462671a00f4e38eba923ba5a665ee4b16759f3e0a1547b0027
```
