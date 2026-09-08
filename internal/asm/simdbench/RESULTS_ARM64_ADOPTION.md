# ARM64 adoption of the September AMD64 bundle

## Baseline and scope

The imported baseline is `53702365864f9d09bad60c5fe54891d8777cc406`,
whose tree matches bundle source `9a1169e8268c8581fa1022a8422aa272903552d9`
(`747704973603960872bf348263b68083e479cc7c`). All delivered SHA256 checks
passed before import. Native validation uses an Apple M1 Pro, darwin/arm64,
stock Go 1.27.1 and `GOEXPERIMENT=simd`. AMD64-specific assembly is not
executed on this host; the imported report remains separate evidence.

The adoption work first reconciles native conformance failures, then evaluates
bounded ARM64 candidates against the unchanged import. Existing portable SIMD
and archsimd kernels are the baseline, not scalar code selected to exaggerate
gains. Handwritten NEON is conditional on generated-code evidence and repeated
native leaf and consumer measurements. Unsupported build modes retain their
existing implementations. No cross-compilation is used in this investigation.

## Import validation

- Default Go toolchain and Go 1.24: affected kernel, BLAS, LAPACK and mat tests pass.
- Go 1.27.1 SIMD with combined `safe noasm bounds` tags: the same scope passes.
- Full native SIMD: fails only in f64, at `TestGemvTSmallIEEEAndPadding`
  (Inf versus NaN) and `TestSIMDPositiveStridePanics` (writes before panic).
- Portable emulation and race tests reproduce those two f64 failures.
- Native Netlib: LAPACK passes; BLAS fails its old Sdsdot zero-length expectation.
- Changed Go files are gofmt-clean; the imported diff passes whitespace checks.

These are baseline results, not acceptance of the subsequent adoption changes.
The bundle separately discloses AMD64 checkptr level-2 allocation assertions;
ordinary and level-1 results do not erase those failures.

## Conformance reconciliation

Sdsdot now agrees directly with Reference-LAPACK 3.12.1 for zero length: it
returns the supplied bias. The old oracle test explicitly expected Gonum to
disagree with Netlib; that special case was removed, not the oracle comparison.

The small transposed GEMV reference distinguishes ARM64 scalar FMA tails from
separately rounded vector products, without changing the production arithmetic
or the AMD64 reference. Oversized-stride tests still require a bounds panic;
they no longer prescribe partial mutations of invalid inputs before that panic.
Go may move bounds checks within an unrolled block. Valid-input numerical,
padding, aliasing and access tests remain in place.

## Candidate selection

- Four-row DGER: reuse y loads across rows instead of entering AXPY for each row.
  First scope is disjoint unit-stride input. This does not cover Dgetf2's
  strided, shared-matrix input and is not a claimed LU improvement.
- DDOT: the current slice intrinsic loop emits extensive per-load bounds and
  address work. Compare a checked pointer-loop archsimd implementation with a
  whole-kernel NEON assembly leaf, preserving its exact FMA/reduction order.
  A baseline CPU profile of `mat.BenchmarkInnerMedMed` attributes 92.67% of
  sampled CPU cumulatively to DotUnitary and its inlined loads. Profile timing
  is not performance acceptance evidence.

## Measurement method and rejected alternatives

All acceptance timings use the skill's alternating prebuilt-binary runner,
`GOMAXPROCS=1`, stock Go 1.27.1 with SIMD, default native NEON width, and no
PGO input or GOFLAGS override. Builds, tests and worker CPU work were stopped
before timing. Both revisions use identical deterministic benchmark fixtures.
Each case has ten samples per revision: 100ms per leaf/public-BLAS sample and
200ms for matrix-inner-product samples. Benchstat is
`golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`, using its default
Mann-Whitney comparison and 95% confidence intervals. These are per-case tests,
not a multiplicity-adjusted aggregate application-throughput estimate.

The unchanged baseline worktree is at the imported commit above. Its only
additions are identical public DDOT/DGER benchmark files; generating all
single-precision sources reproduces the baseline exactly. Raw observations,
run order, settings and binary hashes are retained under
`/tmp/gonum-arm64-adopt.79xQZz/` in `dot-final`, `ddot-final`, `ger-final`,
and `inner-final`. Each directory contains `baseline.txt`, `candidate.txt`,
`metadata.json`, and per-run stdout/stderr. Benchmark commands are reproducible
from the retained Go benchmark names without those temporary binaries.

A real NEON assembly dot leaf was implemented and passed exact arithmetic,
tail, offset, overlap, ABI and Darwin guard-page checks. It used two
post-increment four-vector loads and four FMLA operations per eight elements.
It was nevertheless **rejected**: in ten-round native comparisons against a
checked Go pointer-loop control it took 4.6–100.4% more time for every tested
active length, 16–4096, across aligned and offset inputs. The Go control emits
eight vector loads and four FMLAs without hot-loop bounds checks. Fewer assembly
instructions did not establish faster execution; ABI/load-scheduling costs were
not independently isolated. The rejected source/control checkout and binaries
remain in the temporary evidence directory; assembly, its race-only fallback,
and its comparison-only source files are not retained in the repository.

The first dot wrapper imposed about 0.62ns on short inputs; restoring the
original short path before long-path validation removed the significant cost.
The first DGER entry called its helper even for tiny rejected shapes, costing
about 2.3ns. The final entry checks unit strides and cheap size bounds before
the helper call. Both rejected measurements remain in `dot-import-v-neon`,
`dot-go-v-neon`, and `ger-import-v-candidate`; they are not final-source results.

## Toolchain boundary

Release notes and current proposal pages were checked on 2026-09-08. SIMD
remains behind the experiment in this work; SVE/default-enablement proposals
are not used as evidence that a new API or hardware capability is available.
Use the installed Go 1.27.1 source to resolve exact intrinsic signatures.

- <https://go.dev/doc/go1.27>
- <https://github.com/golang/go/issues/78902>
- <https://github.com/golang/go/issues/73787>
- <https://github.com/golang/go/issues/79781>
- <https://github.com/golang/go/issues/78979>
- <https://github.com/golang/go/issues/76175>

## Retained native results

Times below are medians from the final ten-round cohorts, against the unchanged
import. All reported active-path reductions have p<0.01. These measurements are
for the Apple M1 Pro only; they do not establish AMD64 or other ARM64 timings.

| Operation | Import | Retained Go | Time change |
| --- | ---: | ---: | ---: |
| DotUnitary n=16, offset=0 | 9.800ns | 4.706ns | -51.97% |
| DotUnitary n=64, offset=0 | 35.985ns | 9.077ns | -74.78% |
| DotUnitary n=256, offset=0 | 140.45ns | 30.69ns | -78.15% |
| DotUnitary n=4096, offset=0 | 2237.0ns | 621.9ns | -72.20% |
| BLAS Ddot n=16, offset=0 | 11.130ns | 5.790ns | -47.97% |
| BLAS Ddot n=256, offset=0 | 141.45ns | 31.75ns | -77.55% |
| BLAS Ddot n=4096, offset=0 | 2233.5ns | 624.8ns | -72.03% |
| mat.Inner small/small | 93.39ns | 91.15ns | -2.39% |
| mat.Inner medium/medium | 5.690us | 1.574us | -72.34% |
| mat.Inner large/large | 555.0us | 156.9us | -71.73% |
| mat.Inner large/small | 7.795us | 7.506us | -3.71% |
| Dger 4x2, pad=0 | 19.88ns | 10.30ns | -48.18% |
| Dger 8x8, pad=0 | 39.31ns | 18.39ns | -53.22% |
| Dger 64x16, pad=0 | 374.0ns | 159.1ns | -57.45% |
| Dger 64x64, pad=3 | 920.2ns | 725.6ns | -21.15% |
| Dger 64x512, pad=0 | 7.128us | 5.766us | -19.10% |
| Dger 64x512, pad=3 | 7.231us | 6.804us | -5.91% |

Every final measured case reports zero allocations. DotUnitary n=0/1/7/8/15
and public Ddot n=7/8/15 show no significant difference in either tested offset;
this is not a proof of equivalence. Across all measured active lengths 16–4096,
leaf reductions are 49.48–78.18% and public Ddot reductions are 46.52–77.93%.

DGER's 24 active shape/padding cases improve 5.91–57.45%. Retained tiny fallback
costs are explicit: 3x7 pad0 is 20.56→20.87ns (+1.51%, p=0.001), 3x7 pad3 is
20.60→20.89ns (+1.41%, p=0.008), and 4x1 pad0 is 19.77→20.02ns (+1.26%,
p=0.014). The fourth tiny control is inconclusive. The new guarded route is
retained for the larger useful gains, not presented as universally faster.
The three strided controls measured 11.97–12.97% lower times, but execute the
unchanged scalar path: no new strided algorithm or causal explanation is claimed.

A separate ten-round confirmation uses 200ms per public-BLAS sample, recorded
in `ddot-confirm` and `ger-confirm`. Ddot n=16/128/256, offset=0, takes
48.86%/76.93%/78.23% less time respectively (all p<0.001), with baseline
confidence intervals narrowed to about 1%. DGER 64x512 improves 20.40% with
pad0 and 5.98% with pad3 (both p<0.001). All four tiny DGER controls confirm
a 0.25–0.29ns cost, 1.21–1.39% (p≤0.017). This small fallback regression is
accepted explicitly in exchange for the measured active-path improvements.

Reproduce the focused comparisons using identical harnesses in both revisions:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./internal/asm/f64 -o f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./mat -o mat.test
# Run each prebuilt binary in alternating baseline/candidate order, GOMAXPROCS=1:
# -test.run=^$ -test.bench=^BenchmarkDotUnitaryLengths$ -test.benchtime=100ms -test.benchmem
# -test.run=^$ -test.bench=^BenchmarkDdotUnitaryLengths$ -test.benchtime=100ms -test.benchmem
# -test.run=^$ -test.bench='^BenchmarkDgerARM64SIMD(StridedControl)?$' -test.benchtime=100ms -test.benchmem
# -test.run=^$ -test.bench='^BenchmarkInner(SmSm|MedMed|LgLg|LgSm)$' -test.benchtime=200ms -test.benchmem
```

## Final validation

The retained Go implementations pass the full native Go 1.27.1 SIMD test suite.
Default-toolchain and Go 1.24 tests pass for internal assembly helpers, BLAS,
LAPACK/gonum and mat, as do Go 1.27.1 SIMD tests with combined
`safe noasm bounds` tags. Native Reference-LAPACK 3.12.1 oracle tests pass for
BLAS/gonum and LAPACK/gonum. Focused race tests cover DDOT, DGER, imported GEMVT
arithmetic and stride panics. Portable SIMD emulation also passes f64,
BLAS/gonum, LAPACK/gonum and mat tests. Persistent tests include exact FMA/reduction
behavior, tails, padding, alias fallbacks, invalid geometry and Darwin guard
pages. Focused DDOT checkptr level-2 tests pass; this does not claim that the
bundle's unrelated whole-suite checkptr allocation assertions are resolved.

Final f64/BLAS vet, changed-file goimports, repository import/copyright policy
and whitespace checks pass. No native AMD64 execution or cross-compilation was
performed; preservation of AMD64 routing is a source-review result.

## Scope and remaining opportunities

The import is usable on native ARM64 with the conformance adjustments above.
This is not a claim that every AMD64 optimization has an ARM64 counterpart or
that every BLAS routine was newly accelerated. Both retained kernels use Go
archsimd NEON; the compiler already emits the required instructions. AMD64
production routing is unchanged. No handwritten assembly is retained.

Single-precision GER is a separate candidate, not automatically promoted from
the double-precision timings. Fixed-stride widened dots may merit an interleaved
NEON-load experiment later, but arbitrary NEON gathers and short-call setup need
their own evidence. Dgetf2's strided/shared-matrix DGER calls remain outside the
new route. Recheck generated code and crossovers on future Go releases before
removing guards or switching APIs; this work makes no GA release promise.
