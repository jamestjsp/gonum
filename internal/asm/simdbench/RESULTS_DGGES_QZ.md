# Dgges QZ left-update indexing

## Decision and baseline

The smallest retained candidate bounds the six H/T row windows in
`doQZSweepDouble` once per outer iteration, using one explicit shared width.
All right-side H/T and Q/Z indexing remains unchanged. Two broader row-window
experiments were rejected because compact n=256 sweep controls regressed.

Baseline: `57149f493cf53011dfea6befeded3f82fcf9416d`, fetched and confirmed
equal to origin/codex/arm64-simd-blas before editing. An isolated detached
worktree receives only identical new test/benchmark harnesses, not the proposed
production change. No public API, new intrinsic, architecture dispatch, storage
allocation, workspace contract or algorithm is introduced.

The preceding vector n=256 CPU profile contained 5.48 s of samples:
doQZSweepDouble was 46.53% flat / 47.81% cumulative. Line attribution identified
repeated indexed left H/T and right H/T/Q/Z updates. Dgghrd's remaining A/B
rotations are also expensive, but their outputs feed the next elimination;
they cannot be delayed like independent Q/Z accumulation. This milestone
does not alter them.

## Measurement conditions

Native Apple M1 Pro, darwin/arm64 v8.0, macOS 26.6.2 (25G83), AC attached at
80% battery, normal desktop background activity, no affinity/frequency pinning.
Primary toolchain Go 1.27.1, GOEXPERIMENT=simd, netlib build tag, -pgo=off,
empty GOFLAGS, GOMAXPROCS=1 and test.cpu=1. Baseline and candidate use the
default Gonum BLAS. The oracle is Homebrew reference LAPACK package 3.12.1
(runtime ILAVER reports 3.12.0), not Accelerate or OpenBLAS.

Each primary experiment uses ten serial alternating baseline/candidate rounds,
100 ms per case, with binaries built before timing. Builds, profiles and tests
do not overlap measurement. Benchstat is
`golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`.
Reported intervals are benchstat's 95% median confidence intervals; many per-case
tests can produce incidental significance. Unweighted aggregates are not
application speedups, and inconclusive changes are not equivalence.

The persistent private-sweep harness covers n=3/4/5/32/64/128/256, all four
Q/Z on/off combinations and n=256 stride=259 in addition to compact storage.
It uses deterministic finite upper-Hessenberg H and triangular T, resets inputs
and optional vectors inside timing, and checks finite outputs after timing.
Setup and buffer allocation are outside timing. This measures one forced double
sweep plus reset, not convergence or the public entry-point cost.
The unchanged DggesControl harness supplies the representative public caller:
form (no vectors/sorting), vectors (both vectors/no sorting), left (both vectors,
left-half-plane sorting), unit (both vectors, unit-disk sorting).

## Experiments and generated code

1. Independent high/low row slices on the left and three-element windows on the
   right: removed some indexing/checks but left residual checks in the left
   loop. The symbol grew 3792 to 3904 bytes. Compact n=256 leaf none/q/z
   increased 3.17% / 1.50% / 1.87%; Dgges unit n=32 increased 2.92%
   (p=0.023). Dgges was mostly inconclusive despite several leaf wins. Rejected.
2. Explicit equal width on all six left slices, retaining right windows:
   removed all indexed checks inside the left loop and restored symbol size
   to 3792 bytes. Dgges n=32-128 had several 1-3% gains, but compact n=256
   none/q/z still increased 2.15% / 0.96% / 1.52% (p<=0.004). Rejected.
3. Keep only the explicit-width left change; restore all original right indexing.
   Complete left-loop bounds-check elimination remains, with the same ordered
   FMADDD/FMSUBD arithmetic and no hot-loop calls, stack traffic or floating-point
   spills. Stack frame remains 368 bytes; symbol size is 3872 bytes (+80).
   Right-side generated code returns to the baseline scalar indexed shape.

The right-window variants changed loads (including FLDP), checks, scheduling and
code layout together. The experiments do not isolate one instruction as the
cause of the compact regression. No platform/stride threshold was added to hide
those cases.

## Left-only primary results

All primary leaf cases remain 0 B/op and 0 allocs/op. All Dgges allocation counts
and bytes remain unchanged from the baseline. There is no statistically
significant adverse time result in the left-only cohort.

| Dgges case | Baseline | Left-only | Time change |
| --- | ---: | ---: | ---: |
| form n=32 | 196.6 us +/-6% | 192.7 us +/-0% | -2.01%, p<0.001 |
| form n=64 | 1.101 ms +/-3% | 1.075 ms +/-0% | -2.30%, p<0.001 |
| form n=128 | 7.720 ms +/-0% | 7.536 ms +/-1% | -2.39%, p<0.001 |
| form n=256 | 116.5 ms +/-1% | 114.6 ms +/-1% | -1.59%, p=0.001 |
| vectors n=64 | 1.655 ms +/-2% | 1.629 ms +/-0% | -1.62%, p<0.001 |
| vectors n=128 | 13.53 ms +/-0% | 13.35 ms +/-1% | -1.30%, p<0.001 |
| vectors n=256 | 169.8 ms +/-3% | 169.0 ms +/-0% | -0.48%, p=0.005 |
| left n=64 | 2.401 ms +/-6% | 2.374 ms +/-1% | -1.14%, p=0.003 |
| unit n=64 | 4.296 ms +/-2% | 4.255 ms +/-1% | -0.97%, p=0.003 |

Vectors n=32 is -1.60% (p=0.043). Left n=32 and unit n=128/256 are
inconclusive. Other sorting signals are small: left n=128 -0.72% (p=0.019),
left n=256 -0.62% (p=0.035), unit n=32 -1.79% (p=0.043).
Do not treat these marginal or sub-percent signals as broad application gains.

Representative sweep results: no-vector n=64 -5.54%, compact n=256 -1.72%,
padded n=256 -7.35%; both-vector n=64 -2.64%, compact n=256 -1.13%,
padded n=256 -3.25%. Compact q-only n=256 is inconclusive, not a proven win.
Raw evidence includes every mode, size, allocation metric and inconclusive case.

## Numerical and compatibility evidence

Scoped source pin: Reference-LAPACK v3.12.1,
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, SRC/dhgeqz.f, label 230.
The reachable stack was traced; this work reviews the changed row-index delta,
not every reachable translated routine. It does not establish full transitive
Dhgeqz/Dgges parity.

The explicit width is ilastm-j+1. Each row starts at row*ld+j and covers the
same old inclusive j..ilastm columns. Local range index k maps exactly to old
column j+k. H expression/stores precede T expression/stores in the same order;
floating-point association, exact-zero stores, Householder/shift state,
convergence counters, final Givens operations and optional-vector paths are
unchanged. No numerical tolerances were relaxed.

Two-stage slices can theoretically extend to capacity for an invalid direct
private-helper call. Valid public callers require full minimal backing
(n-1)*ld+n, and all six windows lie within that length. No valid public panic
or failure-output contract changes.

New persistent TestDoQZSweepDoubleStridedRanges covers minimum-width bulges,
n=3/4/5/8/33, full/interior active ranges, nonzero first updated row, differing
ilast/ilastm, four Q/Z combinations, distinct padded strides with minimum legal
backing lengths, padding canaries, untouched leading/trailing envelopes and
bitwise compact-versus-padded results. Unexpected nonfinite values fail.
This is same-kernel geometry coverage, not an independent arithmetic oracle.
Existing native Dhgeqz/Dgges oracle suites cover optional-vector modes,
convergence/failure, real/complex blocks, scaling, active subranges, sorting and
deterministic Dgges benchmark pencils through n=256. Shared/internal tests cover
workspace behavior.

Passed for left-only production:

- Full Go 1.27.1 SIMD repository suite.
- Full SIMD+Netlib lapack/gonum test binary.
- Full default Go 1.26.4 lapack/gonum test binary.
- Go 1.24.0 lapack/gonum and mat tests.
- Go 1.27.1 SIMD safe and noasm lapack/gonum and mat tests.
- SIMD+Netlib race tests for Dhgeqz, Dgges and the direct sweep regression.
- Independent Sol numerical and generated-code reviews.

No native AMD64 measurement or cross compilation was performed. This is generic
Go code; no experimental SIMD API is needed by the production change.
A fresh Netlib speed comparison was not run for this small delta; do not turn
the before/after results into a new Netlib-relative speed claim.

## Raw evidence and reproduction

Retained paired samples:

- Rejected broad slices: [baseline](results/dgges-qz/slices-baseline.txt),
  [candidate](results/dgges-qz/slices-candidate.txt).
- Rejected explicit-width plus right windows:
  [baseline](results/dgges-qz/width-baseline.txt),
  [candidate](results/dgges-qz/width-candidate.txt).
- Left-only primary: [baseline](results/dgges-qz/left-baseline.txt),
  [candidate](results/dgges-qz/left-candidate.txt).

Actual binary SHA-256 values:

| Binary | SHA-256 |
| --- | --- |
| SIMD baseline | `ecffb39e8a5a3e5695d1b9fc9bae1bf9deccea5bb7c4d0a4b95e679e57bff956` |
| Broad slices | `69e513ce7ba1bb96895eb525962058b7d2fac92d15875ab25fa2b12c38f96446` |
| Width plus right windows | `5dbdaed2f7fec6002d7c86717933b803fc53a65ee9dbb126d5cb0a865f229a62` |
| Left-only | `4ee81b7188963165bbc135a7537579a947d5f18508a68f7ff022a1a05f3af225` |
| Default baseline | `dcd05db79fc196dbc176be5b8ac4dfae8e1ae78137346097fd64179eacd1136d` |
| Default left-only | `7441aed0d14f87f5defc3e766c11d33eb703ed685afb0c2606d8a5278d39d8f5` |

The new geometry test was hardened after the first build; V2 and left-only
binaries contain the hardened checks. Benchmark source is identical throughout.
Local worktree, raw rounds, runner metadata, diagnostics and test logs remain at
`/tmp/gonum-qz-tune.JEqPVt`.

Build two isolated revisions with the same benchmark harness:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib \
  -c -o /tmp/qz-candidate.test ./lapack/gonum
GOMAXPROCS=1 python3 /path/to/go-optimisation/scripts/compare_benchmarks.py \
  --baseline /tmp/qz-base.test --candidate /tmp/qz-candidate.test \
  --bench '^Benchmark(DhgeqzSweepDouble|DggesControl)$' \
  --rounds 10 --benchtime 100ms --cpu 1 --output /tmp/qz-comparison
benchstat internal/asm/simdbench/results/dgges-qz/left-baseline.txt \
  internal/asm/simdbench/results/dgges-qz/left-candidate.txt
```

For the longer caller recheck select
`^BenchmarkDggesControl$/mode=(form|vectors)$/n=(64|128|256)$` at 500 ms.
For the default cohort build with `GOTOOLCHAIN=go1.26.4 GOEXPERIMENT=`,
omit the netlib tag, and select `^BenchmarkDggesControl$` at 100 ms.

## Final confirmation and acceptance

Accepted: the left-only production delta. The Go optimization skill's
interleaved caller gates and separate numerical/code-generation reviews rejected
the larger variants and kept the smaller, measured improvement.

The separate longer recheck uses ten alternating 500 ms rounds with unchanged
SIMD baseline/left-only binaries. All six changes have p < 0.001:

| Dgges case | Baseline | Left-only | Time change |
| --- | ---: | ---: | ---: |
| form n=64 | 1.100 ms | 1.075 ms | -2.25% |
| form n=128 | 7.718 ms | 7.547 ms | -2.21% |
| form n=256 | 116.0 ms | 114.6 ms | -1.20% |
| vectors n=64 | 1.653 ms | 1.630 ms | -1.38% |
| vectors n=128 | 13.52 ms | 13.36 ms | -1.17% |
| vectors n=256 | 170.0 ms | 169.1 ms | -0.52% |

Median intervals round to +/-0% except candidate vectors n=64 (+/-1%).
These confirm modest gains; the n=256 vector change is still sub-percent, not
a material closure of the reference-library gap. The primary and recheck data
are retained separately, not pooled.

A separate default Go 1.26.4 cohort, with GOEXPERIMENT empty, no netlib tag and
the same -pgo=off/P=1 settings, uses ten alternating 100 ms rounds. Form at
n=32/64/128/256 improves 1.93% / 2.59% / 2.58% / 1.12%.
Vectors n=64/128 improve 1.51% / 1.57%; vectors n=32/256 and every sorting case
are inconclusive. There is no significant adverse time result. Allocations and
bytes remain unchanged. Compare within each toolchain cohort, not absolute
timings across toolchains.

Additional retained evidence:
[longer baseline](results/dgges-qz/recheck-baseline.txt),
[longer candidate](results/dgges-qz/recheck-candidate.txt),
[default baseline](results/dgges-qz/default-baseline.txt),
[default candidate](results/dgges-qz/default-candidate.txt).

The remaining dependency-sensitive A/B rotations are a separate future target.
No changes to those rotations, QZ shifts or convergence policy were included.
