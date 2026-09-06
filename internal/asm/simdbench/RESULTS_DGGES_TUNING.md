# Dgges rotation and block-swap tuning

## Scratch-layout milestone

Accepted: coalesce already-escaping fixed arrays into one local owner per routine.
Baseline `d390a9babcd21f0ac97f0c5641e6d8a8566a0a47`, refreshed from origin.
Independent baseline and scratch-only worktrees receive identical new leaf
benchmark helpers; the Dgges harness is unchanged. Rotation changes are excluded
from this first comparison.

Production changes are limited to array ownership in Dtgex2, dtgex2SwapLarge and
Dtgsy2. Scratch capacities, arithmetic, operation order, BLAS calls, workspace
queries/minima, and failure paths are unchanged. Arrays that remained on the
stack were not pulled into the aggregate. No pool, unsafe annotation, new public
API or allocation lifecycle is introduced.

Native Apple M1 Pro, darwin/arm64 v8.0, macOS 26.6.2 (25G83), AC power at 80%.
Go 1.27.1, GOEXPERIMENT=simd, -pgo=off, netlib build tag, empty GOFLAGS,
GOMAXPROCS=1, benchmark CPU=1; no affinity/frequency pinning. Reference Homebrew
LAPACK source package 3.12.1 reports runtime ILAVER=3.12.0. No Accelerate/OpenBLAS
substitution. Builds, tests and profiles are separate from timed cohorts.

Ten serial alternating baseline/candidate rounds, 100 ms per case, with
benchstat x/perf v0.0.0-20260312031701-16a31bc5fbd0. Setup/allocation is outside
timing; input copies/reset and normal calls are inside. The swap fixture uses
canonical complex blocks with diagonal B blocks and nonzero surrounding coupling.
Each iteration checks success.

| Native leaf | Baseline | Scratch candidate | Time change | Allocations |
| --- | ---: | ---: | ---: | ---: |
| Dtgex2, 2x2-2x2 swap | 4.095 us +/-3% | 4.021 us +/-2% | -1.82%, p=0.001 | 9 to 3 |
| Dtgsy2, small solve | 512.3 ns +/-3% | 495.6 ns +/-4% | -3.26%, p=0.007 | 2 to 1 |

Both retain the same bytes/op: 1,472 and 576 respectively.
Compiler escape output confirms exactly three scratch owners on the large-swap
call path, replacing four/three/two individual objects.

Dgges sorting allocations fall by two-thirds at every measured size.
At n=256: left sorting 9,216 to 3,072 objects; unit sorting 36,864 to 12,288.
Unit sorting still allocates 6,029,312 bytes per call. This is an allocation-count
improvement, not elimination of heap churn.

Dgges latency is mostly inconclusive; no material end-to-end speedup is claimed.
The significant sorting signals are small: left n=64 -0.61% (p=0.002), unit n=64
-0.52% (p=0.035), unit n=128 -0.34% (p=0.011). The unchanged vectors n=128
control also moves -0.25% (p=0.023), illustrating why these sub-percent changes
should not be over-attributed. There is no significant adverse time result.
Unweighted aggregates are not application speedups. Multiple tests raise the
chance of incidental significance.

Validation on the isolated scratch tree:

- Full native SIMD+Netlib lapack/gonum test binary passes.
- Full Go 1.24.0 lapack/gonum and lapack/testlapack tests pass.
- SIMD+Netlib race tests for Dtgex2/Dtgexc/Dtgsen/Dtgsy2/Dtgsyl/Dgges pass.
- Existing oracle coverage includes all swap sizes, embedded blocks, rejection,
  separated extreme scales, workspace queries and successive Sylvester blocks.
- Formatting and diff checks pass. No numerical tolerances changed.

Source-review pin: Reference-LAPACK v3.12.1,
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, SRC/dtgex2.f and SRC/dtgsy2.f.
The review covers this scratch-layout delta, not complete transitive Netlib parity.
The Dgges transitive tracer and dynamic oracle tests do not replace a complete
branch ledger.

Evidence is retained in [scratch-baseline.txt](results/dgges-tuning/scratch-baseline.txt)
and [scratch-candidate.txt](results/dgges-tuning/scratch-candidate.txt).
Original local metadata, binary hashes, raw rounds, escape diagnostics and logs:
`/tmp/gonum-dgges-tune.IW9Ikl/scratch-gate` and its parent directory.

Reproduce with prebuilt test binaries using the Go optimization skill's
compare_benchmarks.py runner, the same toolchain/flags and:

```sh
GOMAXPROCS=1 python3 /path/to/go-optimisation/scripts/compare_benchmarks.py \
  --baseline /tmp/base-lapack.test --candidate /tmp/scratch-lapack.test \
  --bench '^Benchmark(Dtgex2Scratch|Dtgsy2Scratch|DggesControl)$' \
  --rounds 10 --benchtime 100ms --cpu 1 --output /tmp/scratch-comparison
benchstat internal/asm/simdbench/results/dgges-tuning/scratch-baseline.txt \
  internal/asm/simdbench/results/dgges-tuning/scratch-candidate.txt
```

The new Dgghrd control benchmark is included as an unchanged-implementation
baseline harness for the separately gated rotation milestone.

## Schur-vector rotation milestone

Accepted after the scratch-only milestone `dbc318b6`. The independent
scratch-only binary is the baseline; the candidate adds only Dgghrd rotation
batching to production. Both use identical benchmark helpers. This isolates
rotation gains from the preceding allocation change.

Dgghrd still generates rotations and updates A/B immediately in the original
order. Q/Z cannot affect that elimination: their rotations are saved in chunks
of at most 32 and replayed in the same per-row order, walking contiguous row
segments four rows at a time. Four independent carries reuse coefficients and
reduce strided matrix traffic. Scalar tails retain the same expressions, including
Z's reversed operand order. No identity shortcut changes exceptional-value
behavior.

The fast path requires the default value-type Go BLAS implementation, n >= 32,
at least two eliminations in the current column, and a requested Q or Z.
Pointer, wrapped and custom BLAS implementations retain the original immediate
Drot calls. Public signatures, validations, explicit/postmultiply modes, lower-B
zeroing and quick returns are unchanged. This does not introduce an aliasing
guarantee between distinct LAPACK matrix arguments.

This is generic scalar Go, not a new SIMD intrinsic or ARM64-only implementation.
Native ARM64 compiler inspection found coefficient reuse across four independent
row carries, no floating-point spills or calls in the hot loop, and bounds guards
outside that loop. No cross compilation was performed; native AMD64 performance
remains unmeasured.

### Native before/after results

The primary cohort uses the same M1 Pro / Go 1.27.1 SIMD environment as above:
10 serial alternating rounds, 100 ms per case, P=1, prebuilt binaries. It covers
16 Dgges cases and 40 Dgghrd controls. Entries below are median elapsed time;
negative changes mean less time. These deterministic fixtures are not a claim
about every matrix pencil or application.

| Dgges mode | n | Scratch baseline | With rotations | Time change |
| --- | ---: | ---: | ---: | ---: |
| vectors | 32 | 296.5 us | 281.0 us | -5.22% |
| vectors | 64 | 1.745 ms | 1.657 ms | -5.06% |
| vectors | 128 | 16.01 ms | 13.53 ms | -15.50% |
| vectors | 256 | 240.1 ms | 169.9 ms | -29.24% |
| left | 32 | 420.9 us | 406.2 us | -3.48% |
| left | 64 | 2.502 ms | 2.406 ms | -3.85% |
| left | 128 | 20.60 ms | 18.23 ms | -11.49% |
| left | 256 | 267.9 ms | 199.1 ms | -25.70% |
| unit | 64 | 4.403 ms | 4.289 ms | -2.59% |
| unit | 128 | 32.38 ms | 29.86 ms | -7.79% |
| unit | 256 | 351.3 ms | 280.0 ms | -20.28% |

Vectors requests both Schur-vector matrices without sorting; left/unit also
request both, sorting into the left half-plane / unit disk respectively.
All listed changes have p < 0.001 except unit n=64 (p=0.002). Unit n=32 is
inconclusive (p=0.089); all four no-vector/no-sort form controls are inconclusive.

| Dgghrd, both accumulators | n / stride | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: | ---: |
| compact | 32 / 32 | 67.54 us | 52.64 us | -22.07% |
| compact | 128 / 128 | 6.161 ms | 3.779 ms | -38.66% |
| compact | 256 / 256 | 143.56 ms | 73.15 ms | -49.04% |
| padded | 256 / 257 | 26.27 ms | 20.46 ms | -22.11% |
| padded | 256 / 259 | 25.48 ms | 20.62 ms | -19.07% |

These leaf changes have p < 0.001. Eligible Q-only and Z-only cases improve
10.30-31.57% and 10.72-32.88% respectively. The large compact-versus-padded
difference already exists in the baseline; do not generalize the compact
n=256 result to other strides or hardware. All Dgghrd cases remain 0 B/op and
0 allocs/op. Dgges allocation counts and bytes are unchanged from scratch-only.

Retained adverse control: Z-only n=31, on the immediate fallback, initially
increased 0.22% (p=0.030). A separate longer 10-round, 500 ms recheck gives
49.40 us +/-2% versus 49.55 us +/-1%, inconclusive (p=0.218). Both datasets are
retained separately, not pooled. The unchanged no-accumulator n=32 control
also moved -0.65% (p=0.023); tiny signals should not be over-attributed.

A separate default Go 1.26.4 cohort, without SIMD or the netlib tag, uses the
same scratch baseline and final production delta: 10 alternating 100 ms rounds.
Vectors at n=32/64/128/256 use 5.53%, 5.15%, 14.07% and 27.71% less time.
At n=256, left sorting uses 21.46% less time, unit sorting 19.16% less.
These changes have p < 0.001. This confirms the optimization is useful without
the experiment; absolute timings across different Go toolchains are not a
controlled compiler comparison.

Exploration was not an acceptance gate: a naive per-row prototype had mixed
single-iteration smoke results; a carried-value prototype's three-round screen
still had a roughly 1% slower n=64 vector median. Four-row coefficient reuse
was then tested with the full cohort above. Neither exploratory sample size
supports a significance claim.

### Numerical and compatibility gates

Independent Sol numerical and code-generation reviews found no blocking issue
in this scoped delta. Source-review pin remains Reference-LAPACK v3.12.1,
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, SRC/dgghrd.f.
A/B update ranges, rotation generation and ordering are unchanged. Saved Q/Z
sequences are replayed before the next chunk/column; their live index ranges
are within the active matrix and do not touch padding. This is a delta review,
not a complete transitive QZ/Netlib parity certification.

Persistent tests cover:

- All nine Q/Z None/Explicit/Postmul pairs against immediate rotations at
  n=31/32/33/63/64/65/128, with compact and padded storage.
- A padded n=65 partial active range and a custom BLAS recording original
  immediate rotation calls.
- Replay scalar tails and four-row groups at n=3/4/5/8, both Q and reversed Z,
  identity, signed zeros, subnormals, infinities and NaNs. Compare bitwise with
  immediate Drot except for NaN payloads.
- New native Netlib batch-boundary checks sample seven Q/Z pairs (II, VV, NN,
  VN, NV, IN, NI), full/interior ranges, nonidentity postmultiply inputs,
  distinct padded strides, canaries, finite results and direct comparisons.
  Existing n=6 Netlib tests cover all nine pairs. This is not a full Cartesian
  oracle matrix at every batch size.
- Existing shared reconstruction, orthogonality and quick-return tests; new
  internal/native checks explicitly cover padding.

All pass: full repository Go 1.27.1 SIMD tests; full SIMD+Netlib lapack test
binary; full default Go 1.26.4 lapack test binary; Go 1.24.0 lapack/gonum and
mat tests; SIMD safe and noasm lapack/gonum and mat tests; focused SIMD+Netlib
race tests covering Dgghrd, Dgges, Dtgex2, Dtgexc, Dtgsen, Dtgsy2 and Dtgsyl.
No tolerances were relaxed.

### Updated reference gap and remaining profile

A separate current-binary comparison uses 10 alternating Gonum/Netlib rounds,
500 ms per case, n=256, the same native Go 1.27.1 SIMD environment and reference
library above. Conversion and workspace setup are outside timing; normal
input reset/call/success checks are inside.

| Current implementation | vectors | unit sorting |
| --- | ---: | ---: |
| Gonum | 168.0 ms +/-1% | 282.3 ms +/-3% |
| Reference Netlib | 143.6 ms +/-1% | 192.3 ms +/-0% |

Netlib still uses 14.55% and 31.87% less time respectively (p < 0.001).
Go allocation counters do not measure allocations inside the native library;
its reported zero must not be read as total process allocation parity.

A separate five-second vector n=256 CPU profile collected 5.48 s of samples:
doQZSweepDouble is 46.53% flat / 47.81% cumulative; Drot is 37.04% flat /
40.69% cumulative; dgghrdReplayVectors is 2.92% flat / 3.10% cumulative.
Dgghrd totals 43.07% cumulative. Thus QZ sweeps and the remaining immediate
A/B rotations are the next measured targets, not another speculative rewrite
of the newly reduced vector accumulation. Profiles are not timed-cohort samples.

### Reproduction and retained evidence

Primary data: [rotation baseline](results/dgges-tuning/rotation-baseline.txt),
[rotation candidate](results/dgges-tuning/rotation-candidate.txt).
Default-toolchain data: [baseline](results/dgges-tuning/default-baseline.txt),
[candidate](results/dgges-tuning/default-candidate.txt).
Separate adverse-control recheck:
[baseline](results/dgges-tuning/small-recheck-baseline.txt),
[candidate](results/dgges-tuning/small-recheck-candidate.txt).
Current reference data: [Gonum](results/dgges-tuning/netlib-gonum.txt),
[Netlib](results/dgges-tuning/netlib-netlib.txt).

SHA-256 of the actual prebuilt test binaries:

| Binary | SHA-256 |
| --- | --- |
| Original base | `5a31358ca0880c5056c43ac311d5eb21a19fd7d81cccddb97611966878c7bf4f` |
| Scratch-only / rotation baseline | `295aeed06c9560c6c0fdbc64fff9b770cc437c6fd60c236b1c272775c6c9f496` |
| Primary rotation candidate | `4df5f152bf214d273fcb444263ae9e47335effa536040d36940ccfa6f63c2952` |
| Final extended-test candidate | `da89c085f59f87caab554f2aaa32e66ea4b5a38ec58e8984931482d983b28b2e` |
| Default baseline | `4bddbcf0adce7a23943e755585a7703725451abf76b2935395955ce763ff4c6e` |
| Default candidate | `9d67e5859310f6ed2938779c84da1941cd74275367f95cc57b795b13551581ef` |

The final candidate adds exceptional replay test cases to the primary candidate;
production is identical. It is used for the longer control recheck, current
Netlib comparison and final profile. Raw rounds, runner metadata, compiler
output and validation logs remain in `/tmp/gonum-dgges-tune.IW9Ikl`.

Build baseline and candidate separately with identical harnesses and flags:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib \
  -c -o /tmp/candidate-lapack.test ./lapack/gonum
GOMAXPROCS=1 python3 /path/to/go-optimisation/scripts/compare_benchmarks.py \
  --baseline /tmp/scratch-lapack.test --candidate /tmp/candidate-lapack.test \
  --bench '^Benchmark(DggesControl|DgghrdControl)$' \
  --rounds 10 --benchtime 100ms --cpu 1 --output /tmp/rotation-comparison
benchstat internal/asm/simdbench/results/dgges-tuning/rotation-baseline.txt \
  internal/asm/simdbench/results/dgges-tuning/rotation-candidate.txt
```

For the default cohort build with `GOTOOLCHAIN=go1.26.4 GOEXPERIMENT=`,
omit the netlib tag, and benchmark only `^BenchmarkDggesControl$`.
For the separate 500 ms control recheck use
`^BenchmarkDgghrdControl$/mode=z$/n=31$/stride=31$`.
Run the current reference comparison from one final netlib-enabled binary with
`^BenchmarkDggesNetlibControl$/implementation=(Gonum|Netlib)$/mode=(vectors|unit)$/n=256$`;
alternate separately selected implementation invocations for 10 rounds at P=1.
