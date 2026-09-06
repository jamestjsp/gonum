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
