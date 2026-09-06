# BLAS-to-LAPACK hot-path performance pass

## Scope and reproducibility

Required workflow: `go-optimisation`, together with `gonum-simd` and the scoped
LAPACK numerical-review gates. The user added `go-optimisation` during this pass;
subsequent acceptance uses its prebuilt-binary comparison runner with recorded
binary hashes, sample order, runtime settings, and matching benchmark selection.
No repository PGO profile or GOFLAGS override was present when checked.

Starting implementation: `856a28593860b8ecaac50537134852c1dc8a7a6d` on
`codex/arm64-simd-blas`. Common `mat` APIs are the workload proxy; this is not
usage telemetry or a claim that every LAPACK routine is optimized.

Native host: Apple M1 Pro, darwin/arm64. SIMD measurements use Go 1.27.1,
`GOEXPERIMENT=simd`, initially `GOMAXPROCS=1`. Default-path checks use Go 1.26.4.
Prebuild test binaries before timing; run baseline and candidate serially and
interleave repeated samples. Record allocation counts and distinguish public
API costs from reusable-workspace LAPACK kernels. No AMD64 performance is
inferred from this host.

Native comparisons use Homebrew Reference-LAPACK, not OpenBLAS or Accelerate.
Source review is pinned to v3.12.1 commit
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`. The installed formula is 3.12.1_1,
but ILAVER reports 3.12.0. Source identity and runtime identity are distinct.

## Consumer call ledger

| Public workload | LAPACK path | Shared lower-level work |
| --- | --- | --- |
| QR / tall solve | Dgeqrf, Dtrcon; Dormqr, Dtrtrs | Dlarf/Dlarfb, GEMV/GER, GEMM/TRMM/TRSM |
| LQ / wide solve | Dgelqf, Dorglq, Dtrcon; Dormlq, Dtrtrs | Reflectors and GEMM/TRMM/TRSM |
| LU / square solve | Dgetrf, Dgecon; Dgetrs | Panel GER, trailing GEMM, TRSM |
| Cholesky / SPD solve | Dpotrf, Dpocon; Dpotrs | SYRK, GEMM, TRSM; triangular vector solves |
| SVD | Dgesvd: QR/LQ or direct Dgebrd, then Dbdsqr | Reflectors, GEMM, GEMV/GER, rotations |
| Symmetric eigen | Dsyev: Dsytrd, then Dsterf or Dorgtr/Dsteqr | Symmetric matrix/vector updates, reflectors, rotations |
| General eigen | Dgeev: Dgebal, Dgehrd, Dhseqr, Dtrevc3 | Hessenberg/Schur updates, GEMM/GEMV, triangular solves |

Factorization APIs query preferred workspace. QR, LU, and Cholesky additionally
estimate condition numbers; these costs belong in public API measurements.
SVD solves use matrix multiplication and singular-value scaling rather than a
LAPACK solve routine. `Dense.Solve` chooses LU, QR, or LQ by matrix shape.

This ledger is a prioritization aid, not a full transitive numerical-parity audit.
Changes require persistent regression tests, affected conformance tests, and
independent reconstruction/residual or native checks appropriate to the path.

## Milestones

Persistent entry points:

- `mat`: `BenchmarkFactorization`, `BenchmarkFactorizationSolve` cover sizes
  32/128/256, shape and vector-job variants, and one/16 right-hand sides. PCG
  fixtures and one untimed warm-up make steady-state reuse explicit.
- `blas/gonum`: `BenchmarkDtrsmSizes`, `BenchmarkDsyrkSizes` cover small and
  block-boundary dimensions; `BenchmarkDgemmSharedBacking` and
  `BenchmarkSgemmSharedBacking` isolate LU/upper-Cholesky panel layouts.
- `lapack/gonum`, tag `netlib`: `BenchmarkDgeqrfNetlib`,
  `BenchmarkDgetrfNetlib`, `BenchmarkDpotrfNetlib`, and the existing
  `BenchmarkDgesvdNetlibKernels` separate reusable-workspace kernel costs from
  public API overhead. Input resets are included equally; layout conversion and
  workspace queries are excluded. Native allocation counts cover Go, not C.

The first measured target is GEMM eligibility for shared-backing panels. A
three-second CPU profile of public LU at 256x256 attributed 30.6% of samples
cumulatively to `dgemmSerialNotNot`; the existing SIMD guard rejected entire
overlapping slice suffixes even when their active matrix entries were disjoint.
The proposed repair checks only active row intervals, keeps genuine-overlap
rejection, and reuses the existing SIMD kernel without altering LAPACK algorithms.
Strided AXPY and triangular solves remain substantial independent costs.

Longer representative baseline profiles (three-second benchmark targets,
`GOMAXPROCS=1`) give the next priorities:

| Workload | Cumulative hotspot share | Implication |
| --- | --- | --- |
| LU factorization, n=256 | 30.6% non-transposed GEMM | Enable disjoint shared-storage panels first |
| Tall QR, 512x256 | 43.0% GEMM with transposed B | Separate transpose-aware kernel opportunity |
| Cholesky, n=256 | 25.0% SYRK; 23.2% TRSV | Symmetric updates and condition estimation both matter |
| LU solve, n=256, 16 RHS | 99.0% TRSM | Test blocked triangular solves next |

These are sampled CPU shares for specific inputs, not universal workload weights.

## Shared-panel SIMD results

The benchmark-only checkpoint is `fefd7949` (same implementation as `856a2859`);
the shared-panel candidate is `dee9b2d4`. The initial manual six-sample screen
used 100 ms for broad native/public cohorts and 150 ms for shared-panel kernels.
After the skill audit, both committed trees were rebuilt with Go 1.27.1 SIMD
and explicit `-pgo=off`. The skill runner alternated ten rounds at 400 ms for
LU/Cholesky and 300 ms for shared GEMM, recording binary hashes, raw per-process
output, runtime environment and order. Its results supersede the initial public
and kernel percentages below. Raw local evidence is in
`/tmp/gonum-lapack-hotpaths.NcL3gt/`; the directory is not a permanent artifact.

| Workload, one worker | Before | Active-row guard | Time change |
| --- | ---: | ---: | ---: |
| Dgetrf, n=128, reusable workspace | 276.9 us | 251.8 us | -9.1% |
| Dgetrf, n=256, reusable workspace | 1.681 ms | 1.454 ms | -13.5% |
| Dpotrf Upper, n=256 | 783.7 us | 692.3 us | -11.7% |
| Public LU, n=128, audited | 360.4 us | 339.9 us | -5.7% |
| Public LU, n=256, audited | 2.013 ms | 1.813 ms | -10.0% |
| Public Cholesky, n=256, audited | 1.237 ms | 1.182 ms | -4.5% |
| Public Cholesky, n=128, audited | 256.1 us | 265.3 us | +3.6% |

All tabulated differences have p<0.05 (native rows: six manual samples; public
rows: ten runner samples). The n=128 Cholesky slowdown repeated and is
reported as a tradeoff; that size does not execute the changed GEMM path, so a
causal explanation has not been established. Larger LU/Cholesky gains outweigh
this measured small-size cost in this cohort, not in every possible workload.

Audited shared-panel DGEMM/SGEMM runtimes decreased 27.6-61.1% at one worker
(ten rounds, all p<0.001). Four-worker LU-shaped n=256 decreased 23.3% for DGEMM
and 36.7% for SGEMM (ten rounds, p<0.001). One-worker cases allocate nothing;
the parallel cases retain about 2 KiB and 20 allocations per operation in both
revisions. The initial screen's outliers and overbroad zero-allocation claim
are superseded by these recorded runner results. The
generated kernel still emits ARM64 vector FMLA instructions. Only dispatch
geometry changed, not its arithmetic loop or existing bounds checks.

QR and most SVD/public solve cases were statistically neutral. The broad screen
suggested general-eigen gains but had large baseline outliers, so this pass does
not claim those as established improvements. Lower Cholesky remains unchanged
and slower than this Reference-LAPACK runtime at n=256: 1.668 ms versus 1.147 ms.
Upper Cholesky and LU kernels at n=256 are faster than the native reference on
these fixtures (0.692 versus 1.783 ms; 1.454 versus 2.207 ms respectively).
These comparisons say nothing about optimized Accelerate/OpenBLAS performance.

Runner directories: `audit-mat`, `audit-gemm`, and `audit-gemm-p4` under the
evidence directory. For the public API comparison, SHA-256 identifies the
baseline binary as `d719facb5ec154b65d988d72a18787f93d458679e600fc6788013a9f8b739068`
and candidate as `55b80a875faabc6982087900cb9d8c9a3eb9d4eb5242b8339067d2e817d19966`.
Runtime controls: GOMAXPROCS as labeled; GODEBUG, GOGC and GOMEMLIMIT unset.

## Dedicated skill audit

Three independent Sol reviewers checked measurement quality, numerical behavior,
and code generation/dispatch. The audit corrected parallel-allocation reporting,
added self-contained comparison metadata and legacy-provenance notes, and
strengthened shared-panel exceptional-value and native solve tests.

The uncommitted blocked-TRSM prototype failed a persistent finite-to-infinity
regression in both precisions for Upper/NoTrans. Backward panel traversal also
reverses Lower/Trans accumulation within panels. The revised candidate blocks
only Lower/NoTrans and Upper/Trans/ConjTrans, preserving the original backward
paths. The original Lower/Trans test fixture was nondiscriminating and was
corrected to isolate descending-versus-ascending order within one panel.
No prototype timing is accepted as production or consumer performance evidence.

Native solve gates now compare Gonum and Netlib solutions directly as well as
checking independent residuals. Selected cases cover n=128/129/192/256, RHS
widths 1/16/17/64, both transpose/triangle choices, and RHS scales 1e-200/1e200.
Exponent-normalized residuals avoid overflowing the tolerance scale. These are
bounded numerical gates, not a full LAPACK parity audit.

The final extreme-scale solve cases use 16 RHS at n=192, so they exercise the
blocked path; a separate one-RHS case retains fallback coverage. Forward blocked
DTRSM/STRSM is restricted to left-sided, alpha=1 solves with at least 128 rows
and 16 RHS, using 64-row diagonal solves and serial GEMM updates. Backward
substitution, other scales, small cases, and unsupported builds retain the old
implementation. No workspace allocation or public API was added.

The audit also found that enabling blocking under `GODEBUG=simd=0` routed updates
through emulated GEMM. A three-round diagnostic screen roughly doubled n=256,
16-RHS public solve times, and a separate CPU profile attributed 63% cumulatively
to `dgemmSerialSIMD@simd0`. This was not accepted as a production result. The
revised gate checks `simd.Emulated()` and preserves the original solve path
under emulation; it does not change existing GEMM emulation behavior.

## Forward triangular-solve results

The final comparison uses baseline `4fb7419037b6f7c1fba1f75decc94434db4c18df`
and candidate `4d5e6224539a98917d4b27ed5fd038e853e3871d`. Both include the
shared-panel repair and identical benchmark code. Test-only extreme-RHS coverage
was strengthened after the candidate binary build; no benchmark or production
source changed. All binaries use Go 1.27.1, `GOEXPERIMENT=simd`, `-pgo=off`.

The skill runner collected ten interleaved rounds: 100 ms per BLAS case and
150 ms per public solve case, with `GOMAXPROCS=1`. Timings exclude benchmark
input resets for BLAS and use prefactored public solves. They are solve times,
not factorization-plus-solve times.

| Public solve, 16 RHS | Before | Forward blocking | Time change |
| --- | ---: | ---: | ---: |
| LU, n=128 | 112.29 us | 94.42 us | -15.91% |
| Cholesky, n=128 | 112.50 us | 94.90 us | -15.65% |
| LU, n=256 | 443.5 us | 338.4 us | -23.69% |
| Cholesky, n=256 | 444.4 us | 339.0 us | -23.72% |
| LQ, n=128 | 909.5 us | 892.1 us | -1.92% |
| LQ, n=256 | 3.895 ms | 3.791 ms | -2.67% |

All rows have p<0.01 with ten samples per revision. QR and all measured one-RHS
consumer cases are statistically inconclusive, not proven equivalent. LU and
Cholesky solves remain zero-allocation; QR/LQ retain 32 B and two allocations.

The 104-case public BLAS cohort covers both precisions, all four transpose/
triangle combinations, 127/128/129-row and 15/16/17-RHS dispatch boundaries,
and 256/512-row cases with 16/64 RHS. Enabled forward DTRSM cases reduce time
by 28.75-58.86%, and STRSM by 31.02-64.45% (all p<0.001). Both precisions
remain zero-allocation. These are per-case kernel results, not an application
throughput multiplier. Fallback cases include small statistically significant
slowdowns; the broad run's largest is STRSM Lower/Trans at 127x15 (+3.52%).
Changing source dispatch can alter generated code placement even when a case
does not enter the new helper; this is a hypothesis, not a diagnosed cause.

A focused one-worker repeat at 400 ms confirms the Lower/Trans, 15-RHS STRSM
cost: 127/128/129 rows regress 3.49/3.42/3.13% respectively (ten rounds,
all p<=0.001, `final-fallback-repeat`). This is an accepted, disclosed tradeoff
against the substantially larger enabled-path and consumer gains, not a claim
that all calls improve. No public API consumer regression was found in the
measured solve cohort. Recalibrate these dispatch decisions on other ARM64 CPUs
and after toolchain changes; do not extrapolate an M1 result to AMD64.

Local runner directories are `final-trsm` and `final-consumers`. BLAS binary
SHA-256: baseline `9395e6111828fa0413127d55102053947f2a707e87c9f1abee3600c432b8902e`,
candidate `b3c8a9ed85136d6e18ce631add1f391392513b1436850738c8c9051c71c2a4f3`.

With `GOMAXPROCS=4`, the 128/256-row, 16-RHS cohort retains forward gains of
32.6-49.8% for DTRSM and 33.7-54.5% for STRSM (ten rounds at 100 ms,
all p<0.001). These helpers deliberately use serial GEMM; four available
workers do not imply a parallel triangular solve. Allocation counts remain zero.
DTRSM Lower/Trans at 128x16 regresses 1.73% in this run; it remains on the
original arithmetic path. Results are in `final-trsm-p4`.

The final emulation check (`final-emulation`, ten rounds at 200 ms,
`GOMAXPROCS=1 GODEBUG=simd=0`) finds no statistically significant change in
LU/Cholesky 128/256, 16-RHS solve times (all p>=0.85; unchanged zero allocations).
LU n=256 is 443.2 us on both revisions. This removes the earlier roughly 2x
diagnostic regression; it is not a claim of native SIMD speed under emulation.

Public solve binary SHA-256: baseline
`3d4130294b08fcc0755b309038df14478f5f5c2234a74680ec68c154889131c0`, candidate
`b18fe05633c43c165e2141101811e2b6e0bbbe4ce415cfcf90b7576088be92e7`.
Native solve binary SHA-256: baseline
`e034590031dc19949e7c523a434322df67a42597e38fb75f43320ec128e29ab8`, candidate
`42f3b319e1e5067923333d41f55f241cfed3bbac7ed2e4239fe630362beeeee7`.

The native solve cohort (`final-native`, ten rounds at 100 ms) independently
confirms 15.1-24.2% lower Gonum time for 16 RHS and 9.4-14.9% for 64 RHS
across both DGETRS transpose and DPOTRS triangle choices (all p<0.001).
At n=256 and 16 RHS, DGETRS NoTrans takes 339.1 us versus Reference-LAPACK's
476.5 us; DPOTRS Upper takes 339.0 us versus 642.6 us. Both sides remain
zero Go allocations, and native-reference controls are largely unchanged.

One-RHS solves expose a separate major remaining bottleneck: at n=256,
DGETRS NoTrans is 273.8 us versus 30.20 us in Reference-LAPACK; DPOTRS Upper
is 273.6 us versus 40.97 us. This pass does not improve them. Their many
one-element TRSM updates warrant a dedicated vector-solve dispatch/profile
investigation, with transpose, layout, scaling and numerical gates, before
further instruction-level tuning.

Next priorities are this one-RHS solve path, transposed-B GEMM for QR/LQ
reflector updates, and SYRK/TRSV for Cholesky and condition estimation.
Backward TRSM needs an order-preserving algorithm before being reconsidered.
The earlier n=128 Cholesky factorization regression remains disclosed above;
this solve-only pass neither fixes it nor establishes a general SVD speedup.

A final factorization control (`final-factorizations`, ten rounds at 200 ms)
compares the same solve baseline/candidate for LU/Cholesky at 128 and 256.
No material additional change is established: three cases are statistically
inconclusive; LU n=128 changes -0.20% (p=0.029). Allocations are unchanged.

## Final validation

The finalized implementation passes full `go test ./...` runs with Go 1.26.4
and Go 1.27.1 SIMD, plus BLAS/LAPACK/mat `noasm` and `bounds` suites. Focused
safe-build, race and emulation checks cover blocked DTRSM/STRSM and shared
GEMM regressions. Strengthened native DGETRS/DPOTRS tests pass, including
16-RHS extreme scales that enter blocked dispatch. Prior shared-panel gates
also cover native factorization reconstruction and SVD conformance.

`go generate ./blas/gonum` leaves generated files unchanged. Import/copyright
policy checks and whitespace checks pass. Final binary inspection confirms
ARM64 D2/S4 vector FMLA updates in the reused GEMM kernels; no mallocgc or
makeslice call appears in the inspected TRSM/GEMM symbols. Existing bounds
exits remain; this is not a claim that all bounds checks or spills disappeared.
A final independent Sol review found no correctness or build-compatibility
blockers in the corrected dispatch. AMD64 production dispatch is unchanged
and has not been benchmarked in this pass; no cross-compilation was performed.

## Reproduce

Analysis uses `golang.org/x/perf/cmd/benchstat` at
`v0.0.0-20260312031701-16a31bc5fbd0`. The recorded host is macOS 26.6.2
(25G83), Apple M1 Pro; no affinity or frequency lock was applied. Except for
the explicitly labeled worker/emulation runs, GOMAXPROCS is 1 and GODEBUG,
GOGC and GOMEMLIMIT are unset. Builds, tests and profiles ran outside acceptance
timing windows. Temporary raw evidence is not a permanent repository artifact;
the persistent benchmarks and exact revisions above support reproduction.

Build each tree before timing, with matching benchmark files:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -pgo=off ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -pgo=off ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -pgo=off -tags netlib ./lapack/gonum -o lapack.test
GOMAXPROCS=1 ./mat.test -test.run '^$' -test.bench '^BenchmarkFactorization' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./blas.test -test.run '^$' -test.bench '^Benchmark[DS]gemmSharedBacking$' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./lapack.test -test.run '^$' -test.bench '^BenchmarkD(geqrf|getrf|potrf)Netlib$' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./mat.test -test.run '^$' -test.bench '^BenchmarkFactorization$/LU$/n=256$' -test.benchtime=3s -test.cpuprofile=lu.cpu
GOTOOLCHAIN=go1.27.1 go tool pprof -top mat.test lu.cpu
```

Use the skill comparison runner with ten alternating baseline/candidate rounds
and compare its recorded files with `benchstat`. Repeat selected kernels with
`GOMAXPROCS=4` and solve consumers with `GODEBUG=simd=0`. Native tags need
the optional darwin/cgo Homebrew Reference-LAPACK installation; the `mat` and
BLAS manifests are runnable without it, including on AMD64. AMD64 does not gain
new automatic GEMM dispatch from this ARM64-validated change.

1. Establish persistent factorization/solve and BLAS3 cohorts; profile the current
   implementation and compare representative kernels with Reference-LAPACK.
2. Change a measured shared bottleneck; validate kernel boundaries, supported
   layouts, numerical behavior, and end-to-end consumers against the same base.
3. Run fallback and full-suite gates, publish measured results and remaining
   priorities, and verify each milestone on the remote branch.

Unresolved questions: none.
