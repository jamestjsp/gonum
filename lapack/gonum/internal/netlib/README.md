# Native Netlib oracle tests

These bridges are test-only infrastructure for the `netlib && darwin && cgo`
suite. Implementation code must not import this package. The current native
setup expects Homebrew LAPACK under `/opt/homebrew/opt/lapack`; it is not an
Accelerate or OpenBLAS comparison.

Run from the repository root:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib ./lapack/gonum -count=3
```

The shared tests also run without the external library. SIMD is not required by
the oracle bridges. The command above records the toolchain used for this pass.

## Dgges gap-closure ledger — 2026-09-06

The source audit used Reference-LAPACK v3.12.1, commit
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, starting from Gonum
`19fbe62a05ec633eec7490d48503e84077a0b385`. The installed package is
3.12.1 but its ILAVER runtime reports 3.12.0. Source and runtime evidence are
therefore distinct. Recheck the runtime with `TestNetlibRuntimeVersion` when
changing libraries.

| Audit gap | Persistent evidence and disposition |
| --- | --- |
| Dlapy2 mixed NaN/Inf precedence | Production fix and shared/native classification tests. NaN now takes precedence over infinity. |
| Dlange mixed NaN/Inf precedence | Production fix and direct native norm tests. NaNs cannot be masked by infinity in maximum/row/column norm reductions. |
| Dlarft zero reflectors | Production quick-return fix and no-mutation regression. Existing valid Dgges callers did not hit this case. |
| Dgges QZ failure exit | `TestDggesNetlibNonfiniteFailure`: deterministic NaN active block, all four vector options, B scales 1 and 1e±200, status, selection not called, selected count and each backend's workspace result. Complete state equality is a pinned-fixture regression, not a portable LAPACK failure guarantee. Finite-input nonconvergence remains unproven. |
| Singular QZ branches | `TestDhgeqzNetlibZeroDiagonalChase` and `TestDhgeqzNetlibSingularSplits`: every zero/tiny T diagonal position, small pivot, two-small-subdiagonal split, consecutive/all-zero diagonals; vectors on and eigenvalues only. Native oracle and coverage confirm execution of the previously untested zero chase and ILAZR2 paths. |
| Dtgexc partial rejection | Canonical four-complex-block pencil, movement in both directions, rejection after successful movement, status/destination, Schur form, reconstruction, vectors and eigenvalues. |
| Dtgsen rejection propagation | Same canonical pencil with ijob=0, selected dimension, workspace and semantic partial outputs. This is not proof of full-driver Dgges reorder rejection. |
| Dtgsy2 scaling | Added overflow-scale 1x1 N/T cases; existing successive-block tests cover all four local block-size combinations. |
| Primitive independent coverage | Routine-specific native tests now cover Dlag2, Dlagv2, Dlartg, Dlasv2, Dlapy2, Dlassq, Dlanhs and Dlange. Includes singular and asymmetric scales; selected single-infinity cases where relevant. Not an exhaustive exceptional-value matrix. |
| Reflector independent coverage | Dlarft: both directions/stores. Dlarfb: all 16 direction/store/side/transpose combinations. Dormqr: both sides/transposes, query/minimum/optimal work and genuinely blocked k=70 execution. Output padding checked. Queries validated independently, not required to equal another implementation's tuning. |
| Weak assertions | Shared norm, Schur, orthogonality, residual and comparison checks reject nonfinite false passes. Dlag2 residual failures are fatal for regular pencils. Finite driver/QZ fixtures no longer silently accept both implementations failing. |
| Pair-selection documentation | Clarified that selecting either complex conjugate selects the whole pair; misordered-pair failure has a direct regression. |

Numerical output is compared semantically where representations are nonunique.
The repeated-eigenvalue partial-rejection fixture deliberately does not demand
identical Q/Z entries between implementations. Invalid noncanonical discovery
pencils were excluded after checking their generalized discriminants; their
acceptance differences do not establish valid-contract defects.

The regular pencil A=[2,-1;3,4], B=[0,2;0,3] has two infinite roots.
Its tiny computed beta values differ between implementations, so relative
eigenvalue comparison is ill-conditioned. The named double-infinite cases
retain finite/nonzero homogeneous-output checks, Dlag2 determinant residuals,
and Dlagv2 transformed-matrix/rotation comparisons; only eigenvalue distance
is omitted. Separate one-finite/one-infinite fixtures retain strict comparison.

## Remaining gaps and deliberate differences

- A bounded search did not find a stable finite full-driver Dgges reorder-rejection
  fixture with a stateless eigenvalue selector. The helper rejection fixture has
  identical eigenvalue blocks and cannot select just its final block by an
  eigenvalue threshold.
- Finite QZ maximum-iteration exhaustion and every exceptional-shift,
  tiny-denominator/double-sweep pivot, complex-rescaling and split-state branch
  are not independently exhausted.
- The source-reviewed QR/RQ algorithms differ from current Netlib: sequential
  Dlarft versus recursive Dlarft, and Dlarf with explicit implicit-one handling
  versus DLARF1F/DLARF1L. Workspace heuristics also differ intentionally.
- Dgges calls Dtgsen with ijob=0. Condition-estimation-only paths, all extreme
  Dlatrs scalings and the entire Ilaenv/Iparmq policy surface are not certified
  by these new tests.
- Dlascl band-storage modes and the broader Netlib stride/zero-dimension/API
  contracts remain outside active Dgges scope. This change does not expand them.
- BLAS source parity and native AMD64 execution were not part of this pass.

Passing the suite establishes the listed regressions, not complete transitive
Netlib parity. Statement coverage is not branch coverage or a completeness score.

## Validation of this milestone

Native Apple M1 Pro ARM64: the full repository suite passed with Go 1.27.1
and GOEXPERIMENT=simd. The unskipped Netlib package suite passed three
repetitions after assertion hardening. The final expanded driver failure fixture
also passed three focused repetitions. The lapack/gonum and mat packages passed
safe, noasm, bounds and race checks, and tests on the minimum Go 1.24.0 toolchain.
Netlib-tagged vet, ordinary vet, gofmt and git diff checks passed. No unrelated
baseline failure was identified. No native AMD64 performance claim is made.
