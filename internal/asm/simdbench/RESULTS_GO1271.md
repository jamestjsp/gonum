# Go 1.27.1 SIMD results and limits

SIMD kernels remain comparison candidates. Ordinary AMD64 dispatch retains assembly with isolated correctness repairs. The evidence supports specific improvements and explicit cost tradeoffs on the measured host; it does not show that every routine beats assembly.

Paths below are relative to `benchmark-results/simd-next-c7adb87b/` in the evidence archive. `final-go1271-results-scope-addendum-v3/EVIDENCE_INDEX.json` binds the detailed records by hash. Benchmark binaries and caches need not be embedded in this source file.

## Scope and reproducibility

The main nine binaries used stock **Go1.27.1**, `GOEXPERIMENT=simd`, AMD64v1, PGO off and CGO1; metadata reports `go1.27.1-X:simd`. See `h2-go1271-toolchain-evidence-v1/`. The measured host is an Intel Core i7-11850H (8 performance cores/16 threads), with AVX2/AVX512. Benchmark processes use CPU2, whose SMT sibling is CPU10; the powersave governor is left unchanged. Exact process order, environment and host context are retained. The stock toolchain path is `/home/arjunanj/.local/share/mise/installs/go/1.27.1`; archived metadata and build argv bind its use, not a custom compiler. Software feature overrides and ARM64 compilation do not prove execution on all physical hardware. Go1.24 supplies compatibility evidence; custom-compiler experiments are separate and unadopted.

The historical main comparison is source790: **53,556 usable observations**, comprising49,452 completed direct/BLAS observations from the failed first session and4,104 fresh complete matrix observations. Its lone114 earlier matrix observations remain excluded incomplete-cohort evidence. Neither the failed session nor missing in-memory timestamps were fabricated into a successful run. These are not measurements of the later combined source.

All58 routines are indexed by `h2-final-source-decision-ledger-preparation-v1/ROUTINE_SURFACE_58.json`:57 manifest routines plus the separate48-observation math32.Sqrt supplement, which did not satisfy its both-width direct significance gate. The ledger preserves **2,484 positive costs** (1,413 versus ASM,995 versus c7,76 versus production), all1,185 pointwise-material IDs and overlapping historical32/139/1,379 cohorts. Positive medians need not be significant; material/Holm tags are review signals, not automatic selection rules.

## Retained changes and explicit costs

* **Unitary L1 ASM infinity repair (FBB):**35 finite identities/560 observations all had lower medians,0 positive costs. This is repaired-versus-original ASM, not a Go SIMD gain. Authority: `l1norm-asm-repair-root-retention-v1/DISPOSITION.json`.
* **Strided L1 ASM infinity repair (MH):**76 identities/1,216 observations had70 lower and6 positive medians,0 material costs. All six and the actual original repeated-infinity failures remain recorded. `l1norminc-repair-root-retention-v1/RETENTION.json` is superseded only for old zero-stride expectations by R6.
* **Zero-stride contract repair (R6):**return +0 without reading input. Versus MH-repaired ASM,76 identities retain55 positive,14 pointwise-material and9 Holm costs, with0 Holm gains. Worst direct n100000/inc1: +2,824ns (+10.5679%). Dlangb MaxColumnSum m=n256/kl15/ku16/ldab34: +177ns (+4.9663%), Holm-significant despite being below5%. Ten of12 consumer medians are positive. This is a required correctness tradeoff, not a speed win. Authorities: `l1norminc-zero-stride-root-retention-v1/RETENTION.json`, `l1norminc-zero-guard-finite-analysis-v1/ALL_POSITIVE_COSTS.json`.
* **Four-chain L1 vector remainder:**10,944 observations/1,140 declared comparisons retain286 positive costs. Four-chain versus current has26 positive,0 pointwise-material,1 Holm cost and171 Holm gains; versus repaired ASM,46 positive,13 material,15 Holm costs and170 Holm gains. The retained n32/spread/512 control costs +0.295ns (+4.592869%). Unchanged short source does not explain or erase it. Authority: `l1norm-long-vector-remainder-root-retention-v1/`.
* **f32 short Sum pointer plus shared zero initializers:**of69 changed n4..31 identities,67 medians are lower and52 are Holm gains versus current. All126 identities retain10 current costs, including material/Holm n64 packed controls: width256 +0.6655ns (+10.889307%), width512 +0.673ns (+11.463124%). Versus ASM,30 costs/22 material remain; all three contrasts retain122 costs. Both measured arms include the zero workaround, so this does not measure its isolated speed effect. No production f32 Sum consumer gain or assembly-dispatch replacement is claimed. Authority: `f32-sum-short-pointer-root-retention-v1/RETENTION.json`.
* **K67 c128 stride-pair rejected:**the completed258-identity/9,288-observation trial and separate grouped audit retain292 positive/139 material costs across all774 contrasts. Of72 changed pair-versus-current contexts,64 cost more (31 Holm costs,28 material),8 have lower medians (3 Holm gains). The186 controls retain63 costs (5 Holm costs,2 material) and123 lower medians (17 Holm gains). The instruction-count hypothesis did not justify selection; keep the existing c128 source. Authority: `c128-zscal-stride-pair-root-disposition-v1/DISPOSITION.json`; full exact rows: `c128-zscal-stride-pair-complete-decision-ledger-v1/`.

Source790 already contains the Q3 valid-unaligned c128 AXPY repair, checked dot-span/widened-Sdsdot correctness changes and restored c7 DscalUnitary SIMD entry. Q3's separate aligned-cost trial retains47 positive,3 Holm and0 material costs (maximum+4.636%). Dot evidence distinguishes AMD64 full-width products from386 helper calls. No correctness or rollback cost is waived by source equality.

## Five historical material consumer costs

| Actual caller / width | Cost |
|---|---:|
| Caxpy n5, strides1/1 /512 | +0.810ns, +7.479% |
| Sger n8, inc1 /512 | +1.810ns, +7.594% |
| Dger m8/n32/pad0 /512 | +2.745ns, +5.987% |
| Sger m64/n129/inc7 /256 | +71ns, +6.592% |
| Zscal n33/inc7 /256 | +1.305ns, +7.018% |

These exact rows remain in `h2-final-source-decision-ledger-preparation-v1/FIVE_CONSUMER_COSTS.json`; routes are in `h2-five-consumer-cost-source-review-v1/`. RQPAGI's later GER mask proposal failed actual codegen. That rejection is not proof of a hardware speed limit. The later K67 measurement of Zscal33/inc7/256 found current19.85ns, pair21.51ns and production21.795ns: pair/current +8.363% is a Holm cost, while pair/production remains inconclusive. That separate session does not erase the original +7.018% cost or establish a causal explanation for its change.

## Nineteen historical ASM-losing classes

These classes contain46 positive medians among four manifest cases each, not19 universally slow or significant routines. Exact IDs/prior trials/source-only hypotheses remain in `h2-final-source-decision-ledger-preparation-v1/class-evidence.json`; the updated machine index is `final-go1271-results-scope-addendum-v3/CLASS_STATUS_19.json`.

| Routine | Positive /4 | Investigation and remaining limit |
|---|---:|---|
| c128.DotcInc | 2 | Clear/return trial rejected; custom-compiler extraction measured but unadopted; full-span correctness retained. Short extraction/ABI costs remain. |
| c128.DotuInc | 2 | Same strided-return and exact public-span investigations. No trusted-caller bypass; short costs remain. |
| c128.DscalInc | 3 | Tiny/layout trial rejected; c7 unitary entry restored;32 actual Dscal/mat routes and554-ID link diagnostic reviewed. Source/code equality does not waive costs. |
| c64.AxpyInc | 4 | Equal-stride two-pointer trial retained with95 controls per width. Mixed/reverse/sparse entry costs remain. |
| c64.AxpyIncTo | 4 | Same equal-stride two-pointer trial with destination semantics preserved. All four historical manifest medians remain positive. |
| c64.DotuInc | 2 | Entry flattening retained; paired-span and sign-first candidates rejected for actual extra hot reloads, before correctness/timing. |
| c64.DotuUnitary | 2 | Width/cache routes and actual unitary/public handoff inspected. Unit stride bypasses new full-span work; no new alignment-fault claim. |
| f32.DdotInc | 2 | AVX2 admission/cache retained; accumulator seeding rejected; widened Sdsdot bias fixed separately. Conversion/reduction/ABI costs remain. |
| f32.DdotUnitary | 2 | Same widened-reduction investigations; custom compiler unadopted. No universal conversion or hardware ceiling proved. |
| f32.DotInc | 2 | Pair packing retained; five-argument raw-leaf trial rejected on callers/controls. Nonunit public callers retain checked spans. |
| f32.Ger | 2 | Register tails retained; larger entries rejected; RQPAGI n&7 screen rejected at frame280 to288. Awkward shapes remain. |
| f32.Sum | 2 | Pointer plus shared zero initialization retained after4536 rows.10 current costs including2 material n64 controls remain; no production consumer gain. |
| f64.AxpyInc | 2 | Gather/address implementation retained; unrolled-tail consumer trial rejected. Sparse/short costs remain. |
| f64.AxpyIncTo | 2 | Same address/tail caller investigation, preserving destination semantics. Further hypotheses need actual caller evidence. |
| f64.Div | 2 | Vector division and actual42-shape floats caller trial completed. Reset/alias costs retained; no divider/bandwidth bound proved. |
| f64.L1Norm | 1 | Four-chain vector remainder selected after10944 rows; separate ASM correctness repairs.46 positive ASM comparisons/13 material;7RSCCJ limit remains. |
| f64.ScalInc | 4 | Running addresses retained; RN6 measured/rejected;3WE pretest codegen rejected; actual ScaleVec/link-layout controls reviewed. Tiny/nonunit costs remain. |
| f64.ScalIncTo | 4 | Same running-address, tail and consumer controls. No global alignment selection or proof all alternatives exhausted. |
| f64.Sum | 2 | Actual floats callers complete; static-length entry rejected after4032 rows/504 contrasts. All215 positive/62 material trial costs remain. |

Rejected changes remain evidence, not deliverables: RN6 tiny Scal,3WE loop entry, WINU64 Sum entry, RQPAGI GER mask, both c64 shared-span candidates, the K67 c128 pair, custom-compiler extraction and global funcalign64. Specific frame/reload/caller-cost failures do not prove every alternative exhausted. The separate floats trial retains119 positive/41 material comparisons; its Norm1 is an unchanged scalar control, not long L1Norm.

## Compatibility and final-source evidence

The original f32 pointer gate stays rejected for an existing128-bit AVX-only route emitting an AVX2 broadcast. Common zero-value declarations remove it in the separately reviewed revised f32 routes. The **separate7RSCCJ L1 portable zero/Abs lowering issue remains open**: tested AVX2/AVX512 eligibility is not global physical AVX-only acceptance. No physical crash is inferred from archived codegen.

Full f32 checkptr2 retains four named allocation failures; the matched current-zero control reproduces2/1/2 allocations. Exact classification preserves the failing suite/raw outcomes and is not a full checkptr PASS or a generic waiver. Other ordinary/race/safe/noasm checks remain separate. Temperature/throttle snapshots and repeated-round diagnostics neither establish independent samples nor explain away costs.

Composition records are `final-go1271-composition-v1/{INITIAL_COMPONENTS,R6_COMPONENTS,F32_COMPONENTS}.json`. They map ASM repairs, zero-stride guard/regressions, selected L1 entry/helper and f32 entry/common initializers. Optional L1 environment expectations and removal of f32's experiment-only mode assertion are explicit fixture integration adaptations; numerical/guard assertions stay intact. Level1 generator and generated Sdsdot source require parity.

The delivered bundle's actual final validation receipts, linked codegen/N1 records and source/patch replay identities are the authority for its combined tree. The validation plan requires one full SIMD ./... run; ordinary/safe/noasm/Go1.24 cover all chosen compatibility packages plus focused Dlangb/L1/R6 LAPACK checks. Historical broad LAPACK and standalone candidate evidence is retained with its original identity. This document does not relabel historical timing as final-source timing or assert future validation success. The old790 exporter is not valid for the changed tree. Delivery records must preserve excluded data, failed attempts and all costs, omit build caches, and verify exact-origin patch replay and device readback.
