# I4R128 rotated-expert GPU parity contract (point 85, step 3)

Predeclared contract for the first real-executor GPU parity run on the rotated
I4R128 routed-expert path. Written in response to the step-2 review, which
correctly noted that the step-2 README was a procedural outline rather than an
executable/bounds contract.

Everything below is either **measured** (with the measurement named) or
**proposed for agreement before the run**. Nothing here is a claimed GPU
result: no device was available to the author and no model was executed.

---

## 1. Scope, and what this is not

**In scope.** The rotated routed-expert path through the real HIP executor:
bind a tiny GGUF, upload weights and the five rotation vectors, run
`Executor::Forward` in decode and prefill, compare against the CPU oracle.

**Out of scope, explicitly.**

* This is **not** a production-frontdoor or full-size model run. The fixture
  bypasses `Config::FromGguf` (see §3).
* This is **not** a quality, perplexity or capability claim. It is a
  numerical-parity check on one path.
* This is **not** a performance measurement. Dispatch option (a) is one
  single-token launch per (token, expert) slot and makes no speed claim.
* GPU44's acceptance, the frozen kernel patch `15dcfc04`, and prior bundles
  are untouched. Nothing here re-qualifies them.

---

## 2. Fixture — deterministic, hashed

Built by `BuildTinyHybrid()` in
`tests/models/qwen38_flash_next/i4r_tiny_fixture.hpp`.

| Property | Value | How measured |
|---|---|---|
| GGUF size | **30,658,944 B** (29.24 MiB; FF-GEOM256, supersedes 10,700,928 B) | built in memory, written, `ls` |
| SHA-256 | **`b0963e90e7fb4b51a838d9f980d641567ac0ec816fa40dc075dfaf217e2b8ed4`** (supersedes `8ed1955c…`) | `sha256sum` on the written file |
| Determinism | **identical across two independent builds** | `cmp` over the full byte vector |
| GDN head | **`ssm_head_dim = 128`** | `Config::SsmValueDim() = 512`, `SsmConvChannels() = 1024` |

> **Warning:** this identity supersedes the frozen v7 fixture
> `a00a6bbc…d0ff` (7,529,856 B). The GDN recurrence kernels are compiled
> for a 128-wide head and never receive the runtime head dimension, so the
> fixture's 64-wide head fell through to `GdnKernel` addressing 128-wide
> state against 64-wide buffers — a 4x over-read and over-write. Rebuilding
> at 128 wide changes every SSM tensor shape (`attn_qkv` 512→1024,
> `attn_gate` 256→512, `ssm_conv1d` 512→1024, `ssm_norm` 64→128,
> `ssm_out` 256→512) and therefore the hash. The v7 fixture stays
> preserved as the record of the consumed GPU attempt; everything
> downstream of this contract re-verifies against the new identity.

This hash **supersedes** `37d9dbe9…` (the pre-fix fixture). The fixture
changed because the HC norm gammas, the HC mixer down/up projections and
the full router payload are now populated with deterministic nonzero
values instead of zeros: a zero gamma makes `HcMix`'s RMS-normalised rows
zero, which collapses every downstream expert input and makes a real
`Forward` pass vacuous. The byte size is unchanged (the space was always
reserved); the content is not. The writer now validates every F32 payload
against its declared shape and aborts on a short vector (the old router
supplied 2 floats for a 512-float tensor).

The determinism check matters: the fixture is RNG-driven (`Lcg`), so the
hash is only meaningful if the generator is reproducible. It is.

**Geometry.** hidden 256 (2 blocks), expert_ff 128 (1 block), shared_ff 128,
experts 2, experts_used 2, layers 4 (3 linear + 1 attention), vocab 32.
Every axis is a whole multiple of `kI4rBlock` = 128 where the format requires
it.

**Scale round-trip.** All five rotation vectors are round-tripped through the
F16 encoding they are stored with, *before* both the dense oracle
materialisation and the tensor write. Without this the oracle and the loaded
weights multiply different values and the residual is scale precision, not
rotation error. With it, the CPU rotated-vs-dense residual is ~2.5e-7
(§6).

---

## 3. Reduced-config seam — explicit

The fixture binds through `ModelWeights::BindWithConfig(reader, config, err)`,
**not** `Bind()`.

Why: `Config::FromGguf` pins production geometry (48 layers, 512 experts,
hidden 2560), so no small GGUF can pass it. `BindWithConfig` skips *only*
that gate — every tensor-level check (type, dims, length, pairing table,
rotation marker, all-or-nothing per layer) still runs.

Consequence to keep visible in the report: this exercises the loader contract
and the executor path, **not** production config acceptance. Any output from
this run must not be described as a full-model or production-frontdoor run.

---

## 4. Graph / eager scope

**The rotated path is eager-only.** Dispatch option (a) selects each slot's
expert weight pointer in a host loop, so those offsets become kernel
arguments chosen by CPU code. Capturing that freezes today's routing into the
graph and replay can never route differently.

Enforced in code, not just by convention:

* `Executor::Forward`'s automatic decode-graph predicate now excludes
  rotated models outright (`rotated_experts_`, set once in `Create` from
  the loaded weights). A rotated model never enters the warm→capture→replay
  path, so repeated `Forward(kDecode)` calls stay eager and cannot hit the
  capture guard by accident. This closes the "guard without eager
  selection" gap: previously a matching decode shape warmed eagerly and
  then *attempted* capture on a later call.
* `Executor::MoeExperts` still returns an error if `capturing_` is set
  while the layer is rotated — defense in depth, retained deliberately.
* `s_.ids` (written by `RouterTopK` on the device stream, possibly in
  hipMalloc memory) is downloaded to pinned `ids_host_` and the stream is
  synchronised before the host loop reads it — so the ids are both
  addressable and fresh.

**The driver must assert all three:**

1. the eager path runs and produces correct output,
2. repeated real `Forward(kDecode)` calls (≥3) each succeed eagerly
   without attempting capture, and
3. a deliberate capture attempt on the rotated path **fails with that
   specific error**, rather than silently succeeding and baking routing.

A run that only does (1) is incomplete: it leaves the capture hazard
untested. Device-side selection is the future capturable path and is not part
of this contract.

---

## 5. What the driver must exercise

| # | Step | Why it is required |
|---|---|---|
| 1 | Build fixture, **assert its SHA-256 and size against §2 BEFORE any bind, upload or device work**, and re-hash the written file before opening the reader | pins the input; generator drift must not silently change what is tested. A peer-reported hash is not a run-input check |
| 2 | `BindWithConfig`, assert rotated flag + pairing table resolved | loader contract |
| 3 | Upload via real `DeviceModel` (F16→F32 widen of the 5 vectors) | upload path |
| 4 | `Executor::Create` + `Forward(kPrefill)` | prefill liveness (GEMM primitive); tested separately from final logits |
| 5 | **Repeated `Forward(kDecode)` (≥3 calls)** | each must stay eager and succeed; proves the eager selection, not just a first-call warmup |
| 6 | **Per-layer post-MoE seam parity for EVERY fixture layer**: drive `MoeExperts` with controlled nonzero `x` + controlled `s_.ids`, compare the routed `s_.down_e` against the dense materialised-W oracle of the SAME layer's weights and the SAME token's input | the numeric gate lives at the MoE seam, not at logits; a later-layer rotation-upload error must not escape |
| 7 | **Both dispatch shapes at that seam**: single-token decode-like (matvec primitive) and multi-token prefill-like (GEMM primitive) | a multi-token slot/input-stride bug is invisible to a one-token-only comparison |
| 8 | **Distinct nonzero input row per token**, verified distinct | if every token row were identical, reading the wrong token's row would be undetectable |
| 9 | **Both routing states** (ids `{0,1}` and `{1,0}`) each match their dense oracle | routing must be correct in both states, not merely different |
| 10 | Routing-change sensitivity: swapping ids changes the routed output by > 1e-2 | proves routing is live, not frozen |
| 11 | Assert capture refusal (§4), for every layer | graph-safety |
| 12 | Positive control (correct pairing) and negative control (deliberate mispair) at the same seam, with **zero-safe separation `negative_error > 100 * positive_error`** and both finite | discriminates; a positive of exactly 0 cannot fake the ratio |
| 13 | Finite + size checks on every comparison, **both sides**, enforced not assumed | NaN on either side cannot score as zero error |
| 14 | **Per-comparison evidence rows** per §9 schema, plus gate self-tests that reject malformed / non-finite / zero-reference input and a 1e-3 residual | the gate must be shown to reject what it claims to reject |

**The seam is the per-layer post-MoE routed output (`s_.down_e`), not the
final logits.** Forward decode/prefill are exercised for liveness and
finite outputs; the numerical parity gate is the `MoeExperts`-vs-dense
comparison at the MoE seam with matched input/state/weights. Do not compare
final logits against an isolated `ReferenceModel::Moe` vector — `Moe`
returns the softmax-weighted routed sum plus the shared expert, while
`MoeExperts` produces the unweighted per-slot routed down. The dense
materialised-W oracle and `ReferenceModel::Moe` share the same mathematical
basis (`W = diag(sv)·Hn·M·Hn·diag(su)`), validated against each other in
`i4r_moe_parity_test`.

---

## 6. Metric — one definition, no conflation

Three different normalisations exist across this project and they are **not**
interchangeable:

| Where | Definition |
|---|---|
| `i4r_dispatch_test` | per-element: `\|a-b\| / max(1e-6, max(\|a\|,\|b\|))` |
| `i4r_moe_parity_test` | **peak-normalised: `max\|a-b\| / peak(\|ref\|)`** |
| GPU44 | `max(1, \|reference\|)` denominator |

**This contract uses the peak-normalised metric** (row 2), because that is
what the CPU oracle comparison that grounds the threshold already reports.
Report the metric name alongside every number. Do not compare a
peak-normalised figure against a per-element threshold or a GPU44 figure.

---

## 7. Predeclared thresholds

Agree these **before** the run. Post-hoc threshold adjustment is not
acceptable; if a number lands outside these, the run is inconclusive and
comes back for discussion rather than getting a new limit.

| Check | Threshold | Basis |
|---|---|---|
| Size match (output vs reference) | exact, else **FAIL** | — |
| Finiteness (output **and** reference, every element) | all finite, else **FAIL** | `std::max(worst, NaN)` silently keeps `worst` |
| Non-vacuous reference | `peak > 1e-3`, else **FAIL** | guards an all-zero reference |
| **Primary parity** (GPU `MoeExperts` routed `down_e` vs dense materialised-W oracle, same input/weights, peak-normalised) | **< 1e-4** | CPU rotated-vs-dense is ~2.5e-7; device reduction order may add, but should stay far under 1e-4 |
| Both routing states | each state independently passes the primary gate | routing must be correct, not merely different |
| Decode vs prefill liveness | both Forward modes succeed with finite outputs | different primitives (matvec vs GEMM) |
| Routing-change sensitivity | changed routing changes the routed output by > 1e-2 | routing must be live |
| Negative control (deliberate mispair at the seam) | > 1e-2 | CPU shows ~1e-1 |
| Separation positive:negative | **zero-safe: `negative_error > 100 * positive_error`**, both finite | CPU shows ~10^6; a positive of exactly 0 cannot fake the ratio |

**Status semantics.** A numeric mismatch, a non-finite value, a size
mismatch, or a failed control is a **FAIL**, not a pass and not a
post-hoc looser limit. A missing artifact, a timeout, or a SKIP is an
explicit **non-pass** with its own status — it is not "inconclusive" in a
way that obscures a known assertion failure. §10's generic inconclusive
applies only to genuinely undetermined evidence, never to a check that
fired and failed.

CPU grounding (established prior CPU-suite figures from `i4r_moe_parity_test`,
peak-normalised — carried from the earlier revision, NOT a fresh v5/v6
re-measurement; the suite re-runs green on the current tree but these exact
values are the prior ones, reported as peer figures rather than implied fresh):

```
layer 0: 2.564e-07   layer 1: 2.927e-07
layer 2: 2.426e-07   layer 3: 2.495e-07
negative control: 2.904e-01
```

These are the post-F16-round-trip figures. The pre-fix figures were
2.869e-04 / 2.859e-04 / 5.354e-04 / 5.004e-04 — roughly 1000x worse,
which is what confirmed the residual was scale mismatch rather than
Hadamard-chain error.

---

## 8. Allocation accounting — serialized and source-derived

**What these numbers are and are not.** The tables below are **serialized
GGUF payload sizes** and **source-layout scratch sizes** — derived from the
fixture's declared shapes and the `ExpertStageLayout` arithmetic. They are
**not** measured device-resident allocation. Upload widens the five F16
vectors to F32 and adds allocator margins; real whole-device accounting
also includes the other weights, executor scratch, staging, and the
HIP/BLAS/driver baseline, none of which this box can measure. Host layout
bounds are not atomic GTT containment or fit guarantees. The operator sets
conservative hard ceilings **before** proposing any job; a first authorised
run validates observed use **within** them and is not an uncapped
discovery run used to pick them after the fact.

### Serialized weights, tiny fixture

Corrected after review of an earlier estimate. The per-tensor
figure is `n_out * (n_in / 128) * 66 * experts`.

| Item | Bytes |
|---|---|
| One gate (or up) expert tensor, per layer | **33,792** |
| gate + up + down, one layer | **101,376** (1,536 blocks x 66 B) |
| gate + up + down, **all 4 layers** | **405,504** |
| 5 rotation vectors per layer, F16 | **1,792** (256+128+128+128+256 elems x 2) |
| Rotation vectors, all 4 layers | **7,168** |
| Rotated payload, all 4 layers | **412,672** |
| Whole fixture GGUF (all tensors incl. dense/router/shared/attention) | **7,529,856** |

For three experts the I4R weights alone would be 608,256 B. The
**405,504 B is weights only** — it does not include rotation vectors,
dense/router/shared/attention weights, upload padding, executor scratch or
HIP context. An earlier mail quoted ~188 KiB as a total by mixing a
single-layer weight estimate with four-layer scales; that figure is wrong
and is superseded by this table.

### Launch count (revised out-of-place path)

Counted from the revised dispatch, not inherited from the pre-fix estimate:

| Unit | Count |
|---|---|
| Input rotation | 1 per token (`RotateLayerInput`) |
| Per selected expert slot | **7** — gate matmul, up matmul, rotate gate, rotate up, rotate down-in, down matmul, rotate down-out |
| Per token, `used = 2` | `1 + 7 * 2 = 15` |
| 3 tokens x 4 layers | **180 rotated launches** |

Plus SwiGLU and the rest of the executor's per-layer work. The earlier
"~96 total" was wrong; out-of-place rotation added launches by design
(finding 4). Dispatch option (a) is launch-heavy and makes no speed claim.

### Source-layout stage scratch (rotated expert stage)

From `ExpertStageLayout`, measured:

| T (max batch) | slots | stage floats | stage bytes |
|---|---|---|---|
| 1 | 2 | 1,920 | 7,680 |
| 2 | 4 | 3,200 | 12,800 |
| 4 | 8 | 5,760 | 23,040 |
| 8 | 16 | 10,880 | 43,520 |
| 32 | 64 | 41,600 | 166,400 |

The stage block is the max over the sequential linear / attention / expert
stages, so at these tiny shapes the linear or attention term may dominate
and absorb the rotated slices. The expert-dominant case that *would* have
overrun before the fix is covered by `i4r_stage_budget_test` (old 131,072 vs
needed 139,904 floats at slots=256, ff=128, hidden=256, T=32).

### Not measured here — ceilings precede any run

The following are device/driver-level and cannot be measured from this box.
**No estimate is offered deliberately**; a prior note that gave a combined
figure without a measurement would have been invented.

The operator sets conservative hard ceilings for these **before** proposing any
build or run, from justified reasoning about the platform. A first
authorised run then checks observed use **against** those ceilings. It is
not the source of them: there is no uncapped discovery run and no
post-hoc cap selection. The items that need a ceiling set ahead of any run:

* HIP context + runtime baseline resident
* driver / queue / graph-instantiation overhead
* GTT aperture usage
* host RSS of the test process
* wall-clock per decode step and per prefill step

What *is* bounded above: the rotated-path-specific allocations are the
tables in this section. Anything beyond them is baseline executor cost, not
rotated-path cost, and should be attributed that way.

---

## 9. Output, timeout, disk

* **Raw output**: full stdout/stderr of the driver, unedited, retained with
  the run. Any assertion failure must print the failing index, both values,
  and the metric name.
* **Result schema**: per (mode = decode/prefill, layer) — peak, max_abs,
  peak_normalised_rel, PASS/FAIL; plus the control lines (positive,
  negative, routing-change, capture-refusal) each PASS/FAIL.
* **Timeout**: the fixture is ~7.2 MiB with 4 layers and ≤32 tokens; the
  work is launch-bound, not compute-bound. A **300 s** hard timeout is
  *proposed* for the whole driver — if it exceeds that, something is wrong
  (launch storm or hang), not slow. This is a proposal only: a timeout and
  its signal cannot guarantee recovery from a D-state stall. The actual
  timeout, signal handling and queue wrapper are set by the operator, not
  by this contract.
* **Disk**: the driver's own writes are bounded (fixture ~7.2 MiB plus
  logs; no model output, no checkpoints). That bound is **added to** the
  existing external model/disk reserve policy, which remains mandatory —
  small driver output does not make the reserve negligible.
* **No discovery probe.** Do not run this to find out how big things are.
  The bounds above are the contract; the run either fits them or fails.

---

## 10. Acceptance

A run is a PASS only if every §7 check passes, both modes pass
independently, the capture-refusal assertion fires, and the raw output is
retained. Anything else is inconclusive and returns for discussion — not a
threshold adjustment.

GPU execution remains a separate decision from this contract being agreed.
