# 2×2 Framing × Realism — Results

**Design.** Framing (narrative / numeric) × Realism (real / fictional), run on two
local Ollama models (`llama3.1:8b`, `mistral:7b`). 4 matched event families
(lithium, semiconductor, energy, credit) × 4 cells × 5 reps × 2 models =
**160 runs**, all completed, 0 failures. The interaction is the standard 2×2
contrast, estimated per family and bootstrapped over families:

```
interaction = (narrative_real − numeric_real) − (narrative_fictional − numeric_fictional)
```

The primary CI is a **cluster bootstrap that resamples the 4 event families**
(the unit of independent replication), 1000 iterations, percentile 95% CI.

## Headline

**No framing × realism interaction, and the null replicates across both models.**
On every primary metric, each model's interaction CI spans zero and the paired
cross-model difference spans zero. For a study built to detect whether the models
treat *real* shocks differently from structurally-matched *fictional* ones, this
is a clean, consistent null — not evidence of a memorization back-door.

## Per-metric interaction (95% CI over 4 families)

| Metric | llama3.1:8b | mistral:7b | Paired diff (llama − mistral) |
|---|---|---|---|
| **spread widening** (primary) | −0.04335 [−0.14477, 0.02490] · spans | +0.01775 [−0.06722, 0.10686] · spans | −0.06110 [−0.17741, 0.05322] · spans |
| **Kyle's λ change** (primary) | +0.00364 [−0.00063, 0.00845] · spans | −0.00014 [−0.00894, 0.00800] · spans | +0.00379 [−0.00490, 0.01375] · spans |
| **archetype divergence** (primary) | −0.00012 [−0.00032, 0.00003] · spans | −0.00005 [−0.00012, 0.00001] · spans | −0.00007 [−0.00028, 0.00011] · spans |
| Kyle's λ post-level | **+0.00477 [0.00079, 0.00987] · excludes** | −0.00124 [−0.00761, 0.00396] · spans | +0.00601 [−0.00068, 0.01323] · spans |
| spread post-level | −0.06245 [−0.16833, 0.01070] · spans | +0.01015 [−0.09305, 0.10526] · spans | −0.07260 [−0.18951, 0.02811] · spans |

**The one non-null cell** is llama's interaction on the *post-shock level* of
Kyle's λ (CI excludes zero). It does not survive scrutiny: mistral does not show
it, the paired cross-model difference spans zero ("a difference in significance is
not a significant difference"), and it is 1 of 5 metrics with no multiplicity
correction. Treat it as a lead, not a finding.

## Decision health

| Model | hold rate | forced coin-flips | backend failures |
|---|---|---|---|
| llama3.1:8b | 4.9% | 0.5% | 0.2% |
| mistral:7b | 34.6% | 9.5% | 0.1% |

mistral's high hold rate is **genuine trading behaviour, not JSON unreliability**:
only 5 parse failures were logged across ~4,800 mistral calls. It is simply a more
passive trader. But 9.5% of its "trades" are coin-flip fallbacks (random orders
after 3 consecutive holds), so mistral's null is a lower-participation, noisier null
than llama's.

## Stated limitations

1. **Numeric-framing realism is carried by the entity token alone.** Within each
   matched pair, all numeric values are identical by construction, so in the two
   numeric cells "real vs fictional" differs only by one name token (e.g.
   `lithium` vs `seldrite_carbonate`). The realism manipulation is therefore
   **weaker in the numeric cells than in the narrative cells**. Practical
   consequence: part of the null interaction — specifically anything driven by the
   numeric cells — may reflect a weak manipulation rather than a true absence of
   effect. This is a property of what numeric framing can carry; readability
   matching cannot fix it. *(Documented rather than re-run, by decision, given time
   constraints.)*

2. **Fictional shocks are structural clones of the real templates.** Each fictional
   narrative is a slot-for-slot rewrite of a real one with entities renamed. No
   fictional shock resembles a nameable, dateable real event (the memorization
   constraint is met at the entity level), but the *causal structure* is shared by
   design — so entity-level recall is controlled while structural recall is not.

3. **n = 4 families** → weak support for generalising to supply-chain shocks broadly;
   the 4-family cluster bootstrap has approximately nominal coverage under Gaussian
   family effects but the intervals are wide.

4. **n = 2 models is descriptive, not inferential.** The "combined" estimate averages
   the two per-model interactions; it does not treat models as a resampled sample and
   carries family-level uncertainty only. Both models are small local Ollama models,
   so agreement is evidence about **replication**, weak evidence about generalisation
   across model families (which would need models differing in scale and training).

5. **A replicated null is a statement about consistency, not proof of absence** — with
   4 families the CIs are wide and modest true effects would not be detected.

6. **Stimulus provenance.** These results used the fictional-shock stimuli as they
   existed at sweep time, which included a `3nm` process-node token in the fictional
   `lattice_substrate_shortage` text (a real-node leak). That token has since been
   replaced with a fictional designation (`K3`) in `generate_fictional_shocks.py` and
   `shocks.json`, balance-preserving (word count / Flesch / numeric magnitudes
   unchanged). **The completed 160-run sweep predates this fix**; any future
   replication uses the de-leaked stimuli.

## Reproduce

```
python3 analyze_2x2.py --compare-models        # cross-model interaction + paired contrast
python3 -c "from experiments.design_2x2 import report_sweep_health as h; h('llama3.1:8b'); h('mistral:7b')"
python3 check_shock_balance.py                 # stimulus balance + numeric-realism caveat
```

Raw runs: `results_2x2/{llama3.1_8b,mistral_7b}/{framing}__{realism}/{event}__rep{NN}.json`,
plus a `manifest.json` per model.

---

## Reasoning-Text Leakage Analysis (Step 1)

Read-only textual companion to the behavioral result above (`analyze_reasoning_leakage.py`, no new runs). It scans every LLM agent's `reasoning` field for **real-world recall beyond the shock prompt** — named firms/places, dated crises, or explicit "like the real X" comparisons. Fictional shocks have an empty lexicon by construction (no real referent to recall). A term counts only if it is in the reasoning and NOT in the prompt that record was given, so the same term is in-prompt (narrative) yet out-of-prompt (numeric).

- **Records scored:** 9,102 LLM decisions across 2 models (market-maker, coin-flip, and backend-failure records excluded).
- **Tier 1 — specific recall markers:** 0 instances.
- **Tier 2 — generic inferable synonyms** (crude→"oil", chip→"semiconductor"; reported apart, not counted as recall): 17 instances.

**Tier-1 leakage rate per cell (combined, both models):**

| framing × realism | instances / records | rate |
|---|---|---|
| narrative × real | 0 / 2278 | 0.000% |
| numeric × real | 0 / 2271 | 0.000% |
| narrative × fictional | 0 / 2272 | 0.000% |
| numeric × fictional | 0 / 2281 | 0.000% |

**Pattern:** narrative+real is **not** elevated — leakage is flat (effectively zero) across all four cells, both models.

**Relation to the behavioral null:** this is **independent confirmation** that the replicated behavioral null is real, not a measurement artifact. There is no hidden lexical real-world-recall signal that the trading metrics missed; real and fictional shocks are processed alike in text as in behavior. The only near-misses are Tier-2 synonyms ("oil" for `crude_futures`, "semiconductor" for `chip`) that are trivially inferable from the prompt and are not evidence of event recall.

**Scope/limitation:** this detects *lexical* recall only. A model influenced by a real event without naming any specific entity would not be flagged, so absence of lexical leakage is evidence — not proof — of no recall.

---

## Causal Mediation Extension (Steps 1–2)

Extends the behavioral 2x2 with a mediator. **Step 1** probed each agent's *perceived* realism M (`probe_perceived_realism.py`, 9,600 calls); **Step 2** (`analyze_mediation.py`) decomposes framing's effect on behavior into a part running *through* M (natural indirect effect, NIE) and a part that does not (natural direct effect, NDE), family-cluster bootstrap, 1000 iters.

**Step 1 — manipulation check (fraction perceiving 'real'):**

| model | framing | P(real \| real) | P(real \| fictional) | separation |
|---|---|---|---|---|
| llama3.1:8b | narrative | 85% | 34% | +51% |
| llama3.1:8b | numeric | 56% | 43% | +14% |
| mistral:7b | narrative | 100% | 94% | +6% |
| mistral:7b | numeric | 100% | 100% | +0% |

Only **llama3.1:8b + narrative** shows a clean separation; **mistral:7b** answers “real” to ~94–100% of everything (no realism discrimination), so its mediator is near-constant and its NIE is ~0 by construction, not by finding.

**Step 2 — natural indirect effect (NIE) of framing through M, per metric:**

| model | metric | Total | NIE (indirect) | NDE (direct) | NIE vs 0 |
|---|---|---|---|---|---|
| llama3.1:8b | spread_widening | -0.0053 | -0.0017 [-0.0188, 0.0181] | -0.0036 | spans |
| llama3.1:8b | kyle_lambda_delta | 0.0016 | 0.0001 [-0.0007, 0.0016] | 0.0015 | spans |
| llama3.1:8b | archetype_divergence | -0.0000 | 0.0000 [-0.0000, 0.0001] | -0.0000 | spans |
| mistral:7b | spread_widening | -0.0380 | -0.0011 [-0.0122, 0.0060] | -0.0369 | spans |
| mistral:7b | kyle_lambda_delta | 0.0016 | -0.0005 [-0.0034, 0.0014] | 0.0021 | spans |
| mistral:7b | archetype_divergence | 0.0000 | 0.0000 [-0.0000, 0.0000] | -0.0000 | spans |

**Result:** every NIE spans zero. There is **no evidence** that framing affects behavior *through whether agents believe the event is real*. Combined with the flat behavioral interaction and the flat reasoning-text leakage, the belief channel carries no measurable causal weight here — and for mistral the question is moot because the manipulation never registered perceptually.

*Caveats:* n=4 families (wide CIs); linear mediation, no framing×M interaction; M probed without order-book state so it is collinear with the design; n=2 models is descriptive. This is associational mediation — Step 3 (interventional M manipulation) would be needed to confirm.

---

## Interventional Confirmation (Step 3)

Step 3 manipulates the mediator directly (`generate_anonymization_variant.py`,
80 new runs, 0 probe errors). On the 4 real-narrative shocks, an appended
instruction forces perceived realism down; runs are **seed-paired** to the
originals (`make_seed` depends only on family+rep). A **placebo arm** carries the
same "for this evaluation" frame with **no reality claim**, so `anon − placebo`
isolates the belief manipulation from the mere eval-framing.

**Intervention manipulation check** — M = P(perceived "real"):

| model | original | + placebo | + anon |
|---|---|---|---|
| llama3.1:8b | 84.9% | 57.2% | **0.0%** |
| mistral:7b | 100.0% | 100.0% | **0.0%** |

The anon instruction forced M to 0% for **both** models (manipulation landed —
even mistral, whose Step-1 yea-saying was overridden by the explicit instruction).
Note the placebo *also* moved llama's M (85%→57%): the eval-frame alone lowers
perceived realism — exactly the confound the placebo controls for.

**Paired behavioral effect (family-cluster bootstrap, 1000 iters, same seeds) —
`anon − placebo` isolates belief:**

| model | metric | anon − placebo [95% CI] | vs 0 |
|---|---|---|---|
| llama3.1:8b | spread widening | −0.0136 [−0.042, +0.008] | spans |
| llama3.1:8b | Kyle's λ change | −0.0013 [−0.0029, +0.0002] | spans |
| llama3.1:8b | archetype divergence | −0.00006 [−0.00014, +0.00001] | spans |
| mistral:7b | spread widening | −0.0015 [−0.051, +0.030] | spans |
| mistral:7b | Kyle's λ change | −0.0017 [−0.011, +0.006] | spans |
| mistral:7b | archetype divergence | −0.00002 [−0.0001, +0.00004] | spans |

**Result: all six `anon − placebo` CIs span zero.** Forcing perceived realism from
~85–100% down to 0%, with the eval-frame held fixed, did **not** move trading
behavior on any metric, either model — **interventional confirmation** that the
belief-that-it's-real channel carries no causal weight, upgrading Step 2's
associational null.

**Why the placebo was essential:** llama's Kyle's-λ shows `anon − original`
*excluding* zero (−0.0022 [−0.0041, −0.0002]) — which, uncontrolled, would read as
mediation. But `anon − placebo` spans zero: that shift is the **eval-framing**, not
the belief manipulation. The placebo reassigns it correctly, averting a
false-positive mediation claim.

**Whole chain agrees:** behavioral 2×2 interaction null (replicated) → reasoning-text
leakage flat → mediation NIE spans zero (associational) → interventional
`anon − placebo` spans zero. The "does the model treat real vs fictional
differently" question is negative **behaviorally, textually, and causally**.

*Caveats:* n=4 families (wide CIs — a small mediated effect could still hide);
linear paired design; both are small local models; interventional *within this
simulation*, not a claim about LLMs in general.

---

## Statistical Power (n = 4 families)

Minimum detectable effect (MDE) at 80% power, two-sided α=0.05: `MDE = 2.80 × SE`, SE from the family cluster bootstrap. This makes the "n=4 is small" caveat concrete (`power_analysis.py`, no new LLM calls).

**llama3.1:8b** (n = 4 families):

| effect | estimate | SE | MDE (80%) | observed detectable? | n for ½·MDE |
|---|---|---|---|---|---|
| interaction · spread_widening | -0.04335 | 0.04350 | 0.12188 | **no** | 16 |
| NIE · spread_widening | -0.00170 | 0.00898 | 0.02516 | **no** | 16 |
| interaction · kyle_lambda_delta | 0.00364 | 0.00225 | 0.00629 | **no** | 16 |
| NIE · kyle_lambda_delta | 0.00009 | 0.00080 | 0.00223 | **no** | 16 |
| interaction · archetype_divergence | -0.00012 | 0.00009 | 0.00026 | **no** | 16 |
| NIE · archetype_divergence | 0.00000 | 0.00002 | 0.00007 | **no** | 16 |

**mistral:7b** (n = 4 families):

| effect | estimate | SE | MDE (80%) | observed detectable? | n for ½·MDE |
|---|---|---|---|---|---|
| interaction · spread_widening | 0.01775 | 0.04455 | 0.12481 | **no** | 16 |
| NIE · spread_widening | -0.00109 | 0.00464 | 0.01301 | **no** | 16 |
| interaction · kyle_lambda_delta | -0.00014 | 0.00411 | 0.01150 | **no** | 16 |
| NIE · kyle_lambda_delta | -0.00045 | 0.00130 | 0.00364 | **no** | 16 |
| interaction · archetype_divergence | -0.00005 | 0.00003 | 0.00009 | **no** | 16 |
| NIE · archetype_divergence | 0.00001 | 0.00001 | 0.00004 | **no** | 16 |

**Reading:** across the board the observed estimates are **smaller than their own MDE**, so this design could not have detected effects of the observed magnitude even if they were real. The interaction, mediation, and interventional nulls are all **underpowered nulls** — consistent with no effect, but not proof of zero. Detecting an effect half the current MDE would need roughly **4× the event families** (the `n for ½·MDE` column); e.g. moving from 4 to ~16 families.

---

## Mediator Validity & Rigor Checks (closing)

Three closing checks on the causal-mediation work.

**1. Is mistral's mediator valid? — No: it is a response-format artifact.**
Step 1 found mistral answered "real" ~100% for both real and fictional shocks.
Diagnosis (`diagnose_mistral_mediator.py`, ~70 local calls):

- *Absurd-scenario check* — on 5 physically impossible / self-contradictory
  scenarios, mistral answered "real" **0%** of the time (llama also 0%). So
  mistral is **not** a blanket yea-sayer; it rejects the clearly impossible.
- *Phrasing variants* — on a real + fictional subsample, mistral's P("real") was
  driven entirely by question **format**, not content: 100% when "real" was listed
  first, 0% when "hypothetical" was first or the question was open-ended yes/no,
  and **0% real-vs-fictional separation under every phrasing**.

**Conclusion:** mistral cannot discriminate real from structurally-matched
fictional shocks; its probe answer reflects option-order primacy, not perceived
realism. **mistral's mediator is unusable.** The original 100% was an artifact of
"real"-listed-first plus the fictional shocks being genuinely plausible — not a
belief.

**Evidence is therefore asymmetric across models** (this corrects any symmetric
reading in the sections above):

| leg | llama3.1:8b | mistral:7b |
|---|---|---|
| behavioral 2×2 interaction | ✓ | ✓ |
| reasoning-text leakage | ✓ | ✓ |
| interventional (behavior vs. instruction) | ✓ | ✓ *(behavioral only)* |
| **perceived-realism mediator (M)** | ✓ valid | **✗ invalid (format artifact)** |
| **observational mediation (NIE)** | ✓ | **✗ (mediator invalid)** |

The **perceptual and observational-mediation legs rest on llama alone.** llama's
mediator is validated: it rejects the impossible (0% "real") and discriminates
real vs fictional (+51% separation, narrative), so it tracks content, not format.
mistral still contributes to the behavioral, textual, and interventional legs
(where its behavior is measured directly, not via the probe) — but its Step-3
interventional result is best read as "the anonymization *instruction* did not
change mistral's trading," since we cannot certify it moved mistral's *belief*.

**2. Probe reactivity — precluded by construction.** The mediator probe was never
part of the trading-decision prompt (verified: no probe phrasing in `agents.py`
or any stored run). It is a separate, stateless, post-hoc Ollama call run after
the sweep; the trading agents never saw it, so it cannot prime later ticks.
Reactivity is architecturally impossible — no empirical re-run is required to
establish it.

This paragraph originally predicted that a probe-free replicate "would only
reproduce the originals." It was run, and that prediction was **wrong**: at a
fixed seed, 23-32% of llama/mistral decisions flip (see "Probe Reactivity &
Reproducibility"). The architectural argument survives — the difference is
temperature-0.7 sampling, not probe contamination — but the re-run was not
redundant. It measured a reproducibility floor of the same order as the effects
being estimated, which is why it now anchors the power discussion.

**3. Statistical power.** See "Statistical Power (n = 4 families)": every observed
estimate is below its own MDE, so all nulls are underpowered nulls; halving the
detectable floor needs ~16 families (4x).

---

## Probe Reactivity & Reproducibility (Step 2)

Empirical companion to the architectural argument (`check_probe_reactivity.py`,
30 stratified replicates, 10/model across three models). A subsample was re-run
**probe-free at the same seed** and compared tick-by-tick to the originals.

**Reactivity is impossible at two levels** (both verified in code): (1) the probe
was never in the trading prompt — it is a separate, stateless, post-hoc call;
(2) agents are stateless per tick — each tick builds a fresh prompt with no
conversation history, so nothing from tick *t* (probe or otherwise) can reach
*t+1*. A probe-free re-run therefore cannot differ by probe effects; it differs
only by LLM temperature-0.7 sampling. That is the pipeline's **reproducibility
floor**:

| model | lineage | agreement | flip rate | κ | spread-widening mean·\|Δ\| | Kyle's λ Δ mean·\|Δ\| | archetype div. mean·\|Δ\| |
|---|---|---|---|---|---|---|---|
| llama3.1:8b | llama | 76.7% | 23.3% | 0.303 | 0.040 | 0.0074 | 0.00014 |
| mistral:7b | llama-adjacent | 68.0% | 32.0% | 0.415 | 0.072 | 0.0050 | 0.00007 |
| **gemma2:9b** | **gemma** | **94.5%** | **5.5%** | **0.787** | **0.0102** | **0.0012** | **0.00007** |

n = 600 decisions per model. κ is Cohen's kappa — see "why κ" below.

**Reading.** On llama and mistral, 23–32% of LLM decisions flip on a same-seed
re-run, so a single run's metrics are genuinely noisy. The run-to-run \|Δ\| on
spread widening (~0.04–0.07) is the **same order as the observed interaction
estimate (~0.04)** and well below its MDE (~0.12) — i.e. the effects are
**noise-dominated at n=4**, exactly as the power section implies. This within-run
noise is precisely what the family-cluster bootstrap resamples over, so the
reported CIs already integrate it (llama's spread-widening bootstrap SE, 0.044,
matches this \|Δ\| floor). Conclusion: **no probe reactivity** (impossible by
construction, and nothing anomalous empirically) — the nulls are honest but
noise-limited, not contaminated.

### Does the instability generalise across model families? No.

gemma2:9b was added specifically as a **different architecture lineage**, not as
a third data point — llama3.1 and mistral are close enough that agreement
between them is weak evidence about LLMs in general. It runs the *identical*
selection at the *identical* seeds (seeds are model-independent), so all three
models re-decide the same 20 ticks of the same book.

It flips **5.5%** of decisions against llama's 23.3% and mistral's 32.0% — 4.2x
and 5.8x lower. The gap is not confined to the decision metric: its spread-widening
reproducibility floor is **3.9-7x tighter** and its per-tick volume \|Δ\| is 4.43
against 9.73 / 10.73. Those metrics do not share the decision-agreement
denominator, so this is not one number restated three ways.

**So run-to-run instability is a property of the llama/mistral lineage as tested
here, not of local 8B-class models re-deciding a fixed book.** Any claim of the
latter form should be scoped to the lineage until more architectures are run.

Two artifacts ruled out before believing it:

- **Not degeneracy.** An all-HOLD model would post a low flip rate trivially.
  gemma2 trades actively — BUY 84.7% / SELL 10.0% / hold 5.3%, close to llama's
  mix. It is genuinely re-deciding, and re-deciding the same way.
- **Not a base rate.** *Why κ:* raw agreement is not comparable across models
  whose action mixes differ, and these differ a lot — mistral holds on 36.5% of
  decisions and llama on 5.5%, which moves the chance-agreement floor from 45.3%
  to 66.5%. gemma2's concentrated mix gives it the *highest* chance floor of the
  three (74.1%), so the raw number flatters it most; chance-corrected it still
  leads by ~2.6x (κ 0.787 vs 0.303).

**Correction to the earlier framing.** This section previously read "~23-32% of
LLM decisions flip", which invites reading llama as the stable end of a range.
Chance-corrected, the ordering **inverts**: mistral (κ 0.415) is *more*
self-consistent than llama (κ 0.303), and its higher raw flip rate is largely its
36.5% hold rate depressing its chance floor. Quote κ when comparing models; quote
the raw flip rate for one model's absolute noise level.

**Caveat on scope.** gemma2's 10 replicates cover 2 event families and 4 cells —
the same stratified subsample llama and mistral used, so the comparison is
like-for-like, but it is a reproducibility probe, not a sweep. gemma2 was **not**
run through the main 2x2 design and contributes nothing to any interaction
estimate. Its first draw lives in `results_2x2_reactivity_base/`, deliberately
outside `results_2x2/`, so it cannot be picked up as a third arm of the study.
