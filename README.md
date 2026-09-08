# Synaptic-Plasticity-as-Short-Term-Memory
Interactive Hebbian network explorer - watch synaptic plasticity store, decay, and retrieve memories in real time, with a sharp failure boundary you can break yourself.

---

## The One-Sentence Claim

> A recurrent network with Hebbian synapses can store and retrieve multiple patterns from partial cues, but when retrieval uses a **fixed absolute threshold**, the trade-off between learning rate (η) and synaptic decay (λ) produces a **sharp failure boundary rather than a smooth decline** , because decay shrinks recurrent inputs toward the fixed threshold until retrieval collapses abruptly.

**Why this claim is falsifiable:** the artifact computes recall quality (overlap with the stored pattern) live for every (η, λ) pair. A learner can scan the phase space and directly observe whether the transition is sharp or smooth. If recall degraded linearly with decay, the claim would be refuted.

> **Implementation note (stated, not hidden):** recall uses an *absolute* firing threshold θ, not a sign threshold. Under a pure sign threshold, uniform decay multiplies every input by the same factor and cannot change recall — so the sharp-boundary effect depends specifically on the absolute threshold. The artifact lets you toggle between the two to verify this.

---

## Why This Concept, Why Now

Synaptic plasticity as short-term memory is an old idea returned in an important new form. Hebb (1949) proposed that recent activity strengthens connections; Hopfield (1982) showed such networks retrieve stored patterns from partial cues. For decades this lived mostly in computational neuroscience.

It matters **now** because **BDH (Dragon Hatchling) reformulates attention itself as synaptic memory** — in its conceptual neuron–synapse model, reading a token performs a Hebbian write into fast, decaying connections, turning the network's wiring into working memory. The same η/λ/θ trade-off you explore here is the design tension BDH navigates to make contextual memory work. This artifact teaches the mechanism BDH builds on, from first principles, using a live toy system you can break.

---

## For Whom & Prerequisites

**Intended learner:** ML practitioners comfortable with linear algebra and basic neural-network concepts. No prior knowledge of Hopfield networks, Hebbian learning, or BDH is assumed.

**Prerequisites:**
- Vectors, matrices, and the outer product
- What an activation/threshold function does
- No associative-memory or BDH background required

---

## Learning Objectives

After interacting with this artifact, the learner will be able to:

1. **Explain** how Hebbian plasticity stores patterns via outer-product writes.
2. **Predict** how the η/λ trade-off creates a sharp failure boundary under a fixed absolute threshold.
3. **Describe** how BDH turns attention into synaptic memory through Hebbian writes, and which quantity is actually changing (synaptic weights / fast state, not slow trained parameters).
4. **Diagnose** failure modes: decay, cue damage, and pattern interference.
5. **Distinguish** a sign threshold from an absolute threshold and state why the sharp-boundary claim depends on the latter.
6. **Name** at least one limitation of associative memory (capacity limits, interference) and one thing this toy model does *not* capture about BDH.

---

## The 60-Second Test

The core claim is reproducible in under a minute:

1. The page loads with a preset already running (3 patterns stored).
2. In **Section 3**, drag the **λ (decay)** slider from 0 upward.
3. Watch the recall-overlap readout stay high, then **drop off a cliff** as decay pushes recurrent inputs below θ.
4. Toggle **sign vs. absolute threshold** — under sign threshold the cliff disappears, confirming the mechanism named in the claim.

---

## Artifact Architecture

A single self-contained HTML file that runs entirely in the browser.

| Component | Description |
|-----------|-------------|
| **Frontend** | HTML5 + CSS3, Red Hat Display typeface, dark/light theme, responsive |
| **Visualisation** | Canvas rendering of network states, weight heatmaps, energy landscapes |
| **Computation** | Pure JavaScript Hebbian network — no external libraries or API calls |
| **Interactivity** | Real-time sliders, click-to-draw editing, hover tooltips, audio feedback |
| **State** | In-memory weight matrices, pattern storage, recall history |

---

## Major Components & Their Roles

### Core model — `HebbNet` class
- `W` — synaptic weight matrix (N×N)
- `store()` — outer-product Hebbian write: `Δw_ij = η · x_i · x_j`
- `decay()` — multiplicative decay of all weights by `(1 − λ)` per tick
- `recall()` — iterative thresholded retrieval (absolute **or** sign threshold, selectable), with optional BDH-style fast-weight coupling

### Interactive sections

| # | Section | Purpose | Key controls |
|---|---------|---------|--------------|
| 1 | Flash, store & forget | Sequential storage, decay, recall | Show patterns, age, reset |
| 2 | Hebbian vs GD | Local plasticity vs global optimisation | Hebbian write, GD epochs, reset |
| 3 | Decay & energy landscape | Phase transition via Hopfield energy | λ, θ, time, **threshold mode** |
| 4 | Memory blender | Interference between two patterns | Pattern selectors, age gap |
| 5 | BDH synapse model | Hebbian learning ↔ BDH fast-weight memory | Coupling toggle, sparsity, KV-cache compare |
| 6 | Sandbox | Draw patterns, explore η–λ phase space | Click-to-draw, sliders, microscope |
| 7 | Parameter tasks | Active learning, falsifiable experiments | λ search, diagnosis, threshold test |

### Visualisation tools
- Network state view (neurons as nodes, synapses as edges)
- Weight heatmaps
- Energy landscape (2D contour with rolling-ball state)
- Interference heatmap (pattern–pattern overlap)
- Neuron current bars with the θ threshold line drawn

---

## Live vs Precomputed vs Animated vs Synthetic

| Element | Type | Notes |
|---------|------|-------|
| All network computation | **Live** | Every interaction computes in-browser |
| Pattern storage & recall | **Live** | No pre-recorded animation |
| Energy landscape | **Live** | Computed from the current weight matrix |
| Auto-play demo | **Live** | Timed real store/decay operations, not a video |
| Capacity / interference analysis | **Live** | Generated on the fly from random patterns |
| GD training | **Live** | Real gradient-descent epochs in the browser |
| BDH reported results (≈5% sparsity, monosemantic synapses, power-law connectivity, 1B→600B scaling) | **Precomputed / Reported** | Cited from the Dragon Hatchling paper, **not** computed here |

> **Evidence discipline:** BDH numbers are *reported by its developers*, not independently reproduced. This toy model is a teaching reimplementation and is **not** an official BDH model.

---

## The BDH Connection

Somewhere in the journey (Section 5), the artifact connects this concept to BDH. The link is precise: **BDH reformulates attention as synaptic memory that updates as the model reads.** Where a Transformer compares every token pair and grows a KV cache, BDH performs a Hebbian write into fast, decaying synaptic weights — the same `store()` + `decay()` dynamics you control here.

What's actually changing in BDH is the **fast synaptic state** (activity-dependent connection strengths), **not** the slow trained parameters. The module grounds this in:
- The Hebbian write equation from the Dragon Hatchling paper
- Reported BDH properties: ≈5% sparse non-negative activations that vary with predictability; monosemantic synapses encoding semantic concepts; scale-free (power-law) connectivity in trained models
- A KV-cache comparison showing how fixed-size synaptic memory differs from a growing cache

Any reimplementation here is a **toy** and is labelled as such.

---

## How to Reproduce Results

1. Open `prototype.html` in any modern browser (Chrome, Firefox, Edge, Safari).
2. **No installation, build step, data files, or external dependencies.**
3. For a clean experiment, click **Reset** in the relevant section.
4. For capacity/interference analysis, click **Add random pattern** repeatedly.
5. For GD training, click **Run 50 GD epochs** in Section 2.
6. To verify the sharp-boundary claim: sweep λ in Section 3 under both threshold modes and compare the recall-overlap curves.

### Setup (local)
No setup is required — the artifact is a single self-contained HTML file. To run locally, clone the repo and open `index.html` directly.

---


**Public artifact URL:** `synaptic-plasticity.netlify.app`

---

## Primary Sources & Citations

Recent primary papers (2022–2026) that use, extend, or rely on this concept, cited beside the claims they support:

1. **Pathway (2025).** *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* arXiv:2509.26507.  
   *Supports: BDH reformulates attention as synaptic memory; ≈5% sparse non-negative activations; monosemantic synapses; power-law connectivity.*

2. **Pathway (2026).** *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* arXiv:2608.09888.  
   *Supports: contextual memory as fast-weight/associative state; adaptation in recurrent state rather than weights.*

3. **Millidge, B., Song, Y., Bogacz, S., Lukasiewicz, T., & Bogacz, R. (2024).** *Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Paradigms.* Neural Networks. arXiv:2206.09455.  
   *Supports: retrieval quality depends on the interaction between the storage rule and the threshold - a sharp recall-failure regime emerges as synapses weaken.*

Foundational background 

- Hebb, D.O. (1949). *The Organization of Behavior.* Wiley.
- Hopfield, J.J. (1982). *PNAS* 79(8), 2554–2558.
- Amit, D.J., Gutfreund, H., & Sompolinsky, H. (1985). *PRL* 55(14), 1530–1533.
- Ba, J. et al. (2016). Using Fast Weights to Attend to the Recent Past. *NeurIPS.*
- Schlag, S., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. *ICML.*

---

## Credits & Licenses

| Asset | Source | License |
|-------|--------|---------|
| Code | Original implementation | MIT |
| Red Hat Display font | Google Fonts | OFL |
| Icons / UI | Custom CSS | MIT |
| Illustrations | Canvas-drawn, original | MIT |
| Data | Generated in-browser, no external datasets | N/A |

---

## AI Assistance Disclosure

This project was developed with AI assistance for:
- Code structure and debugging
- UI/UX design suggestions
- Documentation writing

All generated code was reviewed, tested, and refined by the author. The core Hebbian network implementation and mathematical reasoning are the author's own. AI was used as a collaborative tool, not a substitute for understanding.
---

## License

MIT License — see `LICENSE`.

---

*Built for the DataForge 2026 Pathway Track. Inspired by Hebb (1949), Hopfield (1982), and the Dragon Hatchling architecture (Pathway, 2025).*
