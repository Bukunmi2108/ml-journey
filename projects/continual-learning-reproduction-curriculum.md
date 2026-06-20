 # Reproduction-First Curriculum

### Understanding the existing techniques, architectures, and methods — on the way to an on-device continual-learning assistant

*The principle: you reproduce the field's canonical methods at small scale before you try to extend them. Read the paper → rebuild the core from scratch → verify against a known result → write down what you learned. This is the Karpathy "build it to understand it" method, applied to the three pillars you actually need: the architecture, how models are adapted and shrunk, and continual learning itself.*

---

## How to use this document

**Calibration.** This assumes what you already have — micrograd, the full makemore series, the Ng ML specialization — and starts at the transformer and climbs. Where an item overlaps your `ml-journey` plan (transformer, LoRA), do it once and check the box in both places.

**Hardware baseline.**
- A 16GB laptop runs everything here for models ≤ ~1.5B on **CPU** (slow but fine for the small reproductions).
- For training steps, use **Google Colab free T4** (~15GB VRAM) or **Kaggle** (~30 GPU-hrs/week free). Rent an A100 for an afternoon on RunPod/Vast only if a specific step needs it (almost none do).

**Core stack.** `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes` (quantization), `transformer_lens` (interpretability), `wandb` (track every run). Use `uv` or `pip`.

**Default small models** (pick the smallest that shows the effect — small = fast loop = more experiments):
- From-scratch work: your own char-level model / nanoGPT scale.
- Pretrained reproductions: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `HuggingFaceTB/SmolLM2-135M/360M/1.7B-Instruct`, `google/gemma-3-1b`.
- Classic continual-learning testbeds: **Split-MNIST**, **Permuted-MNIST**, **Split-CIFAR-10** (these are where EWC/GEM/generative-replay were originally evaluated — start here, *then* move to the LLM version).

**The loop for every practical.** (1) Read the paper — skim for the problem, then go deep only on the method section. (2) Reproduce the minimal core *from scratch* (no high-level wrapper for the part you're learning). (3) Verify against a known number or a library implementation. (4) Commit a folder with a README stating the result. (5) Do one probe/extension and write one paragraph: *what surprised me*.

**Deliverable convention.** Each practical = a folder/repo with: runnable notebook or script, a README (problem → method → result → what I learned), and the **metric** that proves it worked. That trail is also your portfolio.

---

# Pillar 0 — Inputs

## P0.1 — BPE tokenizer from scratch
**Goal.** Understand exactly how text becomes the integer tokens every model consumes; demystify merges, vocab, and why tokenization causes half of all "weird model behavior."
**Read.** Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* — https://arxiv.org/abs/1508.07909 · Karpathy, **minBPE** (code + video) — https://github.com/karpathy/minbpe
**Specs.** Pure Python + a text corpus (a few MB; e.g., TinyShakespeare or any `.txt`). No GPU. Effort: ~1 day.
**How.** (1) Implement byte-level BPE: count adjacent pairs, merge the most frequent, repeat to a target vocab size. (2) Implement `encode`/`decode`. (3) Train on your corpus; inspect the learned merges. (4) Compare your token counts to `tiktoken`/the HF tokenizer for the same text.
**Deliverable & success.** Your tokenizer round-trips text losslessly and produces a sensible vocab; you can explain why "  hello" and "hello" tokenize differently.

---

# Pillar 1 — Architecture (understand the model itself)

## P1.1 — Transformer / GPT from scratch
**Goal.** Own the base architecture: embeddings, scaled dot-product attention, multi-head, residual + LayerNorm, the block, the training loop, sampling. (This is also `ml-journey` Week 7.)
**Read.** Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762 · Karpathy, *Let's build GPT* (Zero-to-Hero) — https://karpathy.ai/zero-to-hero.html · reference impl: **nanoGPT** — https://github.com/karpathy/nanoGPT
**Specs.** Char-level model, TinyShakespeare (~1MB). ~10M params trains on a free T4 in minutes, on CPU in ~an hour. Effort: 2–3 days (build + rebuild from memory).
**How.** (1) Code along with Karpathy to a working GPT that generates text. (2) **Close the video, rebuild from a blank file** — this is where it becomes yours; the parts you get stuck on are your real gaps. (3) Train, sample, and read the loss curve. (4) Probe: ablate positional embeddings or a residual connection and watch it break.
**Deliverable & success.** A from-memory GPT that generates coherent character-level text; a README explaining every component.

## P1.2 — Positional encodings: learned vs RoPE
**Goal.** Understand how transformers get a sense of order, and why this choice drives long-context behavior.
**Read.** Su et al., *RoFormer (RoPE)* — https://arxiv.org/abs/2104.09864
**Specs.** Reuse your P1.1 model. No new data. Effort: ~1 day.
**How.** (1) Swap your learned positional embedding for sinusoidal, then for RoPE. (2) Train each briefly on the same data. (3) Test **length generalization**: train on sequences of length N, evaluate on >N, and compare. 
**Deliverable & success.** A chart of accuracy/loss vs. test sequence length for each scheme; a paragraph on why RoPE extrapolates better.

## P1.3 — State-space / Mamba block *(optional frontier)*
**Goal.** Understand the leading post-transformer direction: sub-quadratic sequence modeling with a selective recurrent state, and where it wins/loses vs. attention.
**Read.** Gu & Dao, *Mamba* — https://arxiv.org/abs/2312.00752
**Specs.** Tiny model on an algorithmic / long-range task (e.g., copying, selective recall, or a long-context char task). T4. Effort: 3–4 days (this one is genuinely harder).
**How.** (1) Implement a minimal selective-SSM block (you can lean on the official kernels conceptually but write the recurrence yourself for understanding). (2) Drop it into your P1.1 harness. (3) Benchmark against an attention block on a long-range task: quality, memory, speed. (4) Probe a **state-tracking** task to feel the expressivity tradeoff.
**Deliverable & success.** A head-to-head table (quality/memory/throughput) and a written take on the tradeoff.

## P1.4 — Mixture-of-Experts layer *(optional frontier)*
**Goal.** Understand conditional computation — how you add parameters (capacity) without adding per-token compute. Directly relevant to your "capability-per-active-parameter" mission.
**Read.** Fedus et al., *Switch Transformer* — https://arxiv.org/abs/2101.03961
**Specs.** Replace one FFN in your P1.1 model with an MoE FFN (top-1 or top-2 routing, 4–8 experts). T4. Effort: 2 days.
**How.** (1) Implement a router (linear → softmax → top-k) and the expert FFNs. (2) Add the load-balancing auxiliary loss. (3) Train; log which experts fire for which inputs. (4) Compare active-params vs. total-params and quality vs. a dense baseline of equal *active* size.
**Deliverable & success.** A working MoE layer with balanced routing; a note on capability gained per active parameter.

---

# Pillar 2 — Adaptation & efficiency (how models get specialized and shrunk)

## P2.1 — LoRA from scratch
**Goal.** Own low-rank adaptation by hand before you ever call the library — the substrate of the entire continual-learning project.
**Read.** Hu et al., *LoRA* — https://arxiv.org/abs/2106.09685 · reference: HF **peft** — https://github.com/huggingface/peft
**Specs.** Small pretrained model (Qwen2.5-0.5B) + a tiny task dataset. T4. Effort: 1–2 days.
**How.** (1) Implement a `LoRALinear` wrapper: frozen `W` + trainable `B·A` (rank r), scaled by `alpha/r`. (2) Inject it into the attention projections (`q_proj`, `v_proj`). (3) Fine-tune on a small task; verify only the adapters update. (4) Verify your hand-rolled version matches `peft`'s on the same setup. (5) Sweep rank r ∈ {2, 8, 64}.
**Deliverable & success.** Your LoRA matches `peft` behavior; you can state what r and alpha actually control.

## P2.2 — Quantization from scratch (int8 → 4-bit)
**Goal.** Understand the quality/size/speed tradeoff that makes local deployment possible.
**Read.** Dettmers et al., *LLM.int8()* — https://arxiv.org/abs/2208.07339 · Frantar et al., *GPTQ* — https://arxiv.org/abs/2210.17323 · reference: **bitsandbytes** — https://github.com/bitsandbytes-foundation/bitsandbytes
**Specs.** A small pretrained model's weights. CPU is fine. Effort: 2 days.
**How.** (1) Implement per-tensor then per-channel **absmax int8** quantization of a linear layer; dequantize and measure error. (2) Extend to **4-bit with group-wise scales** (group size 128). (3) Measure perplexity/accuracy degradation vs. memory saved at fp16/int8/int4. (4) Note where naive quantization breaks (outlier channels — the LLM.int8 insight).
**Deliverable & success.** A curve of quality vs. bits; an explanation of why outlier-aware schemes exist.

## P2.3 — Ternary / 1-bit (BitNet) *(frontier)*
**Goal.** Understand the most aggressive efficiency frontier — training in ~1.58 bits from scratch, near-multiplication-free inference. This is the literal "runs in your pocket" endpoint.
**Read.** *The Era of 1-bit LLMs: BitNet b1.58* — https://arxiv.org/abs/2402.17764 · reference: **bitnet.cpp** — https://github.com/microsoft/BitNet
**Specs.** A tiny model trained from scratch with a ternary `BitLinear` (weights ∈ {-1,0,1}). T4. Effort: 2–3 days.
**How.** (1) Implement `BitLinear`: quantize weights to ternary in the forward pass, straight-through estimator for the backward. (2) Train a small char model with it. (3) Compare quality and (theoretical) compute vs. an fp32 twin. (4) Optionally run an official BitNet b1.58 model via `bitnet.cpp` on your CPU to feel the speed.
**Deliverable & success.** A trained ternary model that learns; a note on the quality cost vs. the efficiency prize.

## P2.4 — QLoRA
**Goal.** Combine quantization + LoRA — fine-tune a quantized base on consumer hardware. The practical workhorse.
**Read.** Dettmers et al., *QLoRA* — https://arxiv.org/abs/2305.14314
**Specs.** Load a 1.5–3B model in 4-bit (NF4), attach LoRA, fine-tune on a small instruction dataset. T4. Effort: 1 day (you'll reuse P2.1–2.2 understanding).
**How.** (1) Load in 4-bit via bitsandbytes (NF4 + double quant). (2) Attach LoRA via `peft`. (3) Fine-tune; confirm it fits in T4 memory. (4) Compare output quality and VRAM to full fp16 LoRA.
**Deliverable & success.** A 4-bit model fine-tuned on a free GPU; a VRAM/quality comparison table.

## P2.5 — Knowledge distillation
**Goal.** Understand how a small student inherits a large teacher's capability — a core lever for capability-per-parameter.
**Read.** Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531
**Specs.** Teacher = a mid-size model; student = a much smaller one; any classification or next-token task. T4. Effort: 2 days.
**How.** (1) Generate teacher soft labels (logits) at temperature T. (2) Train the student on a mix of the true labels and a KL term to the teacher's softened distribution. (3) Compare student-with-distillation vs. student-alone vs. teacher. (4) Sweep temperature and the loss-mixing weight.
**Deliverable & success.** The distilled student beats the same student trained without the teacher; you can explain why soft targets carry more information than hard labels.

---

# Pillar 3 — Continual learning (the heart of the goal)

> **Why this pillar matters:** every method here is a tool you'll combine in the final assistant. Reproduce each on the **classic small testbeds first** (Split/Permuted-MNIST), because they're tiny, fast, and the metrics are standardized — *then* port the idea to the LLM-with-LoRA setting. Use [**Avalanche**](https://github.com/ContinualAI/avalanche) as a reference implementation to check your numbers against. Survey + paper hub: [LLM continual-learning survey list](https://github.com/Wang-ML-Lab/llm-continual-learning-survey); a general CL survey — De Lange et al., https://arxiv.org/abs/1909.08383.

**The metric for this whole pillar** (build it once, reuse everywhere): after learning a sequence of tasks, report **Average Accuracy**, **Forgetting** (drop on earlier tasks), and **Forward Transfer**. Two curves: *retained old* vs *gained new*. This is the measurement spine.

## P3.1 — Reproduce catastrophic forgetting (the phenomenon)
**Goal.** See the problem cleanly and quantitatively before fixing anything.
**Read.** French, *Catastrophic forgetting in connectionist networks* (Trends Cogn. Sci., 1999 — classic, find via scholar) · context from the De Lange survey above.
**Specs.** Small MLP/CNN on **Split-MNIST** (5 sequential 2-class tasks). CPU is fine. Effort: 1 day.
**How.** (1) Train on Task 1, record accuracy. (2) Train sequentially on Tasks 2–5. (3) Re-test Task 1 after each — watch it collapse. (4) Plot the forgetting curve.
**Deliverable & success.** A plot showing old-task accuracy cratering as new tasks arrive — the baseline every later method must beat.

## P3.2 — Experience replay / rehearsal
**Goal.** The simplest and strongest mitigation: keep and replay a sample of old data.
**Read.** Lopez-Paz & Ranzato, *Gradient Episodic Memory (GEM)* — https://arxiv.org/abs/1706.08840
**Specs.** Same Split-MNIST harness + a small memory buffer (e.g., 100–500 samples/task). Effort: 1 day.
**How.** (1) Add a ring buffer of past examples. (2) On each new-task batch, mix in a replay batch. (3) Re-run the forgetting metric. (4) Sweep buffer size; plot forgetting vs. memory budget.
**Deliverable & success.** Forgetting drops sharply vs. P3.1; a forgetting-vs-buffer-size curve.

## P3.3 — Generative replay
**Goal.** Rehearse without storing data — a generator produces pseudo-old-samples (brain-like; no privacy cost). Directly relevant to an on-device assistant that can't hoard your data.
**Read.** Shin et al., *Deep Generative Replay* — https://arxiv.org/abs/1705.08690
**Specs.** Split-MNIST + a small generator (VAE/GAN, or for text, sample from the model itself). Effort: 2 days.
**How.** (1) Train a generator alongside the classifier. (2) Before each new task, sample "old" data from the generator and mix it in. (3) Compare to real replay (P3.2). (4) Watch for **drift** — quality decay when training on self-generated data.
**Deliverable & success.** Generative replay approaches real-replay performance; you can describe the drift failure mode.

## P3.4 — Elastic Weight Consolidation (EWC)
**Goal.** Mitigate forgetting *without storing data* — protect the weights that mattered for old tasks.
**Read.** Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks (EWC)* — https://arxiv.org/abs/1612.00796
**Specs.** Split/Permuted-MNIST harness. Effort: 1–2 days.
**How.** (1) After Task 1, estimate the **Fisher information** per parameter (importance). (2) Add a quadratic penalty anchoring important weights when training Task 2+. (3) Tune the penalty strength λ. (4) Compare to replay and to no-mitigation.
**Deliverable & success.** EWC reduces forgetting vs. P3.1; a λ sweep showing the stability↔plasticity tradeoff explicitly.

## P3.5 — Parameter isolation (progressive / adapter-per-task)
**Goal.** Make forgetting structurally impossible by giving each task its own parameters. This is the seed of your "society of specialists."
**Read.** Rusu et al., *Progressive Neural Networks* — https://arxiv.org/abs/1606.04671
**Specs.** Two settings — (a) classic: progressive columns on Split-MNIST; (b) LLM: one **LoRA adapter per task** on Qwen2.5-0.5B with a simple task router. T4. Effort: 2–3 days.
**How.** (1) Train a separate adapter/column per task; freeze previous ones. (2) Route inputs to the right adapter (oracle first, then a learned router). (3) Measure: zero forgetting by construction — at the cost of growth and routing. (4) Note the open problem: routing when the task is unknown.
**Deliverable & success.** Zero/near-zero forgetting; a written analysis of the growth/routing cost — and why this motivates the consolidation approach.

## P3.6 — Knowledge editing (ROME)
**Goal.** Understand targeted *surgery* on a single fact — the precise opposite of broad fine-tuning, and a key alternative for "update one thing without disturbing the rest."
**Read.** Meng et al., *Locating and Editing Factual Associations in GPT (ROME)* — https://arxiv.org/abs/2202.05262 · scaling version *MEMIT* — https://arxiv.org/abs/2210.07229 · code + Colab — https://rome.baulab.info · toolkit **EasyEdit** — https://github.com/zjunlp/EasyEdit
**Specs.** GPT-2 XL or GPT-J via the ROME repo (use their Colab if local is tight). Effort: 2 days (mostly reading + running).
**How.** (1) Run causal tracing to locate where a fact lives. (2) Apply a rank-one edit to insert a counterfactual ("Eiffel Tower is in Rome"). (3) Test **specificity** (unrelated facts unchanged) and **generalization** (paraphrases). (4) Compare to fine-tuning the same fact and observe the collateral damage.
**Deliverable & success.** A successful, localized edit that generalizes without breaking neighbors; a note on when editing beats fine-tuning.

## P3.7 — Memory-augmented / test-time learning (Titans-lite)
**Goal.** Understand learning *into a memory module* at inference instead of retraining weights — the current frontier and the most direct route to "learns from use."
**Read.** Behrouz et al., *Titans: Learning to Memorize at Test Time* — https://arxiv.org/abs/2501.00663 · TTT — Sun et al., https://arxiv.org/abs/1909.13231 and TTT-layers, https://arxiv.org/abs/2407.04620 · frontier context — *Nested Learning* — https://arxiv.org/abs/2512.24695
**Specs.** Small model + a learnable memory module updated at test time via a "surprise" gradient. T4. Effort: 3–4 days (hardest in the pillar).
**How.** (1) Implement a small neural memory updated by gradient on a surprise signal during the forward pass. (2) Attach it to your P1.1 model (memory-as-context is the simplest variant). (3) Test recall of information seen only at test time. (4) Compare to a frozen baseline and to retrieval.
**Deliverable & success.** A model that demonstrably stores and recalls test-time information without a training run; a written comparison to RAG.

---

# Pillar 4 — Internals *(optional; fits your glass-box leaning)*

## P4.1 — Find induction heads with TransformerLens
**Goal.** Read a real circuit inside a trained transformer — the skill behind interpretability and "auditable" models.
**Read.** Olsson et al., *In-context Learning and Induction Heads* — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html · tool: **TransformerLens** — https://github.com/TransformerLensOrg/TransformerLens · curriculum: **ARENA** mech-interp — https://www.arena.education
**Specs.** `gpt2-small` via TransformerLens. CPU fine. Effort: 2 days.
**How.** (1) Load the model, cache activations. (2) Build the induction task ("…A B … A → B"). (3) Locate the attention heads implementing the copy pattern. (4) Visualize attention + ablate the head to confirm causality.
**Deliverable & success.** Identified, visualized, and causally-confirmed induction heads; a paragraph on what a "circuit" is.

---

# Synthesis — the project

Only after Pillar 3 do you build the thing this was all for: the **on-device continual-learning assistant** — a local small model + episodic memory + a consolidation loop (replay/EWC/generative replay, chosen because you now understand the tradeoffs) + adapter routing + the dual-curve measurement harness, optionally quantized for true pocket deployment. At that point you won't be guessing; you'll be *composing methods you've reproduced*. (Full project breakdown is in the earlier project ladder — rungs 10–16.)

---

# Suggested sequencing

- **Run alongside `ml-journey`, don't replace it.** Your plan builds breadth; this builds the depth the project needs.
- **Order:** P0 → P1.1 → P2.1 → then start Pillar 3 early (P3.1–3.2) because forgetting is the heart and it's motivating. Interleave P1.2, P2.2–2.5 as you go. Treat P1.3, P1.4, P2.3, P3.7 as the "frontier" set — do them when you have momentum.
- **Pace honestly.** ~1–2 days per practical, frontier items 3–4. Don't move on until the deliverable's metric exists — "if you can't show it, you didn't do it" (your own README's rule).
- **Track everything in `wandb`** and commit daily; the reproductions *are* your portfolio.

---

# Global resources

- **Karpathy, Neural Networks: Zero to Hero** — https://karpathy.ai/zero-to-hero.html (transformer, tokenizer)
- **HF `peft`** (LoRA/QLoRA) — https://github.com/huggingface/peft
- **bitsandbytes** (quantization) — https://github.com/bitsandbytes-foundation/bitsandbytes
- **Avalanche** (continual-learning reference lib) — https://github.com/ContinualAI/avalanche
- **EasyEdit** (knowledge editing) — https://github.com/zjunlp/EasyEdit
- **TransformerLens** (interpretability) — https://github.com/TransformerLensOrg/TransformerLens
- **mergekit** (model merging as a CL technique) — https://github.com/arcee-ai/mergekit
- **LLM continual-learning survey + paper list** — https://github.com/Wang-ML-Lab/llm-continual-learning-survey

---

*Reproduce, verify, write down what surprised you. Do that sixteen times and you won't be "learning to do research" anymore — you'll be doing it, on a problem you understand from the weights up.*
