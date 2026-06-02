# makemore — My Journey

Karpathy's makemore series. One problem throughout: a character-level model that learns from `names.txt` (32K names) and generates new names. Each part solves the *same problem* with more powerful machinery.

## Part 1 — Bigram (`makemore.ipynb`)
- Counting: 27×27 matrix of bigram counts → normalize rows into probabilities → sample names. `+1` smoothing avoids `log(0)`.
- Neural net: one-hot → single linear layer → softmax → NLL loss, trained with gradient descent.
- **Insight:** counting and the trained net reach the *same* probabilities — counting is the closed-form answer, gradient descent the iterative one. Smoothing ≈ regularization.

## Part 2 — MLP (`makemore2.ipynb`)
- `block_size=3`: predict next char from previous 3. Learned **embeddings** instead of one-hot, concatenate → `tanh` hidden layer → output.
- Added train/dev/test split (80/10/10), minibatching, and an LR finder (elbow at `lr≈0.1`).
- **Insight:** embeddings let similar characters sit near each other, so unseen context borrows from seen context. That's how it generalizes where the bigram can't.

## Part 3 — Activations & BatchNorm (`makemore3.ipynb`)
- Kaiming-style init so the initial loss isn't huge and `tanh` neurons don't start saturated (saturated tanh = dead gradients).
- Hand-rolled BatchNorm: normalize pre-activations, learned gain/bias, running stats for inference.
- **Insight:** deep nets are hard because of *signal propagation*, not expressiveness. Init and normalization keep activations and gradients in a healthy range.

## Part 4 — Backprop by hand (`makemore4.ipynb`)
- No `loss.backward()` — backpropped through the whole net by hand, checking each gradient against autograd.
- Hardest part was BatchNorm (one example's gradient depends on the whole batch).
- **Insight:** autograd is just the chain rule, mechanically applied. Softmax + cross-entropy collapses to `probs - y` — why that pairing is everywhere.

## Part 5 — WaveNet (`makemore5.ipynb`)
- Built `Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `FlattenConsecutive`, `Sequential` from scratch.
- `block_size=8`, fused 2 chars at a time across stacked layers (tree-like) instead of one wide layer. Found and fixed a BatchNorm dimension bug.

| change | params | train | val |
|---|---|---|---|
| original (3-char, 200 hidden) | 12K | 2.058 | 2.105 |
| context 3 → 8 | 22K | 1.918 | 2.027 |
| flat → hierarchical | 22K | 1.941 | 2.029 |
| fix batchnorm bug | 22K | 1.912 | 2.022 |
| scale up (n_embd 24, n_hidden 128) | 76K | **1.769** | **1.993** |

---

## Takeaways
- Bigram, MLP, and WaveNet are the same model — separated only by context length and machinery. Val loss fell 2.105 -> 1.993.
- Most of the difficulty is plumbing (init, normalization, LR), not theory.
- Doing backprop once by hand means nothing downstream is a black box.
- Validation loss is the only scoreboard — never train.
