# Inside the LLM Word Factory

Code and data for **"Inside the LLM Word Factory"** (Busigin & Pinter), a mechanistic study of *detokenization* - how transformer language models reconstruct a single word-level representation from the subword fragments a tokenizer produces.

The experiments use **activation patching** on controlled pairs of artificially split words to separate the contribution of attention from that of the MLP, and a **linear probe** to predict whether detokenization will succeed from early-layer activations alone. 
Everything runs on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) `HookedTransformer` models.

## [image goes here]

## What the code does


| Paper section     | What it measures                                                          | Entry point                                                                           |
| ----------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| §2.2 / Appendix A | Build LST/FST contrastive pair datasets                                   | `create_datasets`                                                                     |
| §3.1–3.2          | Layer-wise patching of `attn_out` / `mlp_out` / `resid_post` (gap closed) | `activation_patching.run_patching`                                                    |
| §3.3              | Per-head (`q`/`k`/`v`/`z`) and attention-pattern patching                 | `activation_patching.run_patching(..., heads=...)`                                    |
| §3.4              | MLP necessity (zero-ablation) and continuity (α-scaling)                  | `interventions.scale_component` (necessity: `alphas=[0.0]`; continuity: `layers=[1]`) |
| §4 / Appendix C   | Token-count scaling and intermediate-position relays                      | `activation_patching.run_patching` (multi-position)                                   |
| §5 / Appendix E   | Cross-architecture two-stage localization                                 | `activation_patching.run_patching`                                                    |
| §6 / Appendix G   | Early-layer linear probe (isolated, in-context, transfer)                 | `probing.run_probe`                                                                   |


### Module map

```
token_utils.py          Tokenizer-agnostic segmentation (SentencePiece "▁" and GPT-2 "Ġ"),
                        BOS handling, vocab-consistent k-piece splits.
metrics.py              Canonicity (row/mean cosine), next-token behavioral metrics
                        (KL, top-1, top-5), and the gap-closed-% helpers.
activations.py          Single-pass activation/logit extraction at a (layer, position).
create_datasets.py      Pipeline: enumerate splits → score canonicity → build high/low pairs.
data_utils.py           Resolve a model to its on-disk dataset tag and load pair JSONs
                        into model-ready tensors.
activation_patching.py  All patching variants (layer sweep, per-head, multi-position) +
                        baselines, behind a single run_patching entry point.
interventions.py        Single-run interventions: one scale_component sweep covers
                        both MLP necessity (α=0 ablation) and continuity (α-scaling).
probing.py              Class-mean-difference probe end to end, incl. WikiText-103 mining.
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The heavyweight dependency is `transformer_lens`; it determines the compatible `torch` build. 

A GPU is strongly recommended for the 6–7B models, but the controlled experiments fit comfortably on a single 24 GB card in `bfloat16`.

---

## Word lists

The single-token word pool is built from three public English word lists, merged, lowercased, deduplicated, and filtered to the words the model's tokenizer encodes as exactly one token (Appendix A).  
The raw lists ship under `words/`; the experiment entry points take the already-filtered single-token list as input. 

Sources:

- `top-english-wordlists` (100k most common lowercased words)
- Google 10,000-word common English list
- Tatman (2017), *English Word Frequency* (Kaggle)

---

## Models

The two-stage mechanism replicates across all 12 models; the **depth** at which it runs is governed by positional encoding (RoPE/ALiBi concentrate it in 1–5 layers; learned-absolute spreads it over 5–10).


| Regime                | Models                                                |
| --------------------- | ----------------------------------------------------- |
| Concentrated (RoPE)   | GPT-J-6B, Pythia-410M/1B/6.9B, Llama-2-7B, Gemma-2-2B |
| Intermediate (ALiBi)  | BLOOM-7B1                                             |
| Distributed (learned) | OPT-1.3B/6.7B, GPT-Neo-1.3B, GPT-2 Large/XL           |


Per-model dataset thresholds, pool sizes, and probe depths `l` are in the paper's Tables 1, 3, and 4.  
The probe depths are mirrored in `probing.PAPER_PROBE_LAYERS`.

---

## Citation

```bibtex
@misc{busigin2026wordfactory,
  title         = {Inside the LLM Word Factory},
  author        = {Busigin, Benzi and Pinter, Yuval},
  year          = {2026},
  eprint        = {2606.08562},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

Feel free to contact if you have any thoughts, questions or suggestions.