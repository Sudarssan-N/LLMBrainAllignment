"""Builds notebooks/run_experiments.ipynb (the full research notebook). Run once:
    python scripts/_build_notebook.py
Kept in-repo so the canonical notebook is regenerable rather than hand-edited.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# LLM–Brain Alignment — Surprisal vs Structure (Pereira2018)

**Disentangling statistical surprisal from hierarchical structure in LLM–brain alignment.**

This notebook runs the entire study end-to-end in Colab and renders every figure used in the
paper. It is the single control surface for the project — clone, install, run, plot.

**Research question.** *How much of LLM–brain alignment on Pereira2018 is uniquely explained by
hidden representations beyond next-word surprisal, and how does that residual depend on model
size, layer depth, and instruction tuning?*

**Pipeline.**
```
01 activations → 02 surprisal → 03 encoding(+noise ceiling) → 04 variance partitioning
                                   → 05 RSA → 06 aggregate → 07 compare → 08 figures
```

| Experiment | What | Cell section |
|---|---|---|
| E001b | GPT-2 layer-depth predictivity (reproduction) | §4–5 |
| E002 | Variance partitioning (core result) + confound control | §6 |
| E003 | Size scaling — clean Pythia 70m→1.4b curve | §7–9 |
| E004 | Base vs instruction-tuned (two Qwen pairs, paired test) | §9 |
| E005 | RSA geometry | §10 |
| E006 | Robustness: 243-sentence experiment | §11 |

**Reviewer-driven robustness controls now in the pipeline:**
- **Confound control (Hadidi et al. 2026):** the controlled nuisance space includes word
  rate, word length, word frequency and **sentence position**, plus predictive **entropy** —
  so *unique-hidden* is variance beyond surprisal AND those low-level confounds. §6 plots
  the confound-controlled vs surprisal-only curves so you can see whether it survives.
- **Clean scaling axis:** Pythia 70m/160m/410m/1b/1.4b share training data + step order.
- **Per-subject aggregation (§6b):** guards the result against a voxel-pooling artifact.
""")

md("## 1. Setup — clone & install")
code("""import os
REPO = 'LLMBrainAllignment'
if not os.path.exists(REPO):
    !git clone https://github.com/Sudarssan-N/LLMBrainAllignment.git
%cd {REPO}
!git pull
!pip -q install -e .[models,plot,confounds]""")

md("""### 1a. Brain-score harness (real Pereira2018)
Heavy install (a few minutes). brain-score pins `numpy<2`, which clashes with Colab's
numpy-2 C-extensions, so **this cell auto-restarts the runtime once after installing.**
That is expected — when it restarts, just run *Runtime → Run all* (or re-run from §1; installs
are cached and fast). It will not restart a second time.""")
code("""import importlib.util, os
if importlib.util.find_spec('brainscore_language') is None:
    !pip -q install git+https://github.com/brain-score/language
    !pip -q install "numpy<2"   # keep numpy consistent with brain-score's pin
    print('Installed brain-score. RESTARTING runtime — then Run all again.')
    os.kill(os.getpid(), 9)     # force Colab runtime restart for a clean numpy
else:
    print('brainscore_language present; environment consistent, no restart needed')""")

md("## 2. Smoke test (synthetic, no models, no real data)\\nConfirms the whole chain works before pulling weights.")
code("""!python -m pytest tests/ -q
for stage in ['01_extract_activations','02_compute_surprisal','03_fit_encoding',
              '04_variance_partitioning','05_rsa']:
    !python scripts/{stage}.py --synthetic""")

md("""## 3. Configuration & helpers
The model roster and analysis knobs live in `config/default.yaml`. Helpers below load the JSON
artifacts each stage writes under `data/derived/<model>/`.""")
code("""import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DERIVED = Path('data/derived')
PARAMS_M = {'gpt2':124,'gpt2-medium':355,'opt-125m':125,
            'pythia-70m':70,'pythia-160m':160,'pythia-410m':410,'pythia-1b':1000,
            'pythia-1.4b':1400,
            'qwen2.5-0.5b':500,'qwen2.5-0.5b-instruct':500,
            'qwen2.5-1.5b':1500,'qwen2.5-1.5b-instruct':1500}

def load(model, name):
    p = DERIVED/model/f'{name}.json'
    return json.load(open(p)) if p.exists() else None

def models_present():
    return sorted(p.name for p in DERIVED.iterdir() if p.is_dir()) if DERIVED.exists() else []

def rel_depth(layers):
    layers = np.asarray(sorted(int(l) for l in layers), float)
    return layers/layers.max()

print('config models:')
!python -c "import yaml; print(list(yaml.safe_load(open('config/default.yaml'))['models']))" """)

md("""## 4. Single-model run (tutorial: GPT-2)
Runs all five stages for one model. `--pooling last` is the causal-LM standard (final token has
attended to the whole sentence). The cross-subject noise ceiling is computed once and cached.""")
code("""MODEL = 'gpt2'
!python scripts/01_extract_activations.py --model {MODEL} --pooling last
!python scripts/02_compute_surprisal.py    --model {MODEL}
!python scripts/03_fit_encoding.py         --model {MODEL}
!python scripts/04_variance_partitioning.py --model {MODEL}
!python scripts/05_rsa.py                   --model {MODEL}""")

md("""## 5. E001b — Layer-depth predictivity
Normalized predictivity (r / noise ceiling) vs layer. Reproduces the Schrimpf-style curve;
GPT-2 reaches ~100% of the ceiling at its final layer.""")
code("""MODEL = 'gpt2'
enc = load(MODEL,'encoding'); pl = enc['per_layer']
layers = sorted(int(l) for l in pl)
key = 'normalized_r' if enc.get('has_noise_ceiling') else 'mean_r'
y = [pl[str(l)].get(key) or pl[str(l)]['mean_r'] for l in layers]
plt.figure(figsize=(6,4))
plt.plot(layers, y, 'o-')
plt.axhline(1.0, ls='--', c='gray', lw=.8)
plt.xlabel('layer'); plt.ylabel(key); plt.title(f'{MODEL}: layer-depth predictivity')
plt.show()
print('best layer', enc['best_layer'], '| best normalized_r', enc.get('best_normalized_r'))""")

md("""## 6. E002 — Variance partitioning (the core result)
Decomposes voxel variance into **unique-hidden**, **shared**, and **unique-nuisance**. The
unique-hidden band dwarfs the nuisance contribution and grows with depth — alignment is
carried by representations *beyond* surprisal. The **controlled nuisance space** now bundles
surprisal + entropy + word-rate/length/frequency + sentence position, so unique-hidden is the
variance hidden states add beyond all of those.""")
code("""MODEL = 'gpt2'
vj = load(MODEL,'varpart'); v = vj['per_layer']
layers = sorted(int(l) for l in v)
uh=[v[str(l)]['mean_unique_hidden'] for l in layers]
sh=[v[str(l)]['mean_shared'] for l in layers]
us=[v[str(l)]['mean_unique_surprisal'] for l in layers]
plt.figure(figsize=(7,4))
plt.stackplot(layers, uh, sh, us, labels=['unique hidden','shared','unique nuisance'],
              colors=['#2c7fb8','#7fcdbb','#edf8b1'])
plt.legend(loc='upper left'); plt.xlabel('layer'); plt.ylabel('R$^2$')
ttl = 'variance partitioning' + (' (confound-controlled)' if vj.get('confound_control') else '')
plt.title(f'{MODEL}: {ttl}'); plt.show()
peak = vj.get('peak_layer')
d = v[str(peak)]
ci = d.get('ci_unique_hidden')
print('confound control:', vj.get('confound_control'), '| nuisance =',
      ['surprisal/entropy'] + vj.get('confound_names', []))
print(f"peak layer {peak}: unique_hidden={d['mean_unique_hidden']:.4f}",
      f"[{ci['lo']:.4f},{ci['hi']:.4f}]" if ci else '',
      f"| unique_nuisance={d['mean_unique_surprisal']:.4f}",
      f"| ratio ~{d['mean_unique_hidden']/max(d['mean_unique_surprisal'],1e-9):.0f}x")""")

md("""### 6a. Does unique-hidden survive the confound control? (Hadidi et al. 2026)
Overlays unique-hidden when controlling for **surprisal only** vs **surprisal + entropy +
position + word-rate** confounds. If the two curves are close, the result is not explained by
the low-level positional/word-rate signals that Hadidi et al. show dominate Pereira2018.""")
code("""MODEL = 'gpt2'
vj = load(MODEL,'varpart')
v = vj['per_layer']; so = vj.get('per_layer_surprisal_only') or {}
layers = sorted(int(l) for l in v)
uh_ctrl = [v[str(l)]['mean_unique_hidden'] for l in layers]
plt.figure(figsize=(7,4))
plt.plot(layers, uh_ctrl, 'o-', c='#2c7fb8', label='controlled (surprisal+confounds)')
if so:
    uh_so = [so[str(l)]['mean_unique_hidden'] for l in layers]
    plt.plot(layers, uh_so, 's--', c='#de2d26', label='surprisal only')
plt.xlabel('layer'); plt.ylabel('unique-hidden R$^2$')
plt.title(f'{MODEL}: unique-hidden survival under confound control'); plt.legend(); plt.show()
if so:
    peak = vj['peak_layer']
    drop = so[str(peak)]['mean_unique_hidden'] - v[str(peak)]['mean_unique_hidden']
    print(f"peak layer {peak}: surprisal-only={so[str(peak)]['mean_unique_hidden']:.4f} -> "
          f"controlled={v[str(peak)]['mean_unique_hidden']:.4f}  (confounds absorb {drop:.4f})")""")

md("""### 6b. Per-subject vs pooled (W6 — voxel-pooling control)
`ridge_encode` fits each voxel independently, so pooling never mixes subject signal in the
fit — pooled vs per-subject differs only in aggregation weighting. This compares the pooled
mean to the mean across subjects (each subject weighted equally), with per-subject scatter.""")
code("""MODEL = 'gpt2'
ps = load(MODEL,'varpart').get('per_subject')
if not ps:
    print('No per-subject info (synthetic data has no subject-specific voxels).')
else:
    sm = ps['subject_unique_hidden']
    names = list(sm); vals = [sm[n] for n in names]
    plt.figure(figsize=(6,4))
    plt.scatter(range(len(vals)), vals, c='#2c7fb8', zorder=3, label='per subject')
    plt.axhline(ps['mean_across_subjects'], c='#2c7fb8', ls='-',
                label=f"across-subject mean={ps['mean_across_subjects']:.3f}")
    plt.axhline(ps['pooled_unique_hidden'], c='#de2d26', ls='--',
                label=f"pooled={ps['pooled_unique_hidden']:.3f}")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
    plt.ylabel('peak unique-hidden R$^2$'); plt.title(f'{MODEL}: per-subject vs pooled')
    plt.legend(fontsize=7); plt.tight_layout(); plt.show()
    print(f"n_subjects={ps['n_subjects']} mean={ps['mean_across_subjects']:.4f} "
          f"±{ps['sem_across_subjects']:.4f} (SEM) | pooled={ps['pooled_unique_hidden']:.4f}")""")

md("""## 7. E003/E004 — Full model sweep
Full roster: the **Pythia scaling suite** (70M/160M/410M/1B/1.4B — identical data & step order,
the clean scaling axis), cross-family baselines (GPT-2, OPT-125M), and **two base-vs-instruct
Qwen pairs** (0.5B, 1.5B). Last-token pooling throughout; the cached ceiling makes all but the
first model fast. The 1B/1.4B/1.5B models want a GPU runtime.""")
code("""MODELS = ['gpt2','opt-125m',
          'pythia-70m','pythia-160m','pythia-410m','pythia-1b','pythia-1.4b',
          'qwen2.5-0.5b','qwen2.5-0.5b-instruct',
          'qwen2.5-1.5b','qwen2.5-1.5b-instruct']
for m in MODELS:
    print(f'\\n===== {m} =====')
    !python scripts/01_extract_activations.py --model {m} --pooling last
    !python scripts/02_compute_surprisal.py    --model {m}
    !python scripts/03_fit_encoding.py         --model {m}
    !python scripts/04_variance_partitioning.py --model {m}
    !python scripts/05_rsa.py                   --model {m}""")

md("## 8. Aggregate table\\nOne row per model: best normalized predictivity, peak unique-hidden (with CI), hidden:surprisal ratio, best RSA.")
code("""!python scripts/06_aggregate.py
df = pd.read_csv(DERIVED/'summary.csv')
df""")

md("""## 9. Cross-model figures
**Left:** layer-depth predictivity for every model. **Right:** size axis — peak unique-hidden vs
parameter count. The **Pythia suite is highlighted as a connected line** because it is the only
confound-free scaling axis (same data + step order); cross-family models are shown as faint
points for context, not as a trend.""")
code("""present = models_present()
fig, ax = plt.subplots(1,2, figsize=(13,4.5))
for m in present:
    enc = load(m,'encoding')
    if not enc: continue
    pl = enc['per_layer']; layers = sorted(int(l) for l in pl)
    key = 'normalized_r' if enc.get('has_noise_ceiling') else 'mean_r'
    ax[0].plot(rel_depth(layers), [pl[str(l)].get(key) or pl[str(l)]['mean_r'] for l in layers],
               'o-', ms=3, label=m)
ax[0].axhline(1,ls='--',c='gray',lw=.8); ax[0].set(xlabel='relative depth',
    ylabel='normalized predictivity', title='Layer-depth alignment'); ax[0].legend(fontsize=7)

def peak_uh(m):
    vj = load(m,'varpart')
    if not vj: return None
    peak = vj.get('peak_layer'); pl = vj['per_layer']
    if peak is None: peak = max(pl, key=lambda l: pl[l]['mean_unique_hidden'])
    return pl[str(peak)]['mean_unique_hidden']

# Pythia scaling suite as a connected (clean) line
pythia_order = ['pythia-70m','pythia-160m','pythia-410m','pythia-1b','pythia-1.4b']
px = [PARAMS_M[m] for m in pythia_order if peak_uh(m) is not None]
py = [peak_uh(m) for m in pythia_order if peak_uh(m) is not None]
if px:
    ax[1].plot(px, py, 'o-', c='#2c7fb8', lw=2, ms=6, label='Pythia (clean scaling axis)', zorder=3)
# other families as faint context points
for m in present:
    if m in pythia_order: continue
    y = peak_uh(m)
    if y is None: continue
    ax[1].scatter(PARAMS_M.get(m, np.nan), y, c='gray', alpha=0.6, zorder=2)
    ax[1].annotate(m, (PARAMS_M.get(m, np.nan), y), fontsize=6, xytext=(3,3),
                   textcoords='offset points', color='gray')
ax[1].set(xscale='log', xlabel='params (M, log)', ylabel='peak unique-hidden R$^2$',
          title='Scaling: Pythia suite (line) vs other families (points)')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md("""### 9a. E004 — Base vs instruction-tuned (two pairs, paired bootstrap test)
Two base/instruct pairs at different scales (Qwen2.5-**0.5B** and **1.5B**) test whether the
instruction-tuning effect is real and whether it is scale-dependent — addressing the single-pair
limitation and the tension with Aw et al. (2024, +6.9% on Pereira2018) and Sun et al. (2024, null).""")
code("""PAIRS = [('qwen2.5-0.5b','qwen2.5-0.5b-instruct'),
         ('qwen2.5-1.5b','qwen2.5-1.5b-instruct')]
fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.2*len(PAIRS),4), squeeze=False)
for ax,(base,inst) in zip(axes[0], PAIRS):
    if not (load(base,'varpart') and load(inst,'varpart')):
        ax.set_title(f'{base}\\n(missing — run §7)'); continue
    !python scripts/07_compare.py --a {base} --b {inst}
    cmp_path = DERIVED/f'compare_{base}_vs_{inst}.json'
    if cmp_path.exists(): print(base, '->', json.load(open(cmp_path)))
    vals=[]; errs=[[],[]]
    for m in (base,inst):
        vj=load(m,'varpart'); peak=vj['peak_layer']; d=vj['per_layer'][str(peak)]
        ci=d.get('ci_unique_hidden'); vals.append(d['mean_unique_hidden'])
        errs[0].append(d['mean_unique_hidden']-ci['lo'] if ci else 0)
        errs[1].append(ci['hi']-d['mean_unique_hidden'] if ci else 0)
    ax.bar(['base','instruct'], vals, yerr=errs, capsize=4, color=['#2c7fb8','#de2d26'])
    ax.set_ylabel('peak unique-hidden R$^2$ (95% CI)'); ax.set_title(base.replace('-instruct',''))
plt.tight_layout(); plt.show()""")

md("## 10. E005 — RSA (representational geometry)\\nRDM-to-RDM Spearman correlation between each model layer and the brain, complementing linear predictivity.")
code("""present = models_present()
plt.figure(figsize=(6.5,4))
for m in present:
    r = load(m,'rsa')
    if not r: continue
    pl=r['per_layer']; layers=sorted(int(l) for l in pl)
    plt.plot(rel_depth(layers), [pl[str(l)] for l in layers], 'o-', ms=3, label=m)
plt.axhline(0,ls='--',c='gray',lw=.8)
plt.xlabel('relative depth'); plt.ylabel('RSA (Spearman ρ)'); plt.title('Geometry alignment')
plt.legend(fontsize=7); plt.show()""")

md("""## 11. E006 — Robustness: the 243-sentence experiment
Re-run the analysis on Pereira's other sentence set, written to a separate `data/derived_243/`
via an on-the-fly config. Confirms findings aren't specific to the 384 set.""")
code("""import yaml
cfg = yaml.safe_load(open('config/default.yaml'))
cfg['dataset']['experiments'] = [243]
cfg['paths']['data_derived'] = 'data/derived_243'
yaml.safe_dump(cfg, open('config/exp243.yaml','w'))
for m in ['gpt2','qwen2.5-0.5b']:
    for s in ['01_extract_activations','02_compute_surprisal','03_fit_encoding','04_variance_partitioning']:
        extra = '--pooling last' if s=='01_extract_activations' else ''
        !python scripts/{s}.py --config config/exp243.yaml --model {m} {extra}
!python scripts/06_aggregate.py --config config/exp243.yaml""")

md("## 12. Export paper figures\\nRenders the four figures straight into `paper/figures/` for the LaTeX.")
code("""!python scripts/08_figures.py
from IPython.display import display
import glob
print(glob.glob('paper/figures/*.pdf'))""")

md("""## 13. Build the paper (optional)
Compile the NeurIPS draft. Requires `neurips_2024.sty` (download from neurips.cc and place in
`paper/`). Easiest is to edit/build in Overleaf instead.""")
code("""# !apt-get -qq install texlive-latex-extra texlive-bibtex-extra > /dev/null
# %cd paper && make && %cd ..""")

md("""---
### Notes
- All results use public models + the public Pereira2018 release via `brain-score/language`.
- Push code changes to `origin/main`; re-`git pull` here to pick them up (Colab does **not**
  refresh already-open notebook cells on pull — re-open or re-paste the changed cell).
- Tracker of runs: `TRACKER.md`; running log: `LOG.md`.""")

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "run_experiments.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
