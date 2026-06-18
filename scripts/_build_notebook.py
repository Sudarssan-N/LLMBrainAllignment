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
| E002 | Variance partitioning (core result) | §6 |
| E003 | Size scaling (4 sizes / 4 families) | §7–9 |
| E004 | Base vs instruction-tuned (paired test) | §9 |
| E005 | RSA geometry | §10 |
| E006 | Robustness: 243-sentence experiment | §11 |
""")

md("## 1. Setup — clone & install")
code("""import os
REPO = 'LLMBrainAllignment'
if not os.path.exists(REPO):
    !git clone https://github.com/Sudarssan-N/LLMBrainAllignment.git
%cd {REPO}
!git pull
!pip -q install -e .[models,plot]""")

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
PARAMS_M = {'gpt2':124,'gpt2-medium':355,'opt-125m':125,'pythia-160m':160,
            'pythia-410m':410,'qwen2.5-0.5b':500,'qwen2.5-0.5b-instruct':500}

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
Decomposes voxel variance into **unique-hidden**, **shared**, and **unique-surprisal**. The
unique-hidden band dwarfs unique-surprisal and grows with depth — alignment is carried by
representations *beyond* surprisal.""")
code("""MODEL = 'gpt2'
v = load(MODEL,'varpart')['per_layer']
layers = sorted(int(l) for l in v)
uh=[v[str(l)]['mean_unique_hidden'] for l in layers]
sh=[v[str(l)]['mean_shared'] for l in layers]
us=[v[str(l)]['mean_unique_surprisal'] for l in layers]
plt.figure(figsize=(7,4))
plt.stackplot(layers, uh, sh, us, labels=['unique hidden','shared','unique surprisal'],
              colors=['#2c7fb8','#7fcdbb','#edf8b1'])
plt.legend(loc='upper left'); plt.xlabel('layer'); plt.ylabel('R$^2$')
plt.title(f'{MODEL}: variance partitioning'); plt.show()
peak = load(MODEL,'varpart').get('peak_layer')
d = v[str(peak)]
ci = d.get('ci_unique_hidden')
print(f"peak layer {peak}: unique_hidden={d['mean_unique_hidden']:.4f}",
      f"[{ci['lo']:.4f},{ci['hi']:.4f}]" if ci else '',
      f"| unique_surprisal={d['mean_unique_surprisal']:.4f}",
      f"| ratio ~{d['mean_unique_hidden']/max(d['mean_unique_surprisal'],1e-9):.0f}x")""")

md("""## 7. E003/E004 — Full model sweep
Size scaling (GPT-2 124M, OPT-125M, Pythia-160M/410M, Qwen2.5-0.5B) + the base-vs-instruct
Qwen pair. Last-token pooling throughout; the cached ceiling makes all but the first model fast.""")
code("""MODELS = ['gpt2','opt-125m','pythia-160m','pythia-410m',
          'qwen2.5-0.5b','qwen2.5-0.5b-instruct']
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
parameter count, testing *bigger ≠ more uniquely structured*.""")
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

xs, ys, names = [], [], []
for m in present:
    vj = load(m,'varpart')
    if not vj: continue
    peak = vj.get('peak_layer'); pl = vj['per_layer']
    if peak is None: peak = max(pl, key=lambda l: pl[l]['mean_unique_hidden'])
    xs.append(PARAMS_M.get(m, np.nan)); ys.append(pl[str(peak)]['mean_unique_hidden']); names.append(m)
ax[1].scatter(xs, ys)
for x,y,n in zip(xs,ys,names): ax[1].annotate(n,(x,y),fontsize=7,xytext=(3,3),textcoords='offset points')
ax[1].set(xscale='log', xlabel='params (M, log)', ylabel='peak unique-hidden R$^2$',
          title='Bigger ≠ more uniquely structured')
plt.tight_layout(); plt.show()""")

md("### 9a. E004 — Base vs instruction-tuned (paired bootstrap test)")
code("""!python scripts/07_compare.py --a qwen2.5-0.5b --b qwen2.5-0.5b-instruct
cmp = json.load(open(DERIVED/'compare_qwen2.5-0.5b_vs_qwen2.5-0.5b-instruct.json'))
print(cmp)
# bar of peak unique-hidden, base vs instruct
pairs = [('qwen2.5-0.5b','base'),('qwen2.5-0.5b-instruct','instruct')]
vals=[]; errs=[[],[]]
for m,_ in pairs:
    vj = load(m,'varpart'); peak=vj['peak_layer']; d=vj['per_layer'][str(peak)]
    ci=d.get('ci_unique_hidden'); vals.append(d['mean_unique_hidden'])
    errs[0].append(d['mean_unique_hidden']-ci['lo'] if ci else 0)
    errs[1].append(ci['hi']-d['mean_unique_hidden'] if ci else 0)
plt.figure(figsize=(4,4))
plt.bar([l for _,l in pairs], vals, yerr=errs, capsize=4, color=['#2c7fb8','#de2d26'])
plt.ylabel('peak unique-hidden R$^2$ (95% CI)'); plt.title('Instruction tuning'); plt.show()""")

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
