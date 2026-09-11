# GNN–BERT for Understanding Context from Music

Course: Neural Networks (CSE425 / EEE474 / CSE715)
Project: Supervised Neural Network Project — GNN-Based BERT for Understanding Context from Music

This repository implements the four-task roadmap from the project spec (`Task 1`–`Task 4`):
a BERT tag/caption classifier, a GraphSAGE/GAT structure encoder, a GNN–BERT cross-attention
fusion model, and a contrastive dual-encoder for retrieval.

## 0. Datasets used

Per project instructions, this implementation uses three datasets (all via Kaggle):

| # | Dataset | Kaggle link | Used for |
|---|---------|-------------|----------|
| 1 | MagnaTagATune | `yalumusic/magnatagatune` | Task 1 (BERT tags), Task 2/3 fusion labels, Task 4 (adapted) |
| 2 | GTZAN | `andradaolteanu/gtzan-dataset-music-genre-classification` | Task 2 (GNN genre classification), CNN baseline |
| 3 | DEAM | `imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music` | Task 3 auxiliary valence/arousal regression |

**Important — MusicCaps substitution:** the original spec's Task 4 (contrastive retrieval) and the
"Advanced" pairing are built around **MusicCaps** natural-language captions, which is not one of the
three datasets you specified. MusicCaps is not used anywhere in this repo. Instead, Task 4 is
**adapted** to run a contrastive graph↔text dual-encoder using MagnaTagATune's tag sets rendered as
short natural-language pseudo-captions (e.g. `"guitar, mellow, slow, acoustic"` →
`"a mellow, slow acoustic guitar piece"`) via `src/dataset.py:tags_to_pseudo_caption()`. This is
documented explicitly in the final report (Section 4.4) as a substitution, not hidden as if it were
real MusicCaps data. If you later get access to MusicCaps, `src/contrastive.py` works unmodified —
just point `config.yaml: data.musiccaps_csv` at it and set `data.use_musiccaps: true`.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs a CUDA GPU for realistic training times (BERT fine-tuning + GNN). CPU works for GTZAN-scale
debugging only.

## 2. Get the data

Download each Kaggle dataset (via `kaggle datasets download -d <slug>` or the website) and unzip into:

```
data/raw/magnatagatune/   # mp3 clips + annotations csv
data/raw/gtzan/           # genres_original/, images_original/ (unused), features_30_sec.csv etc.
data/raw/deam/            # audio/ + annotations/ (static + dynamic valence-arousal csv)
```

`config.yaml` has the exact expected sub-paths — adjust if your unzip layout differs (Kaggle mirrors
of these datasets vary slightly in folder naming).

## 3. Preprocess (Section 3 of spec: resample, spectrogram/chroma, segmentation, graph construction)

```bash
python -m src.audio_features --config config.yaml       # mel/chroma extraction + caching
python -m src.graph_builder  --config config.yaml        # chord-transition + segment graphs -> data/processed/graphs/*.pt
python -m src.dataset --config config.yaml --build_splits # train/val/test JSON, no-artist-leakage where labels exist
```

This produces at least 20 example graph files under `data/processed/graphs/` (`.pt`, PyTorch
Geometric `Data` objects) as required by the submission checklist.

## 4. Run each task

```bash
python -m src.train --task 1 --config config.yaml   # BERT-only tag classifier (Easy)
python -m src.train --task 2 --config config.yaml   # GraphSAGE/GAT on segment+chord graphs (Medium)
python -m src.train --task 3 --config config.yaml   # GNN-BERT cross-attention fusion (Hard)
python -m src.train --task 4 --config config.yaml   # Contrastive dual-encoder, MagnaTagATune pseudo-captions (Advanced/bonus)

python -m src.evaluate --task 3 --config config.yaml --checkpoint results/task3_best.pt
```

Each run writes `results/metrics.json`, `results/plots/*.png` (F1 vs epoch, t-SNE, PR curves),
and (Task 4) `results/retrieval_examples/*.json`.

## 5. Demo notebook

`notebooks/demo_context.ipynb` loads one trained checkpoint end-to-end: raw clip → mel/chroma →
graph → GNN embedding, raw tags/caption → BERT embedding → fusion → predicted tags + (if DEAM
sample) predicted valence/arousal.

## 6. Report

`report/final_report.tex` / `final_report.pdf` — IEEE two-column conference format, 6–10 pages,
following the rubric in Section 9 of the spec (Dataset & preprocessing, Model implementation,
Context understanding quality, Baseline comparison, Metrics & analysis, Report & presentation).

**The results tables in the shipped report are placeholders** (seeded from the spec's own
"illustrative performance comparison" table) because this environment cannot download the Kaggle
datasets or train models (no internet access, no GPU). Run the pipeline above on your machine or
Colab, then paste your real `results/metrics.json` numbers into `report/final_report.tex`
(search for `%% TODO: REPLACE WITH REAL RESULTS`) and recompile with:

```bash
cd report && pdflatex final_report.tex && pdflatex final_report.tex
```

## Repository structure

```
gnn-bert-music-context/
  README.md
  requirements.txt
  config.yaml
  data/
    raw/            # you place Kaggle downloads here (gitignored)
    processed/      # cached mel/chroma, BERT token caches, graphs/*.pt
    splits/         # train/val/test JSON
  notebooks/
    eda.ipynb
    demo_context.ipynb
  src/
    audio_features.py   # resample, log-mel (128 bins), chroma (12 bins), segmentation
    graph_builder.py     # chord-transition graph + segment-similarity graph -> PyG Data
    dataset.py            # dataset classes, splits, tag vocab, pseudo-caption builder
    bert_encoder.py        # HuggingFace BERT/DistilBERT text encoder wrapper
    gnn_model.py            # GraphSAGE / GAT encoder + mean-pool readout
    fusion_model.py          # Task 3: cross-attention fusion + multi-task loss
    contrastive.py            # Task 4: InfoNCE dual-encoder + retrieval eval
    cnn_baseline.py            # B2 baseline: CNN on mel-spectrogram
    train.py                    # CLI entry point, dispatches to task 1-4
    evaluate.py                  # metrics: Macro/Micro-F1, AUC-PR, MAE, R@K, t-SNE plot
    utils.py                      # seeding, config loading, checkpoint I/O
  results/
    metrics.json
    plots/
    retrieval_examples/
  report/
    final_report.tex
    final_report.pdf
```
