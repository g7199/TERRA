# TERRA

Post-hoc TF-IDF reranking for sequential recommendation. Given a pretrained backbone and a review corpus, we blend the backbone's scores with a content similarity profile built from TF-IDF + Truncated SVD on item reviews. The backbone is used as-is; no retraining.

## Setup

```bash
bash setup.sh                 # creates .venv, installs deps, smoke test
source .venv/bin/activate
```

Put the raw Amazon 2014 5-core review files in `data/`:

```
data/reviews_Beauty_5.json.gz
data/reviews_Toys_and_Games_5.json.gz
data/reviews_Sports_and_Outdoors_5.json.gz
data/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
data/reviews_Video_Games_5.json.gz
```

Source: http://jmcauley.ucsd.edu/data/amazon/

## Run

```bash
python scripts/run_dataset.py --dataset Beauty --backbone BSARec --seed 42
```

`--backbone` is one of: `GRU4Rec`, `SASRec`, `FMLPRec`, `BERT4Rec`, `DuoRec`, `BSARec`.
`--seed` is any int. `--gpu` picks the CUDA device.

Outputs:

- `data/processed/<Dataset>.{txt,_mapping.json}` — 5-core leave-one-out sequences + id map
- `backbone/output/<Backbone>_<Dataset>_seed<N>.pt` — trained backbone
- `results/tfidfrank.csv` — one row per run with `best_alpha`, `ndcg10_base`, `ndcg10_best`, HR/NDCG at k ∈ {5, 10, 20}

`--skip_preprocess` / `--skip_train` let you reuse previous outputs.

## Pipeline

1. **Preprocess** (`preprocess/preprocess.py`): 5-core iterative filter on users and items, sort by `unixReviewTime`, write `<DS>.txt` (one line per user, 1-indexed) and an id mapping.
2. **Train backbone** (`backbone/main.py`): leave-one-out split (train=`seq[:-2]`, val=`seq[-2]`, test=`seq[-1]`), early stop on val MRR. Default paper hyperparameters per backbone.
3. **Evaluate TF-IDFRank** (`tfidfrank/run_tfidfrank.py`):
   - Build `val_prof` from reviews excluding `(u, seq[-1])` and `(u, seq[-2])`.
   - Build `test_prof` from reviews excluding only `(u, seq[-1])`.
   - For α ∈ {0.0, 0.05, …, 1.0}: `blended = (1-α) · norm(backbone) + α · content_sim`, pick α\* maximizing val NDCG@10.
   - Apply α\* on test. Report NDCG/HR at k ∈ {5, 10, 20}.

## Reproduce Table 2

```bash
for DS in Beauty Toys_and_Games Sports_and_Outdoors Clothing_Shoes_and_Jewelry Video_Games; do
  for BB in GRU4Rec SASRec FMLPRec BERT4Rec DuoRec BSARec; do
    for S in 42 123 456; do
      python scripts/run_dataset.py --dataset $DS --backbone $BB --seed $S
    done
  done
done
```

Then aggregate `results/tfidfrank.csv` by `(dataset, backbone)` over the three seeds.

## Layout

```
tfidfrank_release/
├── README.md
├── requirements.txt
├── setup.sh
├── data/
├── preprocess/preprocess.py
├── backbone/
│   ├── main.py
│   ├── trainers.py
│   ├── dataset.py
│   ├── utils.py
│   ├── metrics.py
│   └── model/          # 6 backbones + shared modules
├── tfidfrank/run_tfidfrank.py
├── scripts/run_dataset.py
└── results/
```
