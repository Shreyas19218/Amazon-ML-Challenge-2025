# 🛒 Smart Product Pricing: Amazon ML Challenge 2025

> A multimodal ML pipeline that predicts e-commerce product prices from text and images, using **price-aware fine-tuned embeddings**, **FAISS KNN price features**, and a **CatBoost + LightGBM ensemble**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA%20fine--tuning-ee4c2c)
![FAISS](https://img.shields.io/badge/FAISS-GPU%20KNN-informational)
![CatBoost](https://img.shields.io/badge/CatBoost-LightGBM-green)

**Result:** `[Leaderboard rank / total teams]` · `[Validation SMAPE: XX.XX%]` · `[Public / private leaderboard SMAPE]`

---

## 📌 The Problem

Given a product's catalog text (title, description, Item Pack Quantity) and its image, predict its price. The task is scored with **SMAPE** (Symmetric Mean Absolute Percentage Error).

| | |
|---|---|
| **Train set** | 75k products with prices |
| **Test set** | 75k products (25k on the public leaderboard) |
| **Inputs** | `catalog_content` (text) + `image_link` (image) |
| **Target** | `price` |

---

## 🎯 Highlights

- **Price-aware embeddings:** fine-tuned text encoders with **triplet loss** so products with similar prices sit close together in embedding space.
- **Parameter-efficient training:** **LoRA** adapters on DistilUSE (small trainable fraction, fast iteration).
- **18 embedding views** (13 text + 5 image), mixing pretrained and fine-tuned models, each turned into KNN price statistics.
- **Rich KNN feature engineering:** about 25 neighbor-price statistics per view, plus cross-model and cross-modality consensus features.
- **Two-stage ensembling:** LightGBM for feature selection, then a LightGBM + CatBoost blend, then a multi-K (K=5, K=10) blend.
- **Scalable inference:** batched, GPU-accelerated FAISS search with leave-one-out handling to avoid target leakage.

---

## 🧠 Approach

```
                ┌─ 8 pretrained text encoders ─────┐
Catalog text ───┤                                  │
                └─ 5 fine-tuned / custom text ─────┤
                                                   ├─► FAISS KNN (cosine) ─► price statistics per view
                ┌─ 3 pretrained image encoders ────┤                                  │
Product image ──┤                                  │                                  ▼
                └─ 2 fine-tuned image encoders ────┘              Consensus + modality-agreement features
                                                                                      │
                                                                                      ▼
                                          LightGBM (feature selection) ─► LightGBM + CatBoost ─► Multi-K blend ─► Price
```

### 1. Target transformation
Models are trained on **log-transformed prices** (`log1p`) to stabilize variance, reduce the impact of outliers, and learn relative rather than absolute differences. Predictions are mapped back with `expm1`.

### 2. Price-aware embedding fine-tuning (`general_finetuner.py`)
Standard sentence embeddings capture *semantic* similarity, but two semantically similar products can have very different prices. To fix this, the encoder is fine-tuned so that *price* structure is reflected in the embedding space.

| Component | Detail |
|---|---|
| Backbone | `distiluse-base-multilingual-cased-v2` (DistilBERT, 512-dim output) |
| Adaptation | LoRA on attention projections (`q_lin`, `k_lin`, `v_lin`, `out_lin`), r=8, α=16 |
| Loss | Triplet loss on cosine distance (margin 0.5) |
| Triplet mining | Prices are quantile-binned in log space (16 bins). The positive comes from the same bin. The negative comes from a bin at least 1.0 apart in log-price, relaxed adaptively if the pool is too small. |
| Training | 500k triplets over 20 epochs, AdamW, linear warmup, mixed precision (AMP), gradient clipping |

The same triplet idea was applied to additional encoders (CLIP text/vision, DistilBERT single-field and two-field, DINOv2).

### 3. Embedding views
- **Pretrained text:** BGE-large, MPNet, Nomic, MixedBread, Snowflake Arctic, MiniLM variants, DistilUSE
- **Fine-tuned text:** CLIP-text, DistilBERT, DistilUSE, DistilBERT 2-field
- **Pretrained image:** CLIP ViT-L/14, SigLIP, Nomic-Vision
- **Fine-tuned image:** DINOv2, CLIP-vision

### 4. KNN price features (`ensemble_model.py`)
For every embedding view, FAISS retrieves the nearest training products and summarizes their prices:

- Raw neighbor prices and similarity scores
- Mean, median, min, max, std, quantiles (P25/P75), IQR, range, coefficient of variation
- Inverse-distance-weighted averages, top-5 / top-10 means
- Price ratios, distance-normalized stats, and "momentum" and gradient across neighbor rank
- **Consensus features:** agreement across all models, and between text vs. image signals

Training rows use **leave-one-out** neighbors, so a product never sees its own price.

### 5. Meta-learner and blending
1. Train LightGBM on the full feature set and drop the bottom 40% of features by importance.
2. Retrain **LightGBM** and **CatBoost** (median/quantile loss) on the selected features with early stopping.
3. Blend per K: `0.2 × LightGBM + 0.8 × CatBoost`.
4. Final prediction: `0.4 × (K=5) + 0.6 × (K=10)`. Small K captures local pricing patterns, larger K adds stability.

---

## 📁 Repository Structure

```
├── general_finetuner.py   # LoRA + triplet-loss fine-tuning of the text encoder
├── ensemble_model.py      # KNN feature generation, LGBM/CatBoost training, blending, submission
├── docs/                  # Technical write-up (PDF)
├── features/              # (not tracked) cached embeddings: train/ and test/
├── train.csv / test.csv   # (not tracked) challenge data
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/Shreyas19218/Amazon-ML-Challenge-2025.git
cd Amazon-ML-Challenge-2025
pip install torch transformers peft sentence-transformers faiss-gpu \
            lightgbm catboost scikit-learn pandas numpy tqdm
```

**Run order**
1. Place `train.csv` and `test.csv` in the project root.
2. Generate and cache embeddings into `features/train/` and `features/test/`.
3. *(Optional)* Fine-tune the encoder: `python general_finetuner.py`
4. Train the ensemble and write predictions: `python ensemble_model.py`

> A CUDA GPU is strongly recommended for fine-tuning and FAISS search.

---

## 🛠️ Tech Stack

**Python · PyTorch · Hugging Face Transformers · PEFT/LoRA · Sentence-Transformers · FAISS · LightGBM · CatBoost · scikit-learn · pandas · NumPy**

---

## 📚 What I Learned

- Turning a regression problem into a **retrieval problem** (KNN price statistics) can beat predicting directly from embeddings.
- Fine-tuning embeddings with a **task-specific metric objective** (price proximity) adds signal that generic embeddings miss.
- Combining many weak and diverse views (text, image, pretrained, fine-tuned) with **consensus features** improves robustness.
- Memory-aware engineering: batched GPU search, leave-one-out handling, and careful leakage control.

## 🔮 Future Improvements

- Out-of-fold KNN features with proper K-fold stacking
- Hyperparameter search (Optuna) for blend weights instead of hand-tuned values
- Fine-tune a vision-language model end to end
- Targeted handling of high-price outliers

---

## 🙏 Acknowledgements

The methodology is inspired by the [Amazon ML Challenge 2023 winning solution](https://github.com/pj-mathematician/Amazon-ML-Challenge-2023/blob/main/amazon-ml-challenge-2023-winner-solution.ipynb) (embedding-based features, KNN statistics, ensemble blending), extended here to a multimodal setting with newer embedding models.

## 👤 Author

**Shreyas** · [LinkedIn](https://linkedin.com/in/your-profile) · [Email](mailto:your-email@example.com) · [Portfolio](https://your-site.com)

*Built for Amazon ML Challenge 2025 using only the provided training data, with no external price lookups.*
