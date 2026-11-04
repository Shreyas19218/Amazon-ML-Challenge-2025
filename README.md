## 🏆 Achievement & Certificate

### Amazon ML Challenge 2025 — All India Rank 32

Proud to have secured **Rank 32 on the Amazon ML Challenge 2025 leaderboard** as part of **TeamEpic**, representing **Birla Institute of Technology, Mesra (BIT Mesra), Ranchi**.

The challenge involved developing machine learning solutions for real-world product price prediction by leveraging multimodal deep learning, representation learning, similarity-based feature engineering, and ensemble modeling techniques.

**Achievement Highlights:**
- 🥇 **Leaderboard Rank:** 32
- 🏢 **Competition:** Amazon ML Challenge 2025
- 👥 **Team Name:** TeamEpic
- 🎓 **Institution:** BIT Mesra, Ranchi
- 📅 **Certificate Date:** November 3, 2025

### Certificate of Participation

<p align="center">
  <img src="assets/certificates/Amazon_ML_Challenge_Certificate.jpeg" alt="Amazon ML Challenge 2025 Certificate - Rank 32" width="850"/>
</p>

<p align="center">
  <i>Official Certificate of Participation — Amazon ML Challenge 2025</i>
</p>

## 🧠 Methodology & Model Architecture

Our solution follows a **multimodal, embedding-driven price prediction framework** that combines deep learning representations, similarity-based feature engineering, and gradient boosting models to capture complex relationships between product attributes and their prices.

### 1. Target Transformation & Data Preprocessing
- Applied logarithmic transformation (`log1p`) to product prices to reduce target skewness, stabilize variance, and make the price distribution more suitable for regression modeling.
- Prepared textual, visual, and categorical product information for downstream embedding generation.
- Standardized the feature preparation pipeline to ensure consistent representations across training and validation datasets.

### 2. Multimodal Representation Learning
Developed rich product representations by leveraging pretrained deep learning models to extract complementary semantic and visual information.

- **CLIP:** Generated joint text-image embeddings to capture semantic relationships between product descriptions and their corresponding images.
- **Vision Transformer (ViT):** Extracted visual embeddings to learn product appearance, design, patterns, and other image-level characteristics.
- **DistilBERT:** Generated contextual text embeddings to capture product descriptions, linguistic patterns, and semantic relationships.
- Fine-tuned the embedding models using **Triplet Loss**, encouraging semantically and price-similar products to have closer representations while separating dissimilar products.
- Incorporated additional pretrained embeddings from product titles, descriptions, and categories to enrich the feature space with complementary semantic information.

### 3. FAISS-Based Similarity Search & KNN Feature Engineering
Leveraged **FAISS (Facebook AI Similarity Search)** to efficiently perform K-Nearest Neighbors (KNN) search across multiple embedding spaces.

For each embedding representation, retrieved the most similar products and engineered price-based neighborhood features, including:

- Mean and median prices of nearest neighbors.
- Standard deviation to capture price variability within similar products.
- Quantile-based statistics to understand the distribution of neighboring product prices.
- Inverse-distance-weighted price averages, assigning greater importance to more similar products.

Experimented with different neighborhood sizes (`K = 5` and `K = 10`) to capture both highly localized pricing patterns and broader product similarity relationships.

### 4. Unified Feature Engineering
- Combined multimodal embeddings, pretrained semantic representations, and FAISS-derived neighborhood statistics into a unified feature matrix.
- Integrated information from different embedding spaces to capture complementary product characteristics.
- Constructed a comprehensive feature representation that incorporates both individual product attributes and pricing patterns observed among similar products.

### 5. Feature Selection & SHAP-Based Optimization
- Trained an initial **LightGBM regression model** on the engineered feature matrix to establish feature importance and identify influential predictors.
- Applied **SHAP (SHapley Additive exPlanations)** to quantify individual feature contributions and understand their impact on price predictions.
- Identified and removed weak, redundant, and less informative features to reduce dimensionality and improve model efficiency.
- Retained the most relevant features to construct an optimized feature set for final model training.

### 6. Ensemble Modeling & Cross-Validation
Trained two complementary gradient boosting models on the refined feature set:

- **LightGBM:** Leveraged its efficient histogram-based gradient boosting approach to capture nonlinear feature interactions and complex pricing patterns.
- **CatBoost:** Utilized its boosting architecture to learn complex relationships and improve predictive robustness.

Implemented **K-Fold Cross-Validation** to evaluate model performance across multiple data splits and generate Out-of-Fold (OOF) predictions. These predictions provided a reliable basis for model evaluation and ensemble construction.

### 7. Multi-Scale KNN Ensemble & Final Price Prediction
- Trained and evaluated models using neighborhood features generated with different KNN configurations (`K = 5` and `K = 10`).
- Combined predictions from LightGBM and CatBoost models trained on these different neighborhood-based feature sets.
- Blended the resulting predictions to leverage both fine-grained local similarity patterns and broader global pricing relationships.
- Applied the appropriate inverse target transformation to convert predicted log-prices back into the original price scale.

The final ensemble produces price predictions by integrating multimodal product understanding, similarity-based pricing intelligence, feature selection, and complementary gradient boosting models.                                                         



             Product Dataset
                    |
          Data Preprocessing
                    |
          Log Price Transformation
                    |
       Multimodal Embedding Generation
                    |
       +------------+------------+
       |            |            |
      CLIP          ViT       DistilBERT
       |            |            |
       +------------+------------+
                    |
          Triplet Loss Fine-Tuning
                    |
        FAISS-Based KNN Retrieval
                    |
          K = 5       K = 10
             \        /
          Neighborhood Features
                    |
          Unified Feature Matrix
                    |
          LightGBM + SHAP Analysis
                    |
             Feature Selection
                    |
           Refined Feature Set
                    |
          K-Fold Cross Validation
                    |
          +---------+---------+
          |                   |
        CatBoost            LightGBM
          |                   |
          +---------+---------+
                    |
             Prediction Blending
                    |
          Inverse Log Transformation
                    |
             Final Price Prediction