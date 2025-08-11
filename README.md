# Negotiation Strategy Recommendation System

Personalized negotiation-strategy recommendations by combining **dialogue understanding** (BERT) with **personality profiling** and lightweight **recommender methods**. Built around the CaSiNo campsite-negotiation dataset.

> 📝 This repository contains the research notebook and the report titled **Negotiation Strategy Recommendation System**.

## Highlights
- **BERT-SC strategy classifier** fine-tuned to recognize **9 negotiation strategies** from dialogue utterances.
- **Personality-aware recommendations** using k-means clustering (k=8) over Big Five traits + resource priorities.
- **Two recommenders implemented**: personality-cluster matcher and SVD-based latent-factor model (ALS was explored but not used due to sparsity).
- **Result:** Macro performance of ~**0.796 F1** across strategies (vs. ~**0.67** in the original CaSiNo paper), with strongest classes: *small-talk*, *elicit-pref*, and *non-strategic*.

## Repository Contents
- `Negotiation Strategy Recommendation System.ipynb` — end-to-end notebook: data prep, modeling, evaluation, and recommendation demos.
- `Final Report.pdf` — full write-up with background, method, results, limitations, and references.

## Problem & Dataset  
The project tackles the challenge of recommending effective negotiation strategies tailored to an individual’s personality and situational context. It focuses on dyadic resource-allocation negotiations, where two users collaborate to reach a mutually beneficial agreement.  

The system is developed and evaluated using the **CaSiNo (Conversational Negotiation) dataset**, which contains campsite negotiation dialogues. Each utterance is annotated with one of nine negotiation strategy labels, and each participant is profiled with **Big Five personality traits**.  

The nine negotiation strategies are:  
- small-talk  
- empathy  
- coordination  
- no-need  
- elicit-pref  
- undervalue-partner  
- vouch-fairness  
- self-need  
- other-need  

## Method Overview
### 1) BERT-SC (Strategy Classifier)
- **Base model:** `bert-base-uncased` (Hugging Face Transformers).
- **Head:** 9-way classifier (one-vs-many setup via softmax).
- **Training:** ~7k labeled utterances, 80/20 split; lr=2e-5, batch=16, epochs=4, dropout=0.1, weight-decay=0.01; early stopping + grad clipping.

### 2) Personality Clustering
- **Feature vector (8-D):** Big Five (5) + resource-priority one-hot/ordinal (3).
- **Clustering:** k-means with **k=8** (balance of cohesion and size).

### 3) Strategy Effectiveness Matrix
- For each cluster, compute normalized “strategy effectiveness” scores from dialogue outcomes to form an **8×9 matrix**.
- Use this matrix both for direct recommendations and as input to factor models.

### 4) Recommenders
- **Personality-based matcher:** find nearest cluster (Euclidean), rank strategies by effectiveness, return Top-N with short “why it works” rationales and example utterances.
- **SVD recommender:** partial SVD over the cluster–strategy matrix to uncover latent relations; project new users and score strategies.
- **ALS:** investigated but not robust with the small, sparse matrix; omitted from final pipeline.

## Results (short version)
- **F1 (weighted): ~0.796** across strategies.
- Best-performing strategies: **small-talk**, **elicit-pref**, **non-strategic**.
- Tough classes: **showing-empathy**, **undervalue-partner**, and **no-need** (negation confusions).
- Personality-aware matrix shows cluster-specific preferences; broadly, **elicit-pref** and **self-need** show strong effectiveness for many profiles, while SVD sometimes surfaces **small-talk** as a useful complementary tactic.

See the report for full tables, per-class scores, and the cluster–strategy matrix.

## Quickstart
### Environment
```bash
# Install PyTorch (CPU version shown here) [GPU HIGHLY RECOMMENDED]
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install transformers implicit scikit-learn scipy numpy pandas matplotlib seaborn tqdm threadpoolctl
```

### Run the notebook
Open `Negotiation Strategy Recommendation System.ipynb` and execute all cells. The notebook will:
1) load & preprocess CaSiNo,
2) fine-tune BERT-SC,
3) build clusters and the strategy-effectiveness matrix,
4) run the recommenders and print Top-N suggestions with rationales.

> 💡 Tip: If you don’t have CaSiNo locally, add a cell to download or place the processed files according to your paths.

## Limitations & Future Work
- The **no-need** class is challenging (negation handling); consider enhanced negation-aware modeling and data augmentation.
- CaSiNo’s **single domain** (camping) limits generalization; test on business-style negotiation corpora when available.
- **ALS** was unstable with the current sparsity; a denser matrix or alternative regularization might be helpful.
- An **interactive assistant** that adapts strategy as the dialogue evolves would be a natural next step.

## Acknowledgments
- [CaSiNo dataset authors](https://github.com/kushalchawla/CaSiNo)
- [Hugging Face Transformers (bert-base-uncased)](https://huggingface.co/google-bert/bert-base-uncased)
- [Prof. Pablo Robles-Granda](https://scholar.google.com/citations?user=f5QdoegAAAAJ&hl=en) and peers who provided feedback during the independent study
