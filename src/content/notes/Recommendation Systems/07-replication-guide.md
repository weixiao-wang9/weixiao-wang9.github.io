---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 7: Replication Guide

> **Series**: X Recommendation Algorithm Deep Dive
> **Audience**: MLE Intern
> **Prerequisites**: Lectures 1-6

---

## Overview

You've learned how X's recommendation algorithm works. Now what?

This lecture is a practical roadmap for:
1. **Understanding what's replicable** vs what requires proprietary data
2. **Building your own two-tower recommender** using public datasets
3. **Training and evaluating** the models
4. **Adding this to your portfolio** to land MLE jobs

---

## What's Replicable vs What's Not

```
┌─────────────────────────────────────────────────────────────┐
│                    YOU CAN REPLICATE                        │
├─────────────────────────────────────────────────────────────┤
│  ✓ Two-tower retrieval architecture                         │
│  ✓ Transformer ranking model                                │
│  ✓ Hash-based embeddings (or standard embeddings)           │
│  ✓ Multi-action prediction (19 engagement types)            │
│  ✓ Candidate isolation attention mask                       │
│  ✓ Training pipeline (loss, sampling, optimization)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                YOU CANNOT EASILY REPLICATE                  │
├─────────────────────────────────────────────────────────────┤
│  ✗ Billions of historical user-post interactions            │
│  ✗ Real-time engagement feedback (for online learning)      │
│  ✗ Production infrastructure (Thunder, Home Mixer)          │
│  ✗ Pre-trained hash embeddings (trained on X's data)        │
│  ✗ The exact model weights (proprietary)                    │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: You can replicate the **algorithm** using public data, even if you can't match X's **model quality**.

---

## Recommended Dataset: MIND (Microsoft News)

The **MIND (Microsoft News Dataset)** is the best public dataset for learning news recommendation. It's similar enough to X's use case (short text, user engagement history).

### Why MIND?

| Feature | MIND | X (For You) |
|---------|------|-------------|
| **Content type** | News articles | Short posts/tweets |
| **User feedback** | Clicks, dwell time | Likes, replies, reposts |
| **Scale** | 1M users, 160K articles | 100M+ users, billions of posts |
| **Data format** | Impression logs | Engagement logs |
| **Accessibility** | Public download | Proprietary |

### Download MIND

```bash
# Create project directory
mkdir ~/mind-recsys && cd ~/mind-recsys

# Download MIND dataset (small version for learning)
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip

# Unzip
unzip MINDsmall_train.zip
unzip MINDsmall_dev.zip

# Structure:
# MINDsmall_train/
#   - news.tsv         # Article metadata
#   - behaviors.tsv    # User impression logs
#   - entity_embedding.vec  # Optional: pretrained embeddings
```

### MIND Data Format

**news.tsv** (Article metadata):
```
N1234    <title>Breaking: New AI Model Released</title>    <category>Tech</category>    <abstract>...</abstract>
N5678    <title>Markets Hit All-Time High</title>         <category>Business</category> <abstract>...</abstract>
...
```

**behaviors.tsv** (User impressions):
```
1   U1234    11/12/2019 10:30:00    N1234 N5678 N9012    N1234-0 N5678-0 N9012-1 N3456-1
                                                 ↑        ↑        ↑        ↑
                                              clicked  clicked  clicked  skipped
```

Each line = one user session:
- `Impression`: 3 candidate articles shown to user
- `Click labels`: 0 = skipped, 1 = clicked

---

## Project Structure

Here's a clean PyTorch project structure:

```
mind-recsys/
├── data/
│   ├── raw/                    # Downloaded MIND files
│   │   ├── news.tsv
│   │   ├── behaviors.tsv
│   │   └── entity_embedding.vec
│   └── processed/              # Preprocessed data
│       ├── train_dataset.pt
│       ├── val_dataset.pt
│       └── corpus_embeddings.pt
│
├── src/
│   ├── dataset.py              # MIND data loading
│   ├── models/
│   │   ├── retrieval.py        # Two-tower model
│   │   ├── ranking.py          # Transformer ranking model
│   │   └── embeddings.py       # Hash embeddings (optional)
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Metrics (AUC, recall@K)
│   └── config.py               # Hyperparameters
│
├── notebooks/
│   ├── 01_exploring_data.ipynb
│   ├── 02_two_tower_retrieval.ipynb
│   └── 03_ranking_model.ipynb
│
├── checkpoints/                # Saved models
├── logs/                       # TensorBoard logs
├── requirements.txt
└── README.md
```

---

## Step-by-Step Implementation Plan

### Step 1: Data Preprocessing (1-2 days)

**Goal**: Convert MIND TSV files into PyTorch datasets.

```python
# src/dataset.py

import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd

class MINDDataset(Dataset):
    """MIND dataset for two-tower training."""

    def __init__(self, behaviors_path, news_path, max_history=50):
        """
        Args:
            behaviors_path: Path to behaviors.tsv
            news_path: Path to news.tsv
            max_history: Max number of past clicks to include in user history
        """
        # Load news metadata
        self.news_df = pd.read_csv(
            news_path,
            sep='	',
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'entity'],
            usecols=['news_id', 'title', 'category']
        )

        # Create news_id to index mapping
        self.news_id_to_idx = {nid: i for i, nid in enumerate(self.news_df['news_id'])}

        # Load behaviors and build user histories
        self.user_histories = defaultdict(list)
        behaviors_df = pd.read_csv(
            behaviors_path,
            sep='	',
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )

        # Build click history per user
        for _, row in behaviors_df.iterrows():
            if pd.notna(row['history']):
                clicked_news = row['history'].split()
                for news_click in clicked_news:
                    nid, _ = news_click.split('-')
                    self.user_histories[row['user_id']].append(nid)

        # Create training samples from impressions
        self.samples = []
        for _, row in behaviors_df.iterrows():
            user_id = row['user_id']
            history = self.user_histories[user_id][-max_history:]  # Last N clicks

            # Parse impression: "N1234-0 N5678-1 N9012-0"
            impressions = row['impressions'].split()
            for imp in impressions:
                nid, label = imp.split('-')
                self.samples.append({
                    'user_id': user_id,
                    'history': history,  # List of clicked news_ids
                    'candidate_id': nid,
                    'label': int(label)  # 0 or 1
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert news_ids to indices
        history_indices = [
            self.news_id_to_idx[nid]
            for nid in sample['history']
            if nid in self.news_id_to_idx
        ]
        candidate_idx = self.news_id_to_idx.get(sample['candidate_id'], 0)

        return {
            'history_indices': torch.tensor(history_indices, dtype=torch.long),
            'candidate_idx': torch.tensor(candidate_idx, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.float32)
        }

# Usage
train_dataset = MINDDataset(
    behaviors_path='data/raw/MINDsmall_train/behaviors.tsv',
    news_path='data/raw/MINDsmall_train/news.tsv',
    max_history=50
)
```

---

### Step 2: Two-Tower Retrieval Model (2-3 days)

**Goal**: Build the two-tower architecture from Lecture 3.

```python
# src/models/retrieval.py

import torch
import torch.nn as nn

class UserTower(nn.Module):
    """Encodes user history into a fixed-length embedding."""

    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.news_embedding = nn.Embedding(vocab_size, emb_dim)

        # Aggregate history: mean pool + projection
        self.history_aggregator = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, history_indices):
        """
        Args:
            history_indices: [B, H] tensor of news indices

        Returns:
            user_embedding: [B, D] normalized user embedding
        """
        # [B, H, D]
        history_emb = self.news_embedding(history_indices)

        # Mean pooling over history: [B, D]
        history_pooled = history_emb.mean(dim=1)

        # Project: [B, D]
        user_emb = self.history_aggregator(history_pooled)

        # L2 normalize
        user_emb = nn.functional.normalize(user_emb, p=2, dim=-1)

        return user_emb


class CandidateTower(nn.Module):
    """Encodes news articles into fixed-length embeddings."""

    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.news_embedding = nn.Embedding(vocab_size, emb_dim)

        # Expand-compress MLP (from Lecture 3)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, emb_dim)
        )

    def forward(self, candidate_idx):
        """
        Args:
            candidate_idx: [B] tensor of news indices

        Returns:
            candidate_embedding: [B, D] normalized candidate embedding
        """
        # [B, D]
        candidate_emb = self.news_embedding(candidate_idx)

        # [B, D]
        candidate_emb = self.mlp(candidate_emb)

        # L2 normalize
        candidate_emb = nn.functional.normalize(candidate_emb, p=2, dim=-1)

        return candidate_emb


class TwoTowerRetrieval(nn.Module):
    """Two-tower retrieval model."""

    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.user_tower = UserTower(vocab_size, emb_dim, hidden_dim)
        self.candidate_tower = CandidateTower(vocab_size, emb_dim, hidden_dim)

    def forward(self, history_indices, candidate_idx):
        """
        Args:
            history_indices: [B, H] tensor of news indices
            candidate_idx: [B] tensor of candidate news indices

        Returns:
            similarity: [B] dot product similarity
        """
        user_emb = self.user_tower(history_indices)  # [B, D]
        candidate_emb = self.candidate_tower(candidate_idx)  # [B, D]

        # Dot product (cosine similarity since normalized)
        similarity = (user_emb * candidate_emb).sum(dim=-1)  # [B]

        return similarity
```

---

### Step 3: Training Loop (1 day)

**Goal**: Train the two-tower model with contrastive loss.

```python
# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_two_tower(model, train_dataset, val_dataset, config):
    """
    Train two-tower model with binary cross-entropy loss.

    Args:
        model: TwoTowerRetrieval model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training hyperparameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()  # Binary classification

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0

        for batch in train_loader:
            history_indices = batch['history_indices'].to(device)
            candidate_idx = batch['candidate_idx'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            similarity = model(history_indices, candidate_idx)
            loss = criterion(similarity, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        val_auc = evaluate(model, val_dataset, device)

        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"  Val AUC: {val_auc:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"checkpoints/two_tower_epoch{epoch}.pt")

def evaluate(model, dataset, device):
    """Compute AUC for validation set."""
    model.eval()
    from sklearn.metrics import roc_auc_score

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=256):
            history_indices = batch['history_indices'].to(device)
            candidate_idx = batch['candidate_idx'].to(device)
            labels = batch['label'].cpu().numpy()

            similarity = model(history_indices, candidate_idx)
            scores = torch.sigmoid(similarity).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels)

    return roc_auc_score(all_labels, all_scores)
```

---

### Step 4: Ranking Model (3-4 days)

**Goal**: Build the transformer ranking model from Lecture 4.

This is more complex. You can either:
- **Option A**: Use a simpler architecture (e.g., concat + MLP) for the ranking model
- **Option B**: Implement the full transformer with candidate isolation

For a portfolio project, **Option A** is sufficient and still impressive.

```python
# src/models/ranking.py

class RankingModel(nn.Module):
    """Simplified ranking model: concat features + MLP."""

    def __init__(self, emb_dim=256, hidden_dim=512, num_actions=1):
        """
        For MIND, num_actions=1 (just click prediction).
        For X, num_actions=19.
        """
        super().__init__()
        self.news_embedding = nn.Embedding(100000, emb_dim)  # Vocab size

        # Aggregate user history
        input_dim = emb_dim * 2  # User emb + candidate emb

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, history_indices, candidate_idx):
        """
        Args:
            history_indices: [B, H] tensor of history news indices
            candidate_idx: [B] tensor of candidate news indices

        Returns:
            logits: [B, num_actions] prediction logits
        """
        # Encode user history
        history_emb = self.news_embedding(history_indices)  # [B, H, D]
        user_emb = history_emb.mean(dim=1)  # [B, D] (mean pooling)

        # Encode candidate
        candidate_emb = self.news_embedding(candidate_idx)  # [B, D]

        # Concatenate
        features = torch.cat([user_emb, candidate_emb], dim=-1)  # [B, 2D]

        # Predict
        logits = self.mlp(features)  # [B, num_actions]

        return logits
```

---

### Step 5: Evaluation Metrics (1 day)

**Key metrics for recommenders:**

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **AUC** | Area under ROC curve | Ranking quality (higher = better) |
| **Recall@K** | `hits@K / total_relevant` | Did we retrieve relevant items? |
| **Precision@K** | `hits@K / K` | How many retrieved are relevant? |
| **NDCG@K** | Weighted relevance | Position-aware ranking quality |

```python
# src/evaluate.py

import torch
import numpy as np

def recall_at_k(model, dataset, k=10, device='cuda'):
    """Compute Recall@K for retrieval."""
    model.eval()

    # Pre-compute all candidate embeddings (corpus)
    corpus_embeddings = []
    news_ids = []

    for idx in range(len(dataset.news_df)):
        news_ids.append(idx)
        candidate_idx = torch.tensor([idx]).to(device)
        emb = model.candidate_tower(candidate_idx)
        corpus_embeddings.append(emb.cpu())

    corpus_embeddings = torch.cat(corpus_embeddings)  # [N, D]

    recalls = []

    for sample in dataset.samples:
        # Get user embedding
        history_indices = torch.tensor([sample['history_indices']]).to(device)
        user_emb = model.user_tower(history_indices)  # [1, D]

        # Compute similarity to all candidates
        scores = (user_emb @ corpus_embeddings.T).squeeze()  # [N]

        # Get top-k
        top_k_indices = torch.topk(scores, k).indices

        # Check if true candidate is in top-k
        true_idx = sample['candidate_idx']
        recall = 1.0 if true_idx in top_k_indices else 0.0
        recalls.append(recall)

    return np.mean(recalls)
```

---

## Training Tips

### 1. Negative Sampling

For retrieval training, you need both positive (clicked) and negative (skipped) examples.

```python
# In dataset.__getitem__

# Option 1: Use implicit negatives (from impressions)
# The dataset already has this: label=0 means skipped

# Option 2: Sample random negatives during training
def sample_negatives(candidate_idx, num_negatives, vocab_size):
    """Sample random negative candidates."""
    neg_indices = torch.randint(0, vocab_size, (num_negatives,))
    # Make sure we don't sample the positive
    neg_indices = neg_indices[neg_indices != candidate_idx][:num_negatives]
    return neg_indices
```

### 2. Loss Functions

```python
# Option 1: Binary Cross-Entropy (simple)
criterion = nn.BCEWithLogitsLoss()

# Option 2: Contrastive Loss (better for retrieval)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, pos_sim, neg_sim):
        """
        pos_sim: [B] similarity with positive candidates
        neg_sim: [B] similarity with negative candidates
        """
        loss = torch.relu(self.margin - pos_sim + neg_sim).mean()
        return loss

# Usage:
# pos_sim = model(user_emb, pos_candidates)
# neg_sim = model(user_emb, neg_candidates)
# loss = contrastive_loss(pos_sim, neg_sim)
```

### 3. Curriculum Learning

Start with easy negatives, progressively use harder ones.

```python
# Training loop
for epoch in range(num_epochs):
    # Early epochs: use random negatives
    if epoch < 5:
        neg_candidates = sample_random_negatives()

    # Later epochs: use hard negatives (high scoring wrong items)
    else:
        neg_candidates = sample_hard_negatives(model, user_emb)
```

---

## Hyperparameter Tuning

Key hyperparameters to tune:

| Hyperparameter | Typical Range | Impact |
|----------------|---------------|--------|
| **Embedding dim** | 128-512 | Higher = more capacity, slower |
| **Learning rate** | 1e-4 to 1e-3 | Too high = unstable, too low = slow |
| **Batch size** | 64-512 | Larger = more stable gradients |
| **History length** | 10-100 | Longer = more context, noisier |
| **Negative samples** | 1-10 | More negatives = better discrimination |

**Tip**: Start with a small model (emb_dim=128, 2-layer MLP) and scale up once it works.

---

## Adding to Your Portfolio

### What to Show

1. **GitHub Repository** (public)
   - Clean code structure
   - README with setup instructions
   - Training logs and model checkpoints

2. **Demo/Visualization**
   - Jupyter notebook showing:
     - Data exploration (word clouds, engagement distribution)
     - Embedding visualization (t-SNE plot)
     - Example recommendations for a sample user

3. **Write-up** (blog post or PDF)
   - Problem statement
   - Architecture diagram
   - Results (AUC, Recall@K curves)
   - Lessons learned

4. **Live Demo** (optional, impressive)
   - Gradio or Streamlit app
   - User enters a news ID
   - Model recommends similar articles

### Resume Bullet Points

```
Projects:
• Built two-tower retrieval model for news recommendation using PyTorch
  - Trained on MIND dataset (1M users, 160K articles)
  - Achieved 0.72 AUC and 0.35 Recall@10
  - Implemented contrastive loss with hard negative mining
  - Deployed demo app with Streamlit (100+ users)
```

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| **Overfitting to training users** | Use regularization, dropout, early stopping |
| **Cold start problem** | Use content features (title, category) for new items |
| **Embedding collapse** | L2 normalize embeddings, use contrastive loss |
| **Slow training** | Pre-compute candidate embeddings, use gradient checkpointing |
| **Poor retrieval quality** | Increase negative samples, use hard negative mining |

---

## Going Beyond: Advanced Topics

Once you have the basics working, consider:

1. **Multi-task learning**: Predict multiple engagement types (like X's 19 actions)
2. **Transformer ranking**: Implement the full candidate isolation transformer
3. **Hash embeddings**: Replace standard embeddings with hash-based embeddings (Lecture 2)
4. **Online learning**: Update model with new user interactions (hard without production infra)
5. **A/B testing framework**: Simulate A/B tests using held-out validation data

---

## Resources

### Datasets
- [MIND Dataset](https://msnews.github.io/) - Microsoft News recommendation
- [MovieLens](https://grouplens.org/datasets/movielens/) - Movie ratings
- [Amazon Reviews](https://nijianmo.github.io/amazon/index.html) - Product recommendations

### Papers
- [Two-Tower Model](https://arxiv.org/abs/1904.04698) - "Self-Attentive Sequential Recommendation"
- [BERT4Rec](https://arxiv.org/abs/1904.06653) - Sequential recommendation with transformers
- [SASRec](https://arxiv.org/abs/1808.09781) - Self-Attentive Sequential Recommendation

### Code Examples
- [X Algorithm Codebase](https://github.com/xai-org/x-algorithm) - This repository
- [TorchRec](https://github.com/pytorch/torchrec) - PyTorch's recommendation library
- [Facebook DLRM](https://github.com/facebookresearch/dlrm) - Deep learning recommendation model

---

## Summary

**You can build a working two-tower recommender in 1-2 weeks:**

1. **Week 1**: Data preprocessing + two-tower model + basic training
2. **Week 2**: Ranking model + evaluation + portfolio write-up

**The key is to start simple and iterate.**

Don't try to replicate X's full system. Focus on the core ML components (retrieval + ranking) and demonstrate understanding of the architecture.

---

## Final Checklist

Before considering this project "portfolio-ready":

- [ ] Two-tower model trained with AUC > 0.65
- [ ] Ranking model implemented (simple or transformer)
- [ ] Evaluation metrics computed (Recall@K, NDCG@K)
- [ ] Clean GitHub repo with README
- [ ] Jupyter notebook with visualizations
- [ ] Demo app (optional but recommended)
- [ ] Blog post or write-up explaining the approach

---

## Congratulations!

You've completed the X Algorithm Deep Dive series. You now understand:

1. How X's "For You" feed works end-to-end
2. Hash-based embeddings for billion-entity vocabularies
3. Two-tower retrieval for efficient candidate generation
4. Transformer ranking with multi-action prediction
5. The full inference pipeline (sources, filters, scorers)
6. How to build your own recommender system

**Next steps:**
- Build the MIND recommender (1-2 weeks)
- Add it to your portfolio
- Apply for MLE/recommendation system roles

**Good luck!**

---

**End of Series**
