
## Ego4D-NLQ

The repository provides two main sections:

### 1. Primary Analysis

Includes evaluations of VSL models with different feature sets:

1. **VSLNet** using Omnivor features
2. **VSLNet** using EgoVLP features
3. **VSLBase** using Omnivor features
4. **VSLBase** using EgoVLP features
5. **VSLNet** using EgoVLP features + GloVe embeddings

### 2. Extensions

Covers additional pretraining and transfer-learning experiments:

1. Pretraining on synthetic data (10% held out for validation)
2. Pretraining on the full synthetic dataset
3. Transfer learning of VSLNet on the original Ego4D dataset
4. Transfer learning of VSLNet on the original Ego4D dataset with a **frozen** feature encoder

---
