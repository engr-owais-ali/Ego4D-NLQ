

## Extensions

**Prerequisites**
* The dataset folder contains all necessary video features, GloVe embeddings and 2 version of synthetic dataset.
* **Full Synthetic dataset**
* * **Train-Val divided synthetic dataset**
* Copy the dataset folder (universal for all notebooks) from:
  [https://drive.google.com/drive/folders/1a9Tfdi\_X5vUuBynKksF3xt6o\_-IsoapF?usp=drive\_link](https://drive.google.com/drive/folders/1a9Tfdi_X5vUuBynKksF3xt6o_-IsoapF?usp=drive_link)
* Place its shortcut at:

  ```
  /Drive/MyDrive/data
  ```

---

### 1. Pretraining on synthetic dataset (10% validation hold-out)

* **Notebook**: `Pretraining-On-Synthetic-Dataset-with-10%-Validation-HoldOut.ipynb`
* **Model folder**: `VSLNetForPreTraining - With 10% Validation HoldOut`
* **Modifications**:

  * Only uses training + validation splits
  * Saves top-1 checkpoint

### 2. Pretraining on whole synthetic dataset

* **Notebook**: `Pretraining-On-Whole-Synthetic-Dataset.ipynb`
* **Model folder**: `VSLNetForPreTraining - Whole Synthetic Dataset`
* **Modifications**:

  * Uses entire synthetic dataset as training
  * Saves top-1 checkpoint

### 3. Transfer learning VSLNet on original Ego4D dataset

* **Notebook**: `Step2-TransferLearningOnOriginalEgo4D.ipynb`
* **Model folder**: `Pretrainable VSLNet`
* **Modifications**:

  * Expects a `.t7` file of pretrained weights

### 4. Transfer learning VSLNet on original Ego4D dataset (frozen feature encoder)

* **Notebook**: `Step2-TransferLearningOnOriginalEgo4D.ipynb`
* **Model folder**: `Pretrainable VSLNet - Frozen Feature Encoder`
* **Modifications**:

  * Freezes the feature-encoder
  * Expects a `.t7` file of pretrained weights


