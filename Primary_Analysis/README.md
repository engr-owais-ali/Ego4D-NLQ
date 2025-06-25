

## Ego4D NLQ

**Prerequisites**

* Download the dataset folder (universal for all notebooks) from
  [https://drive.google.com/drive/folders/1a9Tfdi\_X5vUuBynKksF3xt6o\_-IsoapF?usp=drive\_link](https://drive.google.com/drive/folders/1a9Tfdi_X5vUuBynKksF3xt6o_-IsoapF?usp=drive_link)
* Place the Drive shortcut at:

  ```
  /Drive/MyDrive/data
  ```

---

### 1. VSLNet on Omnivor Features

* **Model**: Pulled from the original GitHub repository
* **Features**: Loaded from Google Drive

### 2. VSLNet on EgoVLP Features

* **Model**: Pulled from the original GitHub repository
* **Features**: Preprocessed EgoVLP features fetched from our Google Cloud bucket
* **Note**: No dataset preparation required

### 3. VSLBase on Omnivor Features

* **Model**: Fetches VSLNet model + preprocessed Omnivor features from Google Cloud bucket
* **Setup**:

  1. Remove the `VSLNet` folder under **Episodic Memory**
  2. Replace it with the `VSLBase` directory

### 4. VSLBase on EgoVLP Features

* **Model**: Fetches VSLNet model + preprocessed EgoVLP features from Google Cloud bucket
* **Setup**:

  1. Remove the `VSLNet` folder under **Episodic Memory**
  2. Replace it with the `VSLBase` directory

### 5. VSLNet on EgoVLP Features + GloVe Embeddings

* **Model**: Pulled from the original GitHub repository
* **Features & Embeddings**:

  * EgoVLP features from Google Drive
  * GloVe embeddings from Google Drive


