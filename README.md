<a id="readme-top"></a>
# About The Project
This repository implements a **multi-class Bangla hate speech detection** pipeline that classifies Bengali text into five hate speech categories: **Geopolitical**, **Personal**, **Political**, **Religious**, and **Gender Abusive**. The project compares eight deep learning architectures — six GloVe-based recurrent models (**RNN**, **Bidirectional RNN**, **LSTM**, **Bidirectional LSTM**, **GRU**, **Bidirectional GRU**) and two fine-tuned transformer models (**BanglaBERT Base** and **BanglaBERT Large**) — to identify the most effective approach for hate speech classification in Bangla.This project was developed as part of the CSE427(Machine Learning) course project at BRAC University.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Built With
* ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
* ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
* ![NLTK](https://img.shields.io/badge/NLTK-154F3C?style=for-the-badge&logo=python&logoColor=white)
* ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
* ![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-333333?style=for-the-badge&logo=matplotlib&logoColor=11557c)
* ![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=for-the-badge&logo=seaborn&logoColor=white)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Dataset Description

* **Source:** Bangla hate speech dataset (3,418 labeled samples) loaded via [Google Drive](https://drive.google.com/file/d/1772iZIXs6mhW6Ywuv4GYIFGjIbdLPJE8/view?usp=sharing)
* **Format:** Tab-separated `text` and `label` columns
* **Task:** Multi-class classification (5 classes)
* **Class Distribution (Imbalanced):**

  | Label | Count | Percentage |
  |:---|:---:|:---:|
  | Geopolitical | 1,379 | 40.35% |
  | Personal | 629 | 18.40% |
  | Political | 592 | 17.32% |
  | Religious | 502 | 14.69% |
  | Gender Abusive | 316 | 9.25% |
<p align="right">(<a href="#readme-top">back to top</a>)</p>
# Preprocessing Pipeline

| Step | Details |
| :--- | :--- |
| **Unicode Normalization** | NFC normalization on all text |
| **Whitespace Cleaning** | Multiple spaces collapsed, leading/trailing stripped |
| **Noise Removal** | URLs, numbers, emojis, and non-Bengali characters removed |
| **Punctuation Removal** | Bengali (`।,॥—` etc.) and English punctuation removed |
| **Tokenization** | Whitespace-based splitting |
| **Stopword Removal** | Bengali stopwords list from [stopwords-iso/stopwords-bn](https://github.com/stopwords-iso/stopwords-bn) |
| **Sequence Encoding** | Keras `Tokenizer` → integer sequences, `pad_sequences` with `maxlen=50`, post-padding |
| **Label Encoding** | `LabelEncoder` (5 classes → 0–4) |
| **Train-Test Split** | 80% / 20%, `random_state=42`, stratified on label |
| **Class Imbalance Handling** | `RandomOverSampler` on training set (balanced to 1,103 samples per class) |
<p align="right">(<a href="#readme-top">back to top</a>)</p>


# Models & Specifications

### GloVe-Based Recurrent Models (×6)

All six share this architecture template — only the recurrent layer changes:

| Component | Details |
| :--- | :--- |
| **Embedding** | Bengali GloVe 300d ([sagorsarker/bangla-glove-vectors](https://huggingface.co/sagorsarker/bangla-glove-vectors)), frozen (`trainable=False`) |
| **Recurrent Layer** | 128 hidden units, `tanh` activation |
| **Dropout** | 0.2 |
| **Output** | Dense(5, softmax) |
| **Optimizer** | Adam (`lr=0.001`) |
| **Loss** | `sparse_categorical_crossentropy` |
| **Training** | 20 epochs, batch size 32, `validation_split=0.1` |

| # | Model Variant |
|:---:|:---|
| 1 | SimpleRNN |
| 2 | Bidirectional SimpleRNN |
| 3 | LSTM |
| 4 | Bidirectional LSTM |
| 5 | GRU |
| 6 | Bidirectional GRU |

### BanglaBERT Transformer Models (×2)

| Component | BanglaBERT Base | BanglaBERT Large |
| :--- | :--- | :--- |
| **Pretrained Model** | `csebuetnlp/banglabert` | `csebuetnlp/banglabert_large` |
| **Tokenizer** | AutoTokenizer (pretrained) | AutoTokenizer (pretrained) |
| **Fine-tuning LR** | 2e-5 | 1e-5 |
| **Epochs** | 3 | 3 |
| **Batch Size** | 16 | 8 |
| **Loss** | SparseCategoricalCrossentropy (from_logits) | SparseCategoricalCrossentropy (from_logits) |
<p align="right">(<a href="#readme-top">back to top</a>)</p>



# Model Performance & Comparison

All metrics use **macro** averaging. Evaluated on the 20% held-out test set (684 samples).

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| RNN | 0.3933 | 0.3157 | 0.3397 | 0.3068 |
| Bidirectional RNN | 0.5789 | 0.5036 | 0.5045 | 0.5006 |
| LSTM | 0.6287 | 0.4946 | 0.5031 | 0.4415 |
| Bidirectional LSTM | 0.6564 | 0.5936 | 0.5872 | 0.5853 |
| GRU | 0.6784 | 0.6084 | 0.5945 | 0.6000 |
| Bidirectional GRU | 0.6740 | 0.6029 | 0.5972 | 0.5990 |
| **BanglaBERT Base** | **0.8173** | **0.7671** | **0.7808** | **0.7721** |
| **BanglaBERT Large** | **0.8216** | **0.7702** | **0.7910** | **0.7778** |

### Evaluation Includes
* Accuracy bar chart comparison across all 8 models
* Precision, Recall & F1 grouped bar chart
* Confusion matrix heatmaps for all 8 models
<p align="right">(<a href="#readme-top">back to top</a>)</p>



# How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/cse427.git
   cd cse427
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy==1.23.5 scikit-learn==0.10.1 imbalanced-learn==0.10.1
   pip install tensorflow pandas matplotlib seaborn nltk gdown transformers sentencepiece
   ```
   (A CUDA-enabled GPU is strongly recommended for BERT fine-tuning).

3. **Run Notebook**
   ```bash
   jupyter notebook code.ipynb
   ```
   The dataset and GloVe embeddings are automatically downloaded when the notebook runs.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



# Project Demo Video
The project demo video is available to watch here: <a href="https://youtu.be/q8EG_TDDddA?si=9QYti-qCqOcdF4Wu">click here</a>.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


# Contributors
* Sajjad Hossain Pappu (sajjad.hasan.pappu@g.bracu.ac.bd)
* YASIR (yasir@g.bracu.ac.bd)
<p align="right">(<a href="#readme-top">back to top</a>)</p>
