# 🧠 Hidden Markov Model PoS Tagger
## Computational Syntax team project

This project implements a **Hidden Markov Model (HMM) part-of-speech tagger** trained and evaluated on two datasets from the **Universal Dependencies (UD)** treebanks.  


## 📌 Project Overview

**#TODO(update README)** We develop an HMM-based POS tagger with:

- Estimation of transition probabilities (tag → tag) and emission probabilities (tag → word), trained on UD data
- Viterbi decoding for inference

---

## 📊 Experiments

We run experiments on **2 Universal Dependencies treebanks** (https://universaldependencies.org/) to observe and compare tagging accuracy of our HMM tagger.

---

## 📁 Project Structure

| File/Folder        | Description |
|--------------------|-------------|
| `hmm.py`           | Core logic for HMM training and Viterbi decoding |
| `utils.py`   | CoNLL-U dataset reading and preprocessing utilities |
| `train.py`         | Script to train and evaluate the HMM on a UD dataset |
| `models/`          | Saved trained HMM models (`.pkl`) |
| `data/`            | UD datasets used for training/testing |
| `README.md`        | Project description and usage guide |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train/Evaluate the HMM tagger on UD data

```bash
python train.py \
  --train ./data/{ud_folder}/{language_iso}-ud-train.conllu \
  --test ./data/{ud_folder}/{language_iso}-ud-test.conllu
  --model ./models/hmm.pkl
```

---


## 🔬 Methods

### Training

- Transition and emission probabilities are estimated from tag sequence and observed word-tag pairs in the training dataset.
- Unknown words are handled using `<UNK>` tokens.

### Decoding

- The **Viterbi algorithm** is used to find the most probable tag sequence for a sentence, via dynamic programming.

### Usage

Supported directly in CLI, the pipeline is as follows: 

```python
from hmm import HMM
from data_loader import load_conllu

# load Universal Dependencies data
train_sentences = load_conllu("./data/en_ewt-ud-train.conllu")
test_sentences = load_conllu("./data/en_ewt-ud-test.conllu")

# train HMM on UD data
hmm = HMM()
hmm.train(train_data)

# evaluate POS tagging accuracy
predictions = hmm.predict(test_sentences)
accuracy = hmm.evaluate(predictions, test_sentences)
print(f"Tagging accuracy: {accuracy:.2f}%")
```

---

## 👥 Authors

**Adriana Rodríguez Flórez**  
**Vera Senderowicz Guerra**  
**Emiel Vanderghinste Julien**

November 2026
EMLCT/HAP-LAP master's students  
University of the Basque Country

