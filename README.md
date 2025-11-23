# 🧠 Hidden Markov Model PoS Tagger
## Computational Syntax team project

This project implements a **Hidden Markov Model (HMM) part-of-speech tagger** trained and evaluated on two datasets from the **Universal Dependencies (UD)** treebanks.  

---

## TODOS

For the code:
- implement dumping of tagging prediction for test data into text file-DONE
- evaluate other metrics apart from accuracy (precision, recall, f1)
- analyze how to improve accuracy
  --> smoothing?
  --> other techniques?

#TODOs-Colab:
from observing the CM, analyze what can be added to help each lang's model improve tagging of most problematic classes


## 📌 Project Overview

**We develop an HMM-based POS tagger with:

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
  --train ./data/{UD_Language_dir}/{iso2_lang}-ud-train.conllu \
  --test ./data/{UD_Language_dir}/{iso2_lang}-ud-test.conllu
  --model ./results/models/hmm.pkl
```

---


## 🔬 Methods

### Training

- Transition and emission probabilities are estimated from tag sequence and observed word-tag pairs in the training dataset via **Maximum Likelihood Estimation**.

- Unknown words are handled using `<UNK>` tokens for words with frequency under a certain threshold.

### Decoding

- The **Viterbi algorithm** is used to find the most probable tag sequence for an untagged sentence, via dynamic programming.

### Usage

Supported directly in CLI, the pipeline for training and evaluating an HMM tagger on training/testing data from Universal Dependencies is as follows: 

```python
from hmm import HMM
from utils import load_conllu, save_models

# load Universal Dependencies data
train_data = load_conllu("./data/UD_Basque-BDT/eu_bdt-ud-train.conllu")
test_data = load_conllu("./data/UD_Basque-BDT/eu_bdt-ud-train.conllu")
model_path = "./results/models/hmm-pos-tagger.pkl"

# train HMM on UD data
hmm = HMM()
hmm.train(train_data)

# (optionally but recommended) persist trained model
save_model(model_path, hmm)

# evaluate POS tagging prediction accuracy
accuracy = hmm.evaluate(test_data)
print(f"Tagging accuracy: {accuracy:.2f}%")
```

---

## 📖 References

- Zeman, Daniel; et al., 2019, 
  Universal Dependencies 2.5, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), 
  http://hdl.handle.net/11234/1-3105.

- Hajič, Jan, 2016, Markov Models, Institute of Formal and Applied Linguistics (ÚFAL) Charles University, https://ufal.mff.cuni.cz/~zabokrtsky/fel/slides/lect03-markov-models.pdf.


---

## 👥 Authors

**Adriana Rodríguez Flórez**  
**Vera Senderowicz Guerra**  
**Emiel Vanderghinste Julien**

November 2026
EMLCT/HAP-LAP master's students  
University of the Basque Country

