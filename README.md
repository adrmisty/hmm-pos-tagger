# 🧠 Hidden Markov model PoS tagger
## Computational Syntax team project

This project implements a **Hidden Markov model (HMM) part-of-speech tagger** trained and evaluated on three language datasets (English, Dutch and Greek) from the **Universal Dependencies (UD)** treebanks.  

---

## TODOS

#TODOs-Repo:
- [] once Colab/code effectively finished, update the README.md accordingly and review the whole project to add/review documentation and clean code practices

#TODOs-Code:
- [] impl. recognition assistance for PNOUN vs. NOUN: possibly there are 'more unseen'/less freq. instances of a word based on its capitalization, which might fall under the UNK threshold, so an idea is to create an extra rare-word class <UNK_CAP> that accounts for rare, capitalized words

#TODOs-Colab:
- [] add more reflection/thinking process/explanations of why results are the way they are, in between CM discussions
- [] test metrics/plot new CMs based on the prev. PNOUN fix

---

## 1. Project overview

We develop an HMM-based POS tagger with:

- Estimation of transition probabilities (tag → tag) and emission probabilities (tag → word), trained on UD data
- Viterbi decoding for inference

---

## 2. Experiments

We run experiments on **three Universal Dependencies treebank datasets** (https://universaldependencies.org/) to observe and compare tagging accuracy, precision and recall of our HMM tagger.
The tested languages of our choice have been: English, Dutch and Greek, as they are all languages that the authors speak and they also present varying degrees of word order freedom, morphology richness and two different alphabets.

The experiments have been ran and discussed on this Google Colab notebook, iteratively testing and improving the code based on our findings and hypotheses: https://colab.research.google.com/drive/1eHTZVZ-hBdAIMua51VrxBwzZwJITv9d7?usp=sharing

---

## 3. Project structure

| File/Folder        | Description |
|--------------------|-------------|
| `hmm.py`           | Core logic for HMM training and Viterbi decoding |
| `utils.py`   | CoNLL-U dataset reading and preprocessing utilities |
| `train.py`         | Script to train and evaluate the HMM on a UD dataset |
| `models/`          | Saved trained HMM models (`.pkl`) |
| `data/`            | UD datasets used for training/testing |
| `README.md`        | Project description and usage guide |

---

## 4. How to run

### a. Install dependencies

```bash
pip install -r requirements.txt
```

### b. Train/Evaluate the HMM tagger on UD data

```bash
python train.py \
  --train ./data/{UD_Language_dir}/{iso2_lang}-ud-train.conllu \
  --test ./data/{UD_Language_dir}/{iso2_lang}-ud-test.conllu
  --model ./results/models/hmm.pkl
```

---


## 5. Methods

### a. Training

- Transition and emission probabilities are estimated from tag sequence and observed word-tag pairs in the training dataset via **Maximum Likelihood Estimation**.

- Unknown words are handled using `<UNK>` tokens for words with frequency under a certain threshold.

### b. Inference

- The **Viterbi algorithm** is used to find the most probable tag sequence for an untagged sentence, via dynamic programming.

### Basic usage

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

## 6. References

- Zeman, Daniel; et al., 2019, 
  Universal Dependencies 2.5, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), 
  http://hdl.handle.net/11234/1-3105.

- Hajič, Jan, 2016, Markov Models, Institute of Formal and Applied Linguistics (ÚFAL) Charles University, https://ufal.mff.cuni.cz/~zabokrtsky/fel/slides/lect03-markov-models.pdf.


---

## 7. Authors

**Adriana Rodríguez Flórez**  
**Vera Senderowicz Guerra**  
**Emiel Vanderghinste Julien**

November 2025
EMLCT/HAP-LAP master's students  
University of the Basque Country
