# 🧠 Hidden Markov model PoS tagger
## Computational Syntax team project

## 1. Project overview

This project implements a **Hidden Markov model (HMM) part-of-speech tagger** trained and evaluated on three language datasets (English, Dutch and Greek) from the **Universal Dependencies (UD)** treebanks.  

It presents dedicated scripts for training, evaluation and post-processing to correct systematic tagging errors.

---

## 2. Experiments

We run experiments on **three Universal Dependencies treebank datasets** (https://universaldependencies.org/) to observe and compare tagging accuracy, precision and recall of our HMM tagger.
The tested languages of our choice have been: English, Dutch and Greek, as they are all languages that the authors speak and they also present varying degrees of word order freedom, morphology richness and two different alphabets.

The experiments have been ran and discussed on this Google Colab notebook, iteratively testing and improving the code based on our findings and hypotheses: https://colab.research.google.com/drive/1eHTZVZ-hBdAIMua51VrxBwzZwJITv9d7?usp=sharing

---

## 3. Project structure

| File/Folder | Description |
| :--- | :--- |
| `hmm.py` | HMM class with training and inference logic |
| `main.py` | Main pipeline for training and evaluation |
| `analyze.py` | Evaluation script for metric evaluation and error analysis |
| `postprocess.py` | Postprocessing script for applying heuristic rules |
| `heuristics.py` | Definitions of post-processing rules |
| `utils_io.py` | Data, model and results loading/saving |
| `utils_eval.py` | Metrics calculation and plotting |
| `data/` | UD datasets |
| `results/` | Output folder for models and predictions |


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

### b. Train/Evaluate a HMM on UD data

```bash
python main.py \
  --train ./data/en_ewt-ud-train.conllu \
  --test ./data/en_ewt-ud-test.conllu \
  --model ./results/models/en_hmm.pkl \
  --smoothing 0.01
```

### c. Predict/Evaluate with an existing model

```bash
python main.py \
  --load-model ./results/models/en_hmm.pkl \
  --test ./data/en_ewt-ud-test.conllu \
  --matrix
```

### d. Evaluate pre-computed predictions

```bash
python main.py \
  --load-predictions ./results/predictions.txt \
  --test ./data/en_ewt-ud-test.conllu
```

### e. Specific error analysis

```bash
python analyze.py \
  --load-model ./results/models/en_hmm.pkl \
  --test ./data/en_ewt-ud-test.conllu \
  --target-tag NOUN --error-type recall
```

### f. Post-process tagging results

```bash
python postprocess.py \
  --input-predictions ./results/predictions.txt \
  --output-dir ./results/post_processed/
```
---


## 5. Methods

### a. Training

- Transition and emission probabilities are estimated from tag sequence and observed word-tag pairs in the training dataset via **Maximum Likelihood Estimation** with Laplace smoothing.

- Unknown words are are mapped to pseudo-tokens based on typographic features (capitalization, digits...).

### b. Inference

- The **Viterbi algorithm** is used to find the most probable tag sequence for an untagged sentence, via dynamic programming, in the log probability space.

### Basic usage

Supported directly in CLI, the pipeline for training and evaluating an HMM tagger on training/testing data from Universal Dependencies is as follows: 

```python
from hmm import HMM
from utils_io import load_conllu
from utils_eval import compute_metrics

# 1. load necessary data
train_data = load_conllu("./data/en_ewt-ud-train.conllu")
test_data = load_conllu("./data/en_ewt-ud-test.conllu")

# 2. train the POS tagger
hmm = HMM(smooth=0.01)
hmm.train(train_data)

# 3. run inference
words = [[w for w, t in sent] for sent in test_data]
gold_tags = [t for sent in test_data for w, t in sent]

# 4. evaluate the predictions
preds = hmm.predict(words)
pred_tags = [t for sent in preds for w, t in sent]

metrics = compute_metrics(gold_tags, pred_tags)
print(f"Accuracy: {metrics['accuracy']:.2%}")
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
