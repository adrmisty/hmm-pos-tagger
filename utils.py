# hmm.py = hidden markov model functions
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

from pathlib import Path
from conllu import parse_incr
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import pickle
from hmm import HMM

def load_conllu(file_path):
    """
    Load a UD dataset in CoNLL-U format.
    
    Args:
        file_path (Path or str): Path to the .conllu file
    
    Returns:
        list of sentences, each a list of (word, pos_tag) tuples
    """
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sentence = [(token["form"], token["upostag"]) for token in tokenlist if token["upostag"]]
            if sentence:
                sentences.append(sentence)
    return sentences

def save_model(obj, path):
    """
    Save an object, e.g. a trained model, to a pickle file.
    
    Args:
        obj: Python object to save
        path (Path or str): output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    
def plot_confusion_matrix(test_data_tagged, predictions, model_name="HMM PoS Tagger"):
    """
    Calculates and plots a confusion matrix for PoS tagging results.

    This function first flattens the list of (word, tag) tuples from the 
    HMM output into single lists of gold tags and predicted tags, then 
    generates a confusion matrix heatmap.

    Args:
        test_data_tagged (list): The list of test sentences, where each sentence
                                 is a list of (word, gold_tag) tuples.
        predictions (list): The list of predicted sentences, where each sentence
                            is a list of (word, predicted_tag) tuples (from HMM.predict).
        model_name (str): The name of the model to display in the plot title.
    """
    
    gold_tags = []
    pred_tags = []
    
    for gold_sent, pred_sent in zip(test_data_tagged, predictions):
        # We assume the sentences have the same length due to the Viterbi design.
        for (_, gold_tag), (_, pred_tag) in zip(gold_sent, pred_sent):
            gold_tags.append(gold_tag)
            pred_tags.append(pred_tag)

    # Use the tags from the gold standard to define the class order
    tags = sorted(list(set(gold_tags)))

    cm = confusion_matrix(gold_tags, pred_tags, labels=tags)

    plt.figure(figsize=(12, 10)) # Increase size for better readability of tag labels
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',          # 'd' for integer counts
        cmap='Blues',
        xticklabels=tags, 
        yticklabels=tags,
        cbar_kws={'label': 'Number of Instances'}
    )
    
    # Labels and title
    plt.xlabel(f"Predicted Tag\nAccuracy: {cm.trace() / cm.sum():.4f}", fontsize=14)
    plt.ylabel("Actual (Gold) Tag", fontsize=14)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()

def load_model(path: Path) -> HMM:
    """Loads a trained HMM model from a file."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        sys.exit(1)

def save_predictions(predictions: list, path: Path):
    """
    Saves predictions (list of tagged sentences) to a plain text file.
    Format: word_1/tag_1 word_2/tag_2 ... [newline for each sentence]
    
    Args:
        predictions (list): A nested list where each inner list is a sentence 
                            of (word, tag) tuples.
        path (Path): The file path where predictions are saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for sentence in predictions:
            line = " ".join([f"{word}/{tag}" for word, tag in sentence])
            f.write(line + '\n')
    print(f"    > Predictions saved to: {path}")

def load_predictions(path: Path) -> list:
    """
    Loads predictions from a text file and returns them as a nested list of (word, tag) tuples.
    
    Args:
        path (Path): The file path from which predictions should be loaded.

    Returns:
        list: A nested list where each inner list is a sentence of (word, tag) tuples.
    """
    try:
        predictions = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tagged_sentence = []
                for item in line.split():
                    if '/' in item:
                        word, tag = item.rsplit('/', 1)
                        tagged_sentence.append((word, tag))
                predictions.append(tagged_sentence)
        print(f"    > Predictions loaded from: {path}")
        return predictions
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading predictions from {path}: {e}")
        return None
