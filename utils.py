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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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
    # Compute per-tag precision, recall, F1, and support
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_tags,
        pred_tags,
        labels=tags,
        zero_division=0
    )

    accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0

    print("\n=== Precision / Recall / F1-score per tag ===\n")
    print(classification_report(gold_tags, pred_tags, labels=tags, digits=4))

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

    # Show precision / recall / F1 in a separate figure
    # Build a small monospaced table as text
    lines = []
    lines.append(f"{'TAG':<6}{'P':>6}{'R':>6}{'F1':>6}{'N':>6}")
    lines.append("-" * 30)
    for tag, p, r, f, n in zip(tags, precision, recall, f1, support):
        lines.append(f"{tag:<6}{p:>6.2f}{r:>6.2f}{f:>6.2f}{int(n):>6}")

    # macro averages
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f = f1.mean()
    lines.append("-" * 30)
    lines.append(f"{'MACRO':<6}{macro_p:>6.2f}{macro_r:>6.2f}{macro_f:>6.2f}{'':>6}")

    text = "\n".join(lines)

    # create a new figure for the text
    plt.figure(figsize=(6, 8))
    plt.axis("off")
    plt.title(f"Precision / Recall / F1 per tag\n{model_name}", fontsize=12)
    plt.text(0.0, 1.0, text, family="monospace", va="top", fontsize=10)
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
    
def analyze_tag_errors(gold_data: list, predictions: list, target_tag: str, error_type: str, mispredicted_as: str = None):
    """
    Analyzes and prints sentences containing problematic instances of a target tag 
    based on recall (false negatives), precision (false positives), or a specific misprediction.

    Args:
        gold_data (list): Nested list of gold standard sentences [(word, gold_tag)].
        predictions (list): Nested list of predicted sentences [(word, pred_tag)].
        target_tag (str): The GOLD tag to analyze (e.g., 'NOUN', 'VERB').
        error_type (str): 'recall', 'precision', or 'specific'.
        mispredicted_as (str, optional): The predicted tag to filter by, used only 
                                         when error_type is 'specific'. Defaults to None.
    """
    
    # --- Logging Setup ---
    if error_type == 'specific':
        # Custom log output for specific error type
        log_type = f"Specific Error (GOLD: {target_tag} -> PRED: {mispredicted_as})"
    else:
        # Standard log output for recall/precision analysis
        log_type = f"{error_type.upper()} Failure ({target_tag})"
        
    print(f"\n--- Error Analysis for Tag: {target_tag} (Mode: {log_type}) ---")
    
    error_count = 0
    
    # We iterate over sentences
    for gold_sent, pred_sent in zip(gold_data, predictions):
        sentence_errors = []
        words = [word for word, _ in gold_sent]

        # We iterate over words and tags in the sentence
        for i, ((gold_word, gold_tag), (pred_word, pred_tag)) in enumerate(zip(gold_sent, pred_sent)):
            
            is_error = False
            highlight_reason = None
            
            if error_type == 'recall':
                # RECALL FAILURE (False Negative): Gold is TARGET_TAG, but Prediction is NOT TARGET_TAG.
                if gold_tag == target_tag and pred_tag != target_tag:
                    is_error = True
                    highlight_reason = f"GOLD: {target_tag} | PRED: {pred_tag}"
            
            elif error_type == 'precision':
                # PRECISION FAILURE (False Positive): Prediction is TARGET_TAG, but Gold is NOT TARGET_TAG.
                if pred_tag == target_tag and gold_tag != target_tag:
                    is_error = True
                    highlight_reason = f"PRED: {target_tag} | GOLD: {gold_tag}"
            
            elif error_type == 'specific':
                # SPECIFIC MISPREDICTION: Gold is TARGET_TAG, AND Prediction is MISPREDICTED_AS tag.
                if gold_tag == target_tag and pred_tag == mispredicted_as:
                    is_error = True
                    highlight_reason = f"GOLD: {target_tag} | PRED: {pred_tag}"
            
            if is_error:
                sentence_errors.append({
                    'index': i,
                    'word': gold_word,
                    'reason': highlight_reason
                })

        # If errors were found in the sentence, print the full context
        if sentence_errors:
            error_count += len(sentence_errors)
            
            # Print the entire sentence for context
            full_sentence = " ".join(words)
            print(f"\n[Error Count: {len(sentence_errors)} instances found in this sentence]")
            print(f"Sentence: {full_sentence}")
            
            # Print details for each error
            for error in sentence_errors:
                print(f"  > WORD: '{error['word']}' (Index: {error['index']}) - {error['reason']}")

    print(f"\n--- Total problematic instances found for {target_tag} ({log_type}): {error_count} ---")