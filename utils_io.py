# utils_io.py = loading and saving stuff (conllu, pickle, preds)
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Any, Optional
from conllu import parse_incr


def load_conllu(file_path: Path) -> List[List[Tuple[str, str]]]:
    """Parses a CoNLL-U file and extracts word/tag pairs."""
    
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                sentence = [(token["form"], token["upostag"]) for token in tokenlist if token["upostag"]]
                if sentence:
                    sentences.append(sentence)
        return sentences
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        sys.exit(1)


def save_model(model_obj: Any, path: Path):
    """Saves a Python object (HMM model) to a pickle file."""
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"    > Model saved to: {path}")

def load_model(path: Path) -> Any:
    """Loads a pickled object (HMM model) from a file."""
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def save_predictions(predictions: List[List[Tuple[str, str]]], path: Path):
    """Saves tagged sentences to a text file in 'word/TAG' format."""
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for sentence in predictions:
            line = " ".join([f"{word}/{tag}" for word, tag in sentence])
            f.write(line + '\n')
    print(f"    > Predictions saved to: {path}")

def load_predictions(path: Path) -> Optional[List[List[Tuple[str, str]]]]:
    """Loads predictions from a text file (word/TAG format)."""
    
    try:
        predictions = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
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