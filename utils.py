# utils.py = utility and data functions
# Adriana R.F. (adrirflorez@gmail.com)
#
# nov-2026

from pathlib import Path
from conllu import parse_incr

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

def save_pickle(obj, path):
    """
    Save an object, e.g. a trained model, to a pickle file.
    
    Args:
        obj: Python object to save
        path (Path or str): output path
    """
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """
    Load a pickle object, e.g. a trained model, from file.
    
    Args:
        path (Path or str): path to the pickle file
    
    Returns:
        the loaded object
    """
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
