# heuristics.py = heuristic rules to fix mistags
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

# new types
from typing import List, Tuple
TaggedSentence = List[Tuple[str, str]]
TaggedData = List[TaggedSentence]

# ----> uncomment rules as needed <----
# also, if u are going to change the rules to-be-executed
# and are going to run more than one, think of the proper ordering for them
# e.g. run the capitalization PNOUN rule before the multi-word entity one
# bc they kinda deal with the same but one makes the second not-so-necessary
 
def apply_heuristics(predictions: TaggedData) -> TaggedData:
    """
    Applies a sequence of heuristic rules to correct HMM predictions.
    [Basis: mistags in PNOUN vs NOUN & other problematic POS tags].
    
    Args:
        predictions (TaggedData): nested list of (word, predicted_tag) tuples.

    Returns:
        TaggedData: the modified list of predictions.
    """
    corrected_predictions = []
    
    for sentence in predictions:
        # mistag: <NUM> --> correcting numbers
        # > forces words with digits (- no digits) to NUM
        sentence = _numerals(sentence)
        
        # mistag: <NOUN> vs <PNOUN> --> capitalization mishaps
        # > more likely to be a proper noun if capitalized and not at sent start
        # sentence = _proper_nouns(sentence)
        
        # mistag: multi-word noun entities --> context based
        # > forces adjacent nouns to pnouns if next to one
        # sentence = _multiword_entities(sentence)

        corrected_predictions.append(sentence)
        
    return corrected_predictions


# --------------------------------------------------------------------------------------



def _numerals(sentence: TaggedSentence) -> TaggedSentence:
    """
    Heuristic: Corrects instances where a word is purely numeric/symbolic 
    but was incorrectly tagged as PROPN or NOUN.
    """
    new_sentence = []
    
    for word, tag in sentence:
        new_tag = tag
        
        if tag != 'NUM':
            # digits but no letters whatsoever
            if any(char.isdigit() for char in word) and not any(char.isalpha() for char in word):
                new_tag = 'NUM'
            
        new_sentence.append((word, new_tag))
        
    return new_sentence


def _proper_nouns(sentence: TaggedSentence) -> TaggedSentence:
    """
    Heuristic: Promotes NOUN -> PROPN if the word is capitalized 
    and is NOT at the start of the sentence.
    """
    new_sentence = []
    
    for i, (word, tag) in enumerate(sentence):
        new_tag = tag
        
        # not sentence initial!!!!
        if i == 0:
            new_sentence.append((word, tag))
            continue
            
        # hopefully it recognizes 'Gay Parade' as a PROPN PROPN
        if tag == 'NOUN' and word and word[0].isupper() and not word[0].isdigit():
            new_tag = 'PROPN'
            
        new_sentence.append((word, new_tag))
        
    return new_sentence


def _multiword_entities(sentence: TaggedSentence) -> TaggedSentence:
    """
    Heuristic: Promotes a NOUN to PROPN if it is bordered by a PROPN.
    Extended this for >2 length.
    Example: "New (PROPN) York (NOUN)" -> "New (PROPN) York (PROPN)"
    """
    new_sentence = []
    
    
    for i, (word, tag) in enumerate(sentence):
        new_tag = tag
        
        if tag == 'NOUN':
            prev_is_propn = (sentence[i - 1][1] == 'PROPN') if i > 0 else False
            next_is_propn = (sentence[i + 1][1] == 'PROPN') if i < len(sentence) - 1 else False
            
            # If surrounded or attached to a PROPN, assume it is part of the entity
            if prev_is_propn or next_is_propn:
                new_tag = 'PROPN'
        
        new_sentence.append((word, new_tag))
            
    return new_sentence
