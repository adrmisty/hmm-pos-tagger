from typing import List, Tuple

# Type alias for clarity
TaggedSentence = List[Tuple[str, str]]
# NOTE: Removed redundant Tuple wrapping from TaggedData
TaggedData = List[TaggedSentence] 

def apply_all_rules(predictions: TaggedData) -> TaggedData:
    """
    Applies all defined post-processing heuristics to the predicted data.
    
    Args:
        predictions (TaggedData): Nested list of (word, predicted_tag) tuples.

    Returns:
        TaggedData: The modified list of predictions.
    """
    
    corrected_predictions = []
    
    for sentence in predictions:
        # Apply sentence-level rules here. The order matters!
        
        # Rule 1: Promote PROPN/NOUN/ADJ sequences to PROPN based on context (Multi-word entity correction).
        # We will keep this rule commented out for now, focusing on the number promotion.
        # sentence = promote_compound_nouns(sentence)
        
        # Rule 2: Force PROPN/NOUN tags containing only numbers/symbols to NUM (Number correction). - THIS IS THE ONLY ACTIVE RULE
        sentence = promote_propn_and_noun_to_num(sentence)
        
        # Rule 3: Enforce capitalization consistency for NOUNs/PROPNs (Last resort lexical fix)
        # sentence = enforce_capitalization_rule(sentence)
        
        # Rule 4: Example Rule (If you want to keep the one I made earlier)
        # sentence = enforce_proper_noun_consistency(sentence)
        
        corrected_predictions.append(sentence)
        
    return corrected_predictions


def promote_compound_nouns(sentence: TaggedSentence) -> TaggedSentence:
    """
    Heuristic: Promotes a NOUN to PROPN if it is bordered by a PROPN.
    This is a stricter rule than the original.
    """
    new_sentence = []
    
    for i, (word, tag) in enumerate(sentence):
        new_tag = tag
        
        # Only check if the current tag is NOUN
        if tag == 'NOUN':
            
            # Check previous tag: must be PROPN
            prev_is_propn = sentence[i - 1][1] == 'PROPN' if i > 0 else False
            
            # Check next tag: must be PROPN
            next_is_propn = sentence[i + 1][1] == 'PROPN' if i < len(sentence) - 1 else False
            
            # Stricter Rule: If the NOUN is bordered by at least one PROPN, promote it.
            if prev_is_propn or next_is_propn:
                new_tag = 'PROPN'
        
        new_sentence.append((word, new_tag))
        
    return new_sentence


def promote_propn_and_noun_to_num(sentence: TaggedSentence) -> TaggedSentence:
    """
    Heuristic: Corrects instances where a word is purely numeric/symbolic 
    but was incorrectly tagged as PROPN or NOUN to the NUM tag.
    """
    new_sentence = []
    
    for word, tag in sentence:
        new_tag = tag
        
        # Check if the tag is one of the target tags (PROPN or NOUN)
        if tag != 'NUM':
        #in {'PROPN', 'NOUN'}:
            
            # Check if the word contains at least one digit AND no alphabetic characters.
            if any(char.isdigit() for char in word) and not any(char.isalpha() for char in word):
                new_tag = 'NUM'
            
        new_sentence.append((word, new_tag))
        
    return new_sentence


def enforce_capitalization_rule(sentence: TaggedSentence) -> TaggedSentence:
    """
    Heuristic: Corrects NOUN tags to PROPN when the word is capitalized (not at sentence start).
    
    Rules:
    1. If word starts with uppercase AND is NOUN -> change to PROPN.
    2. (Former Rule 2 removed, no longer downgrading PROPN to NOUN.)
    """
    new_sentence = []
    
    for i, (word, tag) in enumerate(sentence):
        new_tag = tag
        
        # We must skip the first word of the sentence, as it is always capitalized.
        # This prevents common nouns like "The" or "A" from becoming PROPNs.
        if i == 0:
            new_sentence.append((word, tag))
            continue
            
        # 1. NOUN to PROPN correction (If tagged NOUN but starts with uppercase)
        # Check if word starts with a capital and isn't a common initialism/digit
        if tag == 'NOUN' and word and word[0].isupper() and not word[0].isdigit():
            new_tag = 'PROPN'
            
        # The 'elif' block for PROPN-to-NOUN correction is REMOVED as requested.
        
        new_sentence.append((word, new_tag))
        
    return new_sentence


# --- Example Rule (Optional) ---
# You can keep this function if you want to use it later, or remove it.
def enforce_proper_noun_consistency(sentence: TaggedSentence) -> TaggedSentence:
    """
    Simple heuristic: Corrects PROPNs tagged as NOUNs when surrounded by PROPNs.
    """
    new_sentence = []
    for i, (word, tag) in enumerate(sentence):
        new_tag = tag
        if tag == 'NOUN' and i > 0 and i < len(sentence) - 1:
            prev_tag = sentence[i - 1][1]
            next_tag = sentence[i + 1][1]
            if prev_tag == 'PROPN' and next_tag == 'PROPN':
                new_tag = 'PROPN'
        new_sentence.append((word, new_tag if new_tag != tag else tag))
    return new_sentence