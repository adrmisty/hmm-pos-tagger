# hmm.py = hidden markov model functions
# adriana r.f. (adrirflorez@gmail.com)
#
#
# nov-2026

from pathlib import Path
from collections import defaultdict

class HMM:
    """Hidden Markov Model (HMM) POS tagger class.
       
    Attributes:
        start_p (dict): start of a sequence probabilities of a tag, P(tag | <s>).
        transition_p(dict): transition probabilities of a tag given a previous tag, P(tag_i | tag_{i-1}).
        emission_p (dict): emission probabilities of a word given its tag, P(word | tag)
        ud_tags (set): set of unique UD tags seen during training.  
    """
    def __init__(self, unk_th : int = 1, smooth : float = None):
        
        # probabilities
        self.start_p = defaultdict(float)   
        self.transition_p = defaultdict(lambda: defaultdict(float))
        self.emission_p = defaultdict(lambda: defaultdict(float))  
        
        # data
        self.tags = set()
        self.vocab = set()
        
        # handling unseen data in training
        # (?) TODO should we impl smoothing? 
        self.unk_th = unk_th

    def train(self, training_data: list):
        """
        Trains the HMM model with training data using Maximum Likelihood Estimation (MLE).
        During training, initial/transition/emission probabilities are calculated.
        
        Args:
            training_data (list): a list of sentences from the UD datasset. 
                > each sentence is a list of (word, tag) tuples
        """
        
        print("> Training a Hidden-Markov-Model part-of-speech tagger...")
        print(f"\n  >> Training data: {len(training_data)} sentences.< from Universal Dependencies dataset")
        
        # 1. building vocabulary, counting words/tag occurrences
        words = self._build_vocab(training_data)
        print(f"    >> Vocabulary built with {len(self.vocab)} words (including <UNK>) for a UD tagset of {len(self.tags)}.")
        
        # 2.1 probabilities counting
        prob_counts = self._count_probs(training_data, words)
        print(f"    >> Tagset: Counted {len(self.tags)}.")
        
        # 2.2 probabilities estimation
        self._estimate_probs(*prob_counts)
        print("    >> HMM parameters estimated (start/transition/emission probabilities).")        


    def evaluate(self, test_data: list) -> float:
        """
        Evaluate the trained HMM model on a test dataset,
        according to HMM predictions based on the Viterbi algorithm
        with regards to gold standard tags.
        
        Args:
            test_data (list): a list of sentences from the UD dataset.
                > each sentence is a list of (word, tag) tuples
        
        Returns:
            accuracy (float): tagging accuracy on the test set
        """
        pass #TODO: notyetimplemented


    def predict(self, test_data: list) -> float:
        """
        Predicts tags for tokens of a list of untagged sentences using
        the Viterbi algorithm.

        Args:
            test_sentences (list): a list of sentences, 
                each a list of no-tag words

        Returns:
            list: a list of the same sentences, 
                each a list of (word, predicted_tag) tuples.
        """
        pass #TODO: notyetimplemented



    # --------------------------------------------------------------------------------------


    def _estimate_probs(self, start_c, transition_c, emission_c, tag_c, sentence_c):
        
        # total sizes of each set
        T = len(self.tags)
        V = len(self.vocab)
        N = sentence_c
        

        # TODO: smoothing [i am adding |T| and |V| to denominators but i guess i should be multiplying by some λ
        for tag in self.tags:
            # start of sequence: P(tag | *)
            # (times of tags seen at start of sentences) / (total sentences + |tagset|)
            self.start_p[tag] = start_c[tag] / (N + T)


        for prev_tag in self.tags:
            # transition: P(tag_i | tag_{i-1})
            # (times of transitions from prev_tag to tag) / (total times prev_tag seen + |tagset|)
            prev_transitions = tag_c[prev_tag]
            
            for tag in self.tags: # O(n2)
                self.transition_p[prev_tag][tag] = (transition_c[prev_tag][tag]) / (prev_transitions + T)

        for tag in self.tags:
            # emission: P(word | tag)
            # (times of word output by tag) / (total times tag seen + |vocab|)
            tag_emissions = tag_c[tag]
            for word in self.vocab:
                self.emission_p[tag][word] = (emission_c[tag][word]) / (tag_emissions + V)


    def _count_probs(self, training_data: list, words: tuple):
        
        N = len(training_data)
        _, unk_words = words
        
        # these counts are used to calculate the actual probs
        # for estimating the HMM params (diving by totals of tagset/vocab size)
        start_c = defaultdict(int)
        transition_c = defaultdict(lambda: defaultdict(int))
        emission_c = defaultdict(lambda: defaultdict(int))
        tag_c = defaultdict(int)

        for sentence in training_data:
            if not sentence:
                continue

            # start of sequence tag
            start_c[sentence[0][1]] += 1
            
            prev_tag = None
            for word, tag in sentence:
                # if unique, add to tagset
                self.tags.add(tag)
                tag_c[tag] += 1

                # rare word
                if word in unk_words:
                    word = "<UNK>"
                
                # emission (output) of a word given a tag
                emission_c[tag][word] += 1
                                
                # transition from prev tag to current it tag
                if prev_tag is not None:
                    transition_c[prev_tag][tag] += 1
                prev_tag = tag
        
        return start_c, transition_c, emission_c, tag_c, N


    def _build_vocab(self, training_data: list) -> tuple:

        # get word counts + identify <unk> words
        word_counts = defaultdict(int)
        for sentence in training_data:
            for word, _ in sentence:
                word_counts[word] += 1
        
        # vocabulary
        unk_words = set()
        for word, count in word_counts.items():
            if count <= self.unk_th:
                unk_words.add(word)
            else:
                self.vocab.add(word)
        self.vocab.add("<UNK>")
        
        return word_counts, unk_words

    def _viterbi(self, words: list) -> list:
        """
        Finds the most probable tag sequence using the Viterbi algorithm.

        Args:
            words (list): a list of word strings for ONE sentence.

        Returns:
            list: a list of predicted tag strings (same length as `words`).
        """
        if not words:
            return []

        # Replace unknown words with <UNK>
        obs = [w if w in self.vocab else "<UNK>" for w in words]
        T = len(obs)
        tags = list(self.tags)

        # Dynamic programming tables:
        # dp[t][tag] = best probability for any path ending in `tag` at position t
        # backpointer[t][tag] = previous tag that gave that best probability
        dp = []
        backpointer = []

        # ----- 1. Initialization (t = 0) -----
        dp0 = {}
        bp0 = {}
        w0 = obs[0]
        for tag in tags:
            start_prob = self.start_p[tag]              # P(tag | <s>)
            emit_prob = self.emission_p[tag].get(w0, 0) # P(w0 | tag)
            dp0[tag] = start_prob * emit_prob
            bp0[tag] = None
        dp.append(dp0)
        backpointer.append(bp0)

        # ----- 2. Recursion (t = 1 .. T-1) -----
        for t in range(1, T):
            dp_t = {}
            bp_t = {}
            w_t = obs[t]

            for tag in tags:
                emit_prob = self.emission_p[tag].get(w_t, 0)  # P(w_t | tag)
                best_prob = 0.0
                best_prev = None

                for prev_tag in tags:
                    prev_prob = dp[t - 1].get(prev_tag, 0.0)
                    trans_prob = self.transition_p[prev_tag].get(tag, 0.0)
                    prob = prev_prob * trans_prob * emit_prob

                    if best_prev is None or prob > best_prob:
                        best_prob = prob
                        best_prev = prev_tag

                dp_t[tag] = best_prob
                bp_t[tag] = best_prev

            dp.append(dp_t)
            backpointer.append(bp_t)

        # ----- 3. Termination: pick best final tag -----
        last_probs = dp[-1]
        # if everything is zero (pathological case), just pick an arbitrary tag
        best_last_tag = max(last_probs, key=last_probs.get)

        # ----- 4. Backtrack to recover best tag sequence -----
        best_tags = [None] * T
        best_tags[-1] = best_last_tag
        for t in range(T - 1, 0, -1):
            best_tags[t - 1] = backpointer[t][best_tags[t]]

        return best_tags
