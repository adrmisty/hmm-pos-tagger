# train.py = training for pos tagger
# adriana r.f. (adrirflorez@gmail.com)
#
#
# nov-2026

import sys
from pathlib import Path
import argparse
import pickle

# redirection of stdout/stderr to a log file
log_file = Path("./results/log.txt")
log_file.parent.mkdir(parents=True, exist_ok=True)
sys.stdout = open(log_file, "w", encoding="utf-8")
sys.stderr = sys.stdout

# logic
from utils import load_conllu
from hmm.hmm import HMM

def main(args):
    """
    Train an HMM POS tagger and [optionally] evaluate on a test set.
    
    Args:
        args: command line arguments
            - training UD dataset
            - evaluation UD dataset (optional)
            - output trained model path
            - smoothing method (optional)
            - verbosity flag (optional)
    """
    print(f"> Loading training data from {args.train}")
    train_sentences = load_conllu(args.train)

    print(f"> Initializing HMM (smoothing='{args.smoothing})'")
    hmm = HMM(args.smoothing)

    print("> Training HMM...")
    hmm.train(train_sentences, verbose=args.verbose)

    print("> Saving trained model...")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model, "wb") as f:
        pickle.dump(hmm, f)
    print(f"Trained model saved to {args.model}")

    if args.test:
        print(f"> Evaluating on test data from {args.test}")
        test_sentences = load_conllu(args.test)
        predictions = hmm.predict(test_sentences)
        accuracy = hmm.evaluate(predictions, test_sentences)
        print(f"Tagging Accuracy: {accuracy:.2f}%")

# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an HMM POS tagger")

    parser.add_argument(
        "--train", type=Path, required=True,
        help="Path to the UD training dataset (.conllu format)"
    )

    parser.add_argument(
        "--test", type=Path, default=None,
        help="Path to UD test dataset (.conllu format)"
    )

    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to save the trained HMM model (.pkl format)"
    )

    parser.add_argument(
        "--smoothing", type=str, default="none",
        choices=["none", "add-one"],
        help="Smoothing method for unknown words"
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress and training info"
    )

    args = parser.parse_args()
    main(args)
