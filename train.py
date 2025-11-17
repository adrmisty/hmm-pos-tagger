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
from utils import load_conllu, save_model
from hmm import HMM

def main(args):
    """
    Train an HMM POS tagger and [optionally] evaluate on a test set.
    
    Args:
        args: command line arguments
            - training UD dataset
            - evaluation UD dataset (optional)
            - output trained model path
            - smoothing method (optional)
    """
    print(f"** Training data: {args.train} **")
    train_sentences = load_conllu(args.train)

    print("\n ** Training HMM part-of-speech tagger **")
    hmm = HMM() 
    # TODO hmm = HMM(smooth=args.smooth)
    hmm.train(train_sentences)


    print("\n ** Saving trained HMM part-of-speech tagger **")
    save_model(hmm, args.model)
    print(f"    > Saved to: {args.model}")

    if args.test:
        # TODO impl evaluation
        pass


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

    """ TODO
    parser.add_argument(
        "--smoothing", type=str, default="none",
        choices=["none", "add-one"],
        help="Smoothing method for unknown words"
    )
    """
    
    args = parser.parse_args()
    main(args)