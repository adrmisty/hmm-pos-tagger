# train.py = training for pos tagger
# adriana r.f. (adrirflorez@gmail.com)
#
#
# nov-2026

import sys
from pathlib import Path
import argparse

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
    train_data = load_conllu(args.train)
    if args.test:
        print(f"** Test data: {args.test} **")
        test_data = load_conllu(args.test)
    print(f"    > Data downloaded from (Universal Dependencies v2.17): https://lindat.mff.cuni.cz/repository/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0")

    print("\n ** Training HMM part-of-speech tagger **")
    hmm = HMM() 
    # TODO: hmm = HMM(smooth=args.smooth)
    hmm.train(train_data)


    print("\n ** Saving trained HMM part-of-speech tagger **")
    save_model(hmm, args.model)
    print(f"    > Saved to: {args.model}")

    if args.test:
        # TODO: impl evaluation, using test.data()
        pass

    print("\n\n** References **")
    print("- Zeman, Daniel; et al., 2019, Universal Dependencies 2.5, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), http://hdl.handle.net/11234/1-3105.")
    print("- Hajič, Jan, 2016, Markov Models, Institute of Formal and Applied Linguistics (ÚFAL) Charles University, https://ufal.mff.cuni.cz/~zabokrtsky/fel/slides/lect03-markov-models.pdf.")


    # --------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HMM POS tagger")

    parser.add_argument(
        "--train", type=Path, required=True,
        help="Path to the UD training dataset (.conllu format)"
    )

    parser.add_argument(
        "--test", type=Path, default=None,
        help="Path to UD test dataset (.conllu format)"
    )

    parser.add_argument(
        "--model", type=Path, required=True, default="./results/models/hmm_model.pkl",
        help="Path to save the trained HMM POS tagger (.pkl format)"
    )

    """ TODO:
    parser.add_argument(
        "--smoothing", type=float, default="none",
        help="Smoothing method for unseen/rare data"
    )
    """
    
    args = parser.parse_args()
    main(args)
    