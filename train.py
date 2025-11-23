# hmm.py = hidden markov model functions
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

import sys
from pathlib import Path
import argparse

# logic
from utils import load_conllu, save_model, plot_confusion_matrix, load_model
from hmm import HMM

def main(args):
    """
    Handles two modes:
    1) Train an HMM POS tagger and [optionally] evaluate on a test set.
    2) Load a pre-trained HMM POS tagger and evaluate on a test set.
    
    Args:
        args: command line arguments
            - training UD dataset
            - evaluation UD dataset (optional)
            - output trained model path
            - smoothing method (optional)
    """
    hmm = None
    test_data = None

    if args.test:
        print(f"** Test data: {args.test} **")
        test_data = load_conllu(args.test)

    is_training_mode = args.train is not None and args.load_model is None

    if is_training_mode:
        print(f"** Training data: {args.train} **")
        train_data = load_conllu(args.train)
        print(f"    > Data downloaded from (Universal Dependencies v2.17): https://lindat.mff.cuni.cz/repository/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0")

        print("\n ** Training HMM part-of-speech tagger **")
        hmm = HMM(smooth=args.smoothing) 
        hmm.train(train_data)


        print("\n ** Saving trained HMM part-of-speech tagger **")
        save_model(hmm, args.model)
        print(f"    > Saved to: {args.model}")
    elif args.load_model:
        print(f"** Loading pre-trained HMM model from: {args.load_model} **")
        # Calls the function imported from utils.py
        hmm = load_model(args.load_model)
    
    if hmm is None:
        print("Error: Model was not loaded or trained. Check input arguments.")
        sys.exit(1)
    
    if args.test and test_data is not None:
        test_words_nested = [[word for (word, tag) in sent] for sent in test_data]
        predictions_nested = hmm.predict(test_words_nested)
        print("\n ** Evaluating HMM part-of-speech tagger **")
        accuracy = hmm.evaluate(test_data)
        print(f"    > Tagging accuracy on test set: {accuracy * 100:.2f}%")
        
    if args.matrix:
            print("\n ** Plotting confusion matrix **")
            
            # Determine the title based on the mode
            model_title = args.load_model.name if args.load_model else f"New HMM (λ={args.smoothing})"
            
            plot_confusion_matrix(
                test_data_tagged=test_data, 
                predictions=predictions_nested, 
                model_name=f"HMM PoS Tagger - {model_title}"
            )
            print(" ** Confusion matrix plotted **")

    #print("\n\n** References **")
    #print("- Zeman, Daniel; et al., 2019, Universal Dependencies 2.5, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), http://hdl.handle.net/11234/1-3105.")
    #print("- Hajič, Jan, 2016, Markov Models, Institute of Formal and Applied Linguistics (ÚFAL) Charles University, https://ufal.mff.cuni.cz/~zabokrtsky/fel/slides/lect03-markov-models.pdf.")


    # --------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate an already trained HMM POS tagger")

    parser.add_argument(
        "--train", type=Path, required=False,
        help="Path to the UD training dataset (.conllu format)"
    )

    parser.add_argument(
        "--test", type=Path, default=None,
        help="Path to UD test dataset (.conllu format)"
    )

    parser.add_argument(
        "--load-model", type=Path, required=False,
        help="Load an already trained model for evaluation (.pkl format)"
    )

    parser.add_argument(
        "--model", type=Path, required=False, default="./results/models/hmm_model.pkl",
        help="Path to save the trained HMM POS tagger (.pkl format)"
    )

    parser.add_argument(
        "--smoothing", type=float, default=0.00,
        help="Smoothing method for unseen/rare data"
    )

    parser.add_argument(
        "--matrix", action="store_true",
        help="If True, plot confusion matrix after evaluation"
    )
    
    args = parser.parse_args()
    main(args)    
