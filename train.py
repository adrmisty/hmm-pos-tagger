# hmm.py = hidden markov model functions
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

import sys
from pathlib import Path
import argparse

# logic
from utils import load_conllu, save_model, plot_confusion_matrix, load_model, load_predictions, save_predictions
from hmm import HMM

def main(args):
    """
    Handles the execution of the HMM Part-of-Speech Tagger, supporting three
    modes: Training, loading model, and analysis of pre-calculated results.
    
    1) Training Mode:
       Requires --train. Trains a new HMM and saves it (--model). Can be used with --test and --matrix to evaluate and plot.
    
    2) Model Evaluation Mode:
       Requires --load-model and --test. Loads a pre-trained HMM, runs the algorithm on the test data, and reports accuracy. Can be used with --matrix to plot confusion matrix.
       
    3) Analysis Mode:
       Requires --load-predictions and --test. Skips the prediction step, loads pre-calculated predictions from a file, and reports accuracy. Can be used with --matrix to plot confusion matrix.
    
    Args:
        args: command line arguments from argparse:
            --train: Path to the UD training dataset (required for training).
            --test: Path to the UD test dataset (required for evaluation/plotting).
            --load-model: Path to load a pre-trained HMM for evaluation.
            --model: Path to save the trained HMM.
            --smoothing: Smoothing factor (lambda) used during training.
            --matrix: Flag to plot the confusion matrix.
            --save-predictions: Path to save the evaluation output to a text file.
            --load-predictions: Path to load predictions from a file (Analysis Mode).
    """
    hmm = None
    test_data = None
    predictions_nested = None

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
    
    if hmm is None and args.load_predictions is None:
        print("Error: Model was not loaded or trained, and no predictions file was provided.")
        sys.exit(1)

    if args.test and test_data is None:
        print("Error: Test data required for evaluation but could not be loaded.")
        sys.exit(1)
    
    if args.load_predictions:
        # Load predictions from file
        print(f"\n ** Loading pre-calculated predictions **")
        predictions_nested = load_predictions(args.load_predictions)
        if predictions_nested is None:
             print("Error: Could not load predictions file.")
             sys.exit(1)

    elif args.test and hmm is not None:
        print("\n ** Running HMM prediction **")
        test_words_nested = [[word for (word, tag) in sent] for sent in test_data]
        predictions_nested = hmm.predict(test_words_nested)
        
        # Save predictions if specified
        if args.save_predictions:
            save_predictions(predictions_nested, args.save_predictions)
        
    if predictions_nested is not None and args.test and test_data is not None:

        print("\n ** Evaluating HMM part-of-speech tagger **")
        
        # Calculate accuracy manually from the loaded/predicted data 
        # (Since we might have loaded predictions without the HMM object)
        total = 0
        correct = 0
        for gold_sent, pred_sent in zip(test_data, predictions_nested):
            for (_, gold_tag), (_, pred_tag) in zip(gold_sent, pred_sent):
                total += 1
                if gold_tag == pred_tag:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"    > Tagging accuracy on test set: {accuracy * 100:.2f}%")
        
        if args.matrix:
            print("\n ** Plotting confusion matrix **")
            
            # Determine the title based on the mode
            if args.load_model:
                model_title = args.load_model.name
            elif args.load_predictions:
                 # Title reflects the loaded prediction file
                 model_title = f"Loaded Predictions: {args.load_predictions.name}"
            else:
                 model_title = f"New HMM (λ={args.smoothing})"
            
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

    parser.add_argument(
        "--save-predictions", type=Path, default=None,
        help="Path to save the predicted tags to a text file."
    )

    parser.add_argument(
        "--load-predictions", type=Path, default=None,
        help="Path to load pre-calculated predicted tags from a text file. \nRequires --test data to verify against gold tags."
    )
    
    args = parser.parse_args()
    main(args)    
