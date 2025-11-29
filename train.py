# hmm.py = hidden markov model functions
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

import sys
from pathlib import Path
import argparse

from utils_io import *
from utils_eval import plot_confusion_matrix, compute_metrics

from hmm import HMM

# redundantly reimplemented this here for simplicity
def _quick_eval(test_data, predictions, model_name, args):
    """
    Calculates accuracy using shared utils, handles saving, and plots matrix.
    """
    if args.save_predictions:
        save_predictions(predictions, args.save_predictions)

    gold_tags = [tag for sent in test_data for word, tag in sent]
    pred_tags = [tag for sent in predictions for word, tag in sent]

    metrics = compute_metrics(gold_tags, pred_tags)

    print(f"\n ** Evaluating HMM part-of-speech tagger ({model_name}) **")
    print(f"    > Tagging Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"    > Precision: {metrics['precision'] * 100:.2f}%")
    print(f"    > Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"    > F1-Score:  {metrics['f1'] * 100:.2f}%")

    if args.matrix:
        print("\n ** Plotting confusion matrix **")
        plot_confusion_matrix(
            test_data_tagged=test_data, 
            predictions=predictions, 
            model_name=f"{model_name} (F1: {metrics['f1']:.4f})"
        )
        print(" ** Confusion matrix plotted **")

def main(args):
    """
    HMM POS tagger - Pipeline
    --------------------------------------------------
    Authors: adriana r.f. (@adrmisty), vera s.g. (@verasenderowiczg), emiel v.j (@emielvanderghinste).
    Date:    nov-2026

    Description:
        The main.py script handles the full lifecycle of the Hidden Markov Model tagger.
        It operates in 3 mutually exclusive modes based on the arguments provided.

    EXECUTION MODES:

        1. TRAINING MODE (--train)
        > Full training process (*hmm.py): Builds vocab, calculates probabilities, and saves the model.
        > Can optionally run evaluation _immediately_ if --test provided.
        
        2. INFERENCE MODE (--load-model)
        > Fast evaluation: Loads a pre-trained .pkl model.
        (@veritavera, this is ur quick evaluation-without-training mode ;D)
        > Runs the Viterbi algorithm on the test set.

        3. ANALYSIS MODE (--load-predictions)
            [See eval.py for more fine-grained analysis]
        > Instant evaluation: skips already-executed Viterbi.
        > Loads a text file of pre-computed tags (word/TAG) and compares to Gold data.
        > For regenerating confusion matrices or metrics instantly.

    ARGUMENTS:
        --train <path>            Path to .conllu training data.
        --test <path>             Path to .conllu test data (required for accuracy/matrix).
        --model <path>            Output path to save the trained model (default: ./results/hmm.pkl).
        --load-model <path>       Input path to load a .pkl model file.
        --load-predictions <path> Input path to load a .txt file of tagged sentences.
        --save-predictions <path> Output path to write predicted tags to .txt.
        --smoothing <float>       Smoothing lambda value (default: 0.0).
        --matrix                  If set, displays the confusion matrix.

    USAGE EXAMPLES:
        # 1. infer predictions from a newly-trained model
        python main.py --train data/train.conllu --test data/test.conllu --smoothing 0.1

        # 2. infer predictions with a loaded model
        python main.py --load-model results/hmm.pkl --test data/test.conllu --matrix

        # 3. load predictions to (re)-analyze them
        python main.py --load-predictions results/preds.txt --test data/test.conllu --matrix
    """
    hmm = None
    test_data = None
    predictions_nested = None
    model_name = "unhinged_hmm"

    # 1. training mode
    # case 1.A: train new model
    if args.train:
        print(f"** Training data: {args.train} **")
        print(f"    > Source: Universal Dependencies v2.17")
        train_data = load_conllu(args.train)
        
        print("\n** Training HMM **")
        hmm = HMM(smooth=args.smoothing)
        hmm.train(train_data)
        
        print(f"\n** Saving model to {args.model} **")
        save_model(hmm, args.model)
        model_name = f"trained-{args.smoothing})"
    # case 1.B: load existing model
    elif args.load_model:
        print(f"** Loading model from: {args.load_model} **")
        hmm = load_model(args.load_model)
        model_name = args.load_model.name
    # finally: load test data if provided
    if args.test:
        print(f"** Test data: {args.test} **")
        test_data = load_conllu(args.test)
        if test_data is None:
            sys.exit("Error: Could not load test data.")

    # 2. inference mode
    # case 2.A: run inference on a model (whatever type, trained now/loaded from before
    if hmm and test_data:
        print("\n** Running HMM prediction **")
        test_words = [[word for (word, _) in sent] for sent in test_data]
        predictions_nested = hmm.predict(test_words)
        model_name = f"{model_name}-inference-model" # to identify in eval
        
    # case 2.B: predictions have already been inferred (in our beautiful unhinged Google Colab)
    elif args.load_predictions and test_data:
        print(f"\n** Loading pre-calculated predictions: {args.load_predictions} **")
        predictions_nested = load_predictions(args.load_predictions)
        model_name = f"Loaded predictions: {args.load_predictions.name}"
        if predictions_nested is None:
            sys.exit("Error: Could not load predictions file.")
    
    # (!!!!!) dramaaaaa
    # require test data but no predictions/model to proceed?
    elif args.test and not (hmm or args.load_predictions):
        sys.exit("Error: You must provide --train, --load-model, or --load-predictions to evaluate.")

    # finally: evaluate
    # only if we have predictions and test data
    # a more in-depth evaluation run at eval.py
    if predictions_nested and test_data:
        _quick_eval(test_data, predictions_nested, model_name, args)

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger Pipeline")

    # data args
    parser.add_argument("--train", type=Path, help="Path to UD training data (.conllu)")
    parser.add_argument("--test", type=Path, help="Path to UD test data (.conllu)")
    
    # model args
    parser.add_argument("--model", type=Path, default="./results/models/hmm_model.pkl", help="Path to SAVE trained model")
    parser.add_argument("--load-model", type=Path, help="Path to LOAD existing model")
    parser.add_argument("--smoothing", type=float, default=0.00, help="Smoothing lambda")

    # analysis/output args
    parser.add_argument("--matrix", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--save-predictions", type=Path, help="Save predictions to .txt")
    parser.add_argument("--load-predictions", type=Path, help="Load predictions from .txt (Analysis Mode)")

    args = parser.parse_args()
    
    # the bare minimum i swear to god
    if not (args.train or args.load_model or args.load_predictions):
        parser.error("(!) ERROR: No action specified. Use --train, --load-model, or --load-predictions.")
    main(args)

# --------------------------------------------------------------------------------------