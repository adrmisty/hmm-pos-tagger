# heuristics.py = evaluation of pos tagging results
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2025

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from utils_io import load_conllu, load_model, load_predictions
from utils_eval import plot_confusion_matrix, analyze_tag_errors, compute_metrics

def _get_predictions(args, test_data):
    """
    Determines the source of predictions (File vs Model) and retrieves them.
    Returns: (predictions_nested, source_name)
    """
    if args.load_predictions:
        print(f"\n ** Loading pre-calculated predictions **")
        preds = load_predictions(args.load_predictions)
        if preds is None:
            sys.exit("Error: Could not load predictions file.")
        return preds, f"File: {args.load_predictions.name}"

    elif args.load_model:
        print(f"** Loading pre-trained HMM model from: {args.load_model} **")
        hmm = load_model(args.load_model)
        
        print("\n ** Running HMM prediction (Viterbi algorithm) **")
        # Extract just the words for prediction
        test_words_nested = [[word for (word, _) in sent] for sent in test_data]
        preds = hmm.predict(test_words_nested)
        return preds, f"Model: {args.load_model.name}"

    else:
        sys.exit("Error: Must provide either --load-model or --load-predictions.")


def _display_metrics(metrics):
    """Prints formatted evaluation metrics to stdout."""
    print("\n ** Evaluating HMM part-of-speech tagger **")
    print(f"    > Tagging Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"    > Precision: {metrics['precision'] * 100:.2f}%")
    print(f"    > Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"    > F1-Score:  {metrics['f1'] * 100:.2f}%")


def _visualize(args, test_data, predictions, f1_score, source_name):
    """Conditional logic for plotting the confusion matrix."""
    if args.matrix:
        print("\n ** Plotting confusion matrix **")
        
        # Construct a descriptive title
        plot_title = f"{source_name} (F1: {f1_score:.4f})"
        
        plot_confusion_matrix(
            test_data_tagged=test_data, 
            predictions=predictions, 
            model_name=plot_title
        )
        plt.show()


def _specific_error_analysis(args, test_data, predictions):
    """Conditional logic for specific tag error analysis."""
    if args.target_tag:
        analyze_tag_errors(
            gold_data=test_data, 
            predictions=predictions, 
            target_tag=args.target_tag, 
            error_type=args.error_type,
            mispredicted_as=args.mispredicted_as
        )
# --------------------------------------------------------------------------------------
def main(args):
    """
    HMM POS tagger - evaluation & analysis
    --------------------------------------------------
    Authors: adriana r.f. (@adrmisty), vera s.g. (@verasenderowiczg), emiel v.j (@emielvanderghinste).
    Date:    Nov-2026

    Description:
        The eval.py script handles the post-training (& post-processing too!) evaluation of the HMM tagger.
        It focuses on calculating performance metrics, visualization, and SPECIFIC error inspection.

    EXECUTION MODES:

        1. INFERENCE EVALUATION (--load-model)
        > Loads a pre-trained .pkl model.
        > Runs the Viterbi algorithm on the provided test set.
        > Best for: Checking how a specific model configuration performs on new data.

        2. STATIC ANALYSIS (--load-predictions)
        > Loads pre-computed tags from a text file (skipping Viterbi).
        > Best for: Instantly re-generating plots or analyzing results from 
          post-processed (heuristic-corrected) files.

    KEY FEATURES:

        [1] Metrics:  accuracy, precision, recall, and f1-score (from sklearn).
        [2] Visualization:    Confusion matrix heatmaps (--matrix).
        [3] Error inspection:  Deep-dive into specific mistags, e.g. showing nouns incorrectly classified
            as proper nouns, or viceversa.

    ARGUMENTS:
        --test <path>              Path to .conllu test data (Gold Standard).
        --load-model <path>        Path to a trained .pkl model (Mode 1).
        --load-predictions <path>  Path to a .txt prediction file (Mode 2).
        --matrix                   If set, plots the confusion matrix.

        [Error Analysis Arguments]
        --target-tag <str>         The Gold tag to investigate (e.g., 'NOUN').
        --error-type <str>         Type of error: 'recall', 'precision', or 'specific'.
        --mispredicted-as <str>    The predicted tag (required only for 'specific' mode).

    USAGE EXAMPLES:
        # 1. Standard evaluation of a model with a CM plot
        python analyze.py --load-model results/hmm.pkl --test data/test.conllu --matrix

        # 2. Analyze post-processed text files
        python analyze.py --load-predictions results/post_processed/preds.txt --test data/test.conllu

        # 3. Specific error analysis
        python analyze.py --load-model results/hmm.pkl --test data/test.conllu \
               --target-tag NOUN --error-type specific --mispredicted-as PROPN
    """
        
    # 1. load test data and the predictions (either from a loaded model,
    # or from loaded predictions from file)
    print(f"** Loading test data: {args.test} **")
    test_data = load_conllu(args.test)
    if test_data is None:
        sys.exit("Error: Test data required but could not be loaded.")
    predictions_nested, source_name = _get_predictions(args, test_data)

    # 2. metrics
    gold_tags = [tag for sent in test_data for word, tag in sent]
    pred_tags = [tag for sent in predictions_nested for word, tag in sent]
    metrics = compute_metrics(gold_tags, pred_tags)
    _display_metrics(metrics)
    
    # 3. visualize results
    _visualize(args, test_data, predictions_nested, metrics['f1'], source_name)

    # (!!!) further analyze results PER TAG
    # e.g. if u want to analyze pnouns/nouns/numerals/whatchamacallit especifically
    _specific_error_analysis(args, test_data, predictions_nested)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model performance (metrics, matrix, error analysis).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--test", type=Path, required=True, help="Path to UD test dataset (.conllu)")
    parser.add_argument("--load-model", type=Path, default=None, help="Path to trained HMM model (.pkl)")
    parser.add_argument("--load-predictions", type=Path, default=None, help="Path to pre-calculated predictions (.txt)")
    
    # Visualization
    parser.add_argument("--matrix", action="store_true", help="Plot the confusion matrix.")
    
    # Error Analysis
    parser.add_argument("--target-tag", type=str, default=None, help="GOLD tag to analyze.")
    parser.add_argument("--error-type", type=str, default='recall', choices=['recall', 'precision', 'specific'], help="Type of error analysis.")
    parser.add_argument("--mispredicted-as", type=str, default=None, help="PREDICTED tag (for 'specific' error type).")
    
    args = parser.parse_args()
    
    if args.load_model and args.load_predictions:
        parser.error("Cannot use both --load-model and --load-predictions simultaneously.")
    if args.error_type == 'specific' and (not args.target_tag or not args.mispredicted_as):
        parser.error("--error-type specific requires both --target-tag and --mispredicted-as.")

    main(args)

