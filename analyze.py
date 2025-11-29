import argparse
from pathlib import Path
import sys


from utils import load_conllu, load_model, load_predictions, plot_confusion_matrix, analyze_tag_errors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main(args):
    """
    Handles all post-training evaluation and analysis modes:
    1) Run Viterbi on test data using a loaded model.
    2) Load pre-calculated predictions for analysis.
    """
    hmm = None
    
    # 1. LOAD GOLD TEST DATA (REQUIRED FOR ALL MODES)
    print(f"** Loading test data: {args.test} **")
    test_data = load_conllu(args.test)
    if test_data is None:
        print("Error: Test data required but could not be loaded.")
        sys.exit(1)

    # 2. DECIDE PREDICTION SOURCE (Load Model vs. Load Predictions)
    if args.load_predictions:
        # MODE A: Load predictions from file (Skip Viterbi)
        print(f"\n ** Loading pre-calculated predictions **")
        predictions_nested = load_predictions(args.load_predictions)
        if predictions_nested is None:
             print("Error: Could not load predictions file.")
             sys.exit(1)

    elif args.load_model:
        # MODE B: Load model and run Viterbi
        print(f"** Loading pre-trained HMM model from: {args.load_model} **")
        hmm = load_model(args.load_model)
        
        print("\n ** Running HMM prediction (Viterbi) **")
        test_words_nested = [[word for (word, tag) in sent] for sent in test_data]
        predictions_nested = hmm.predict(test_words_nested)
        
    else:
        # No model or predictions source provided
        print("Error: Must provide either --load-model or --load-predictions.")
        sys.exit(1)

    # --- 3. EVALUATION AND PLOTTING ---
    
    # Flatten the data once for metrics calculation and plotting
    gold_tags = [tag for sent in test_data for word, tag in sent]
    pred_tags = [tag for sent in predictions_nested for word, tag in sent]
    
    # Calculate all metrics using sklearn
    # Note: We use 'macro' average for the general report.
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_tags, 
        pred_tags, 
        average='macro', 
        beta=1, 
        zero_division=0
    )
    accuracy_total = accuracy_score(gold_tags, pred_tags)

    print("\n ** Evaluating HMM part-of-speech tagger **")
    
    # Print all requested metrics
    print(f"    > Overall Tagging Accuracy: {accuracy_total * 100:.2f}%")
    print(f"    > Macro-Averaged Precision: {precision_macro * 100:.2f}%")
    print(f"    > Macro-Averaged Recall:    {recall_macro * 100:.2f}%")
    print(f"    > Macro-Averaged F1-Score:  {f1_macro * 100:.2f}%")


    if args.matrix:
        print("\n ** Plotting confusion matrix **")
        
        # Determine the title based on the mode
        if args.load_model:
            model_title = args.load_model.name
        else:
             model_title = f"Loaded Predictions: {args.load_predictions.name}"
        
        plot_confusion_matrix(
            test_data_tagged=test_data, 
            predictions=predictions_nested, 
            # Include F1 score in the title
            model_name=f"HMM PoS Tagger - {model_title} (F1: {f1_macro:.4f})"
        )
        print(" ** Confusion matrix plotted **")
        
    # --- 4. ERROR ANALYSIS (NEW FEATURE) ---
    if args.target_tag:
        # Pass the new argument to the utility function
        analyze_tag_errors(
            gold_data=test_data, 
            predictions=predictions_nested, 
            target_tag=args.target_tag, 
            error_type=args.error_type,
            mispredicted_as=args.mispredicted_as # NEW ARGUMENT
        )
        
# --- ARGUMENT PARSING AND ENTRY POINT ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model performance (metrics, matrix, error analysis).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--test", type=Path, required=True,
        help="Path to UD test dataset (.conllu format). Required for evaluation."
    )
    
    parser.add_argument(
        "--load-model", type=Path, default=None,
        help="Path to an already trained HMM model (.pkl) for prediction."
    )
    
    parser.add_argument(
        "--load-predictions", type=Path, default=None,
        help="Path to load pre-calculated predicted tags from a text file, skipping Viterbi."
    )

    parser.add_argument(
        "--matrix", action="store_true", 
        help="If set, plot the confusion matrix."
    )
    
    # New arguments for analyze_tag_errors
    parser.add_argument(
        "--target-tag", type=str, default=None,
        help="The GOLD tag (e.g., 'NOUN') to analyze for errors."
    )
    
    parser.add_argument(
        "--error-type", type=str, default='recall',
        choices=['recall', 'precision', 'specific'], # Added 'specific' choice
        help="Type of error to analyze: 'recall' (False Negatives), 'precision' (False Positives), or 'specific' (use with --mispredicted-as)."
    )
    
    # NEW ARGUMENT FOR SPECIFIC ERROR ANALYSIS
    parser.add_argument(
        "--mispredicted-as", type=str, default=None,
        help="The PREDICTED tag (e.g., 'NOUN') when --target-tag is the GOLD tag. Must be used with --error-type specific."
    )
    
    args = parser.parse_args()
    
    # Final check for mutual exclusivity
    if args.load_model and args.load_predictions:
        parser.error("Cannot use both --load-model and --load-predictions simultaneously.")
        
    # Validation for new specific analysis mode
    if args.error_type == 'specific':
        if not args.target_tag or not args.mispredicted_as:
            parser.error("When using --error-type specific, both --target-tag (GOLD) and --mispredicted-as (PREDICTED) must be provided.")

    main(args)