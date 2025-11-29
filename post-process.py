import argparse
from pathlib import Path
import sys

# Logic from your existing project structure
from utils import load_predictions, save_predictions 
from postprocessing_rules import apply_all_rules


def main(args):
    """
    Loads raw predictions, applies post-processing rules (heuristics), 
    and saves the corrected output to a new location.
    
    Args:
        args: Command line arguments including input/output paths.
    """
    
    # --- 1. LOAD RAW PREDICTIONS ---
    print(f"** Loading raw predictions from: {args.input_predictions} **")
    predictions_nested = load_predictions(args.input_predictions)
    
    if predictions_nested is None:
        print("Error: Failed to load input predictions. Exiting.")
        sys.exit(1)
        
    initial_count = len(predictions_nested)
    print(f"    > Loaded {initial_count} sentences.")

    # --- 2. APPLY POST-PROCESSING RULES ---
    print("\n ** Applying post-processing heuristics... **")
    
    # This calls the functions defined in postprocessing_rules.py
    corrected_predictions = apply_all_rules(predictions_nested)
    
    # --- 3. SAVE CORRECTED PREDICTIONS ---
    
    # Determine the output path
    output_path = args.output_dir / args.input_predictions.name.replace(".txt", "_postprocessed.txt")
    
    print(f"\n ** Saving post-processed predictions to: {output_path} **")
    
    # Ensure output directory exists (handled by save_predictions in your utils.py)
    # The Path object in save_predictions will ensure the directory exists.
    save_predictions(corrected_predictions, output_path)

    print(f"\n ** Post-processing complete. Results saved. **")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applies heuristics to HMM prediction output and saves corrected results."
    )

    parser.add_argument(
        "--input-predictions", type=Path, required=True,
        help="Path to the raw HMM prediction file (.txt) saved via --save-predictions."
    )

    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory where post-processed prediction files should be saved (e.g., ./results/post_processed/)."
    )

    args = parser.parse_args()
    main(args)