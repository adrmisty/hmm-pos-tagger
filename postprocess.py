# postprocess.py
# adriana r.f. (@adrmisty)
# vera s.g. (@verasenderowiczg)
# emiel v.j. (@emielvanderghinste)
# nov-2026

import sys
import argparse
from pathlib import Path

from utils_io import load_predictions, save_predictions
from heuristics import apply_heuristics

def main(args):
    """Applies a series of heuristic rules to post-process HMM model predictions."""
    
    print(f"** Loading raw predictions from: {args.input_predictions} **")
    predictions = load_predictions(args.input_predictions)
    
    if predictions is None:
        print("Error: Failed to load predictions.")
        sys.exit(1)
    print(f"    > Loaded {len(predictions)} sentences.")
    
    # apply heuristics 
    # to file {original_name}_postprocessed.txt
    print("\n ** Applying post-processing heuristics... **")
    corrected_predictions = apply_heuristics(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.input_predictions.stem + "_postprocessed" + args.input_predictions.suffix
    output_path = args.output_dir / filename
    
    print(f"\n** Saving post-processed predictions to: {output_path} **")
    save_predictions(corrected_predictions, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applies post processing rules to HMM prediction output to POS tagging errors from the model's implementation."
    )

    parser.add_argument(
        "--input-predictions", type=Path, required=True,
        help="Path to the raw HMM prediction file (.txt)."
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("./results/post_processed/"),
        help="Directory where corrected files should be saved."
    )

    args = parser.parse_args()    
    main(args)