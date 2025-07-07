import os
import argparse
import pandas as pd
import semantic_compiler as semantic
import sense_compiler as sense

def run_sense_compile(folder_path, decade_filter=None, output_dir="exports"):
    """
    Run sense compilation on all CSVs in the specified folder.
    If decade_filter is provided, only include CSVs whose filename contains that string.
    Outputs .npy and .csv files for each adjective in each CSV to the specified output_dir.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path is not a directory: {folder_path}")

    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv") and (decade_filter in f if decade_filter else True)
    ]

    if not csv_files:
        print("No matching CSV files found in the directory.")
        return

    print(f"Found {len(csv_files)} CSV file(s). Running sense compiler...")
    sense.run_sense_compiler(csv_files, output_dir=output_dir)
    print("Sense compilation completed.")

def run_semantic_compile(staging_root, output_csv_dir):
    """
    Traverse the staging directory (e.g. './staging') and run semantic analysis on each word/decade folder.
    Save per-word results as CSVs in output_csv_dir.
    """
    if not os.path.isdir(staging_root):
        raise ValueError(f"Provided staging root is not a directory: {staging_root}")
    os.makedirs(output_csv_dir, exist_ok=True)

    for word in os.listdir(staging_root):
        word_path = os.path.join(staging_root, word)
        if not os.path.isdir(word_path):
            continue
        for decade in os.listdir(word_path):
            matrix_path = os.path.join(word_path, decade, f"{word}_embeddings.npy")
            if os.path.exists(matrix_path):
                try:
                    semantic.run_semantic_compiler(word, decade, output_dir=output_csv_dir)
                except Exception as e:
                    print(f"[WARN] Failed to process {word}/{decade}: {e}")
            else:
                print(f"[SKIP] Missing matrix for {word}/{decade}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sense/Semantic Compiler Pipeline")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run_sense_compile
    parser_sense = subparsers.add_parser("run_sense_compile", help="Run sense compiler on all CSVs in a folder")
    parser_sense.add_argument("csv_folder", type=str, help="Path to folder containing CSV files")
    parser_sense.add_argument("output_dir", type=str, default="exports", help="Directory to write .npy and metadata outputs")

    parser_sense.add_argument("--decade", type=str, default=None, help="Optional: Filter to only CSVs with this decade string in the filename")

    # Subcommand: run_semantic_compile
    parser_semantic = subparsers.add_parser("run_semantic_compile", help="Run semantic analysis on all word/decade embeddings in staging")
    parser_semantic.add_argument("staging_root", type=str, help="Path to the ./staging directory containing word folders")
    parser_semantic.add_argument("output_csv_dir", type=str, help="Path to output directory for semantic metric CSVs")

    args = parser.parse_args()

    if args.command == "run_sense_compile":
        run_sense_compile(args.csv_folder, args.decade, args.output_dir)

    elif args.command == "run_semantic_compile":
        run_semantic_compile(args.staging_root, args.output_csv_dir)
