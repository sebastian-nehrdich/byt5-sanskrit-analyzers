import argparse
from tqdm import tqdm
import pandas as pd
from inf.tags import postprocess_sentence
from inf.model import process_batch
import os

def process_file(input_file, mode, batch_size):
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    # Process batches
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc=f"Processing {os.path.basename(input_file)}"):
        batch = lines[i:i+batch_size]
        results.extend(process_batch(batch, mode))
    results = [postprocess_sentence(result, mode=mode) for result in results]

    # Create output DataFrame
    segmentnr_base_name = os.path.basename(input_file).split(".")[0]
    output = pd.DataFrame(columns=["segmentnr", "original", "analyzed"])
    segmentnrs = [f"{segmentnr_base_name}-{i}" for i in range(len(results))]
    output["segmentnr"] = segmentnrs
    output["original"] = lines
    output["analyzed"] = results

    return output

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on ByT5-Sanskrit with nexus-style TSV output.")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing .txt files")
    parser.add_argument("--mode", required=True, choices=['lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'], help="Processing mode")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing.")
    args = parser.parse_args()


    # Process all .txt files in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(args.input_folder, filename)
            output_file = os.path.join(args.input_folder, filename.replace(".txt", ".tsv"))

            output_df = process_file(input_file, args.mode, args.batch_size)
            output_df.to_csv(output_file, sep="\t", index=False)

            print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()