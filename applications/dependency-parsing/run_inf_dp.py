from tqdm import tqdm
import argparse
from dp.model import process_batch
from utils.dp_serialization import turn_conllu_into_sentence_list


batch_size = 20
max_length = 256

def main():
    parser = argparse.ArgumentParser(description="Run inference on the ByT5 model for Vedic Sanskrit dependency parsing.")
    parser.add_argument("--input-file", required=True, help="Path to the input file. This should be a conllu file.")    
    parser.add_argument("--output-file", required=True, help="Path to the output file where arcs and labels will be written.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing.")
    args = parser.parse_args()

    batch_size = args.batch_size    

    # Read input file
    sentences = turn_conllu_into_sentence_list(args.input_file)
    processed_sentences = process_batch(sentences, batch_size, max_length)
    output_string = "\n".join(processed_sentences)
    with open(args.output_file, "w") as f:
        f.write(output_string)
        

    print(f"Results written to {args.output_file}")

if __name__ == "__main__":
    main()