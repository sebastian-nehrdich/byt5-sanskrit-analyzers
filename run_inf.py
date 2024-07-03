import argparse
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from utils.tags import postprocess_sentence

model_name = "chronbmm/sanskrit5-multitask"
max_length = 512



def process_batch(model, tokenizer, batch, mode, max_length):
    prefix = {
        'segmentation': "S ",
        'segmentation-morphosyntax': "SM ",
        'lemma': "L ",
        'lemma-morphosyntax': "LM ",
        'segmentation-lemma-morphosyntax': "SLM "
    }

    input_texts = [f"{prefix[mode]}{text}" for text in batch]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a T5 model for Sanskrit processing")
    parser.add_argument("--input-file", required=True, help="Path to the input file")
    parser.add_argument("--mode", required=True, choices=['lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'], help="Processing mode")
    parser.add_argument("--output-mode", type=str, default="txt", choices=["txt", "tsv"], help="Output mode. TXT is default, TSV will create nexus-friendly TSV output with the format segmentnr<TAB>segment<TAB>analysis.")
    parser.add_argument("--output-file", required=True, help="Path to the output file")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing.")
    args = parser.parse_args()

    batch_size = args.batch_size
    
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    # Process batches
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
        batch = [line.strip() for line in lines[i:i+batch_size]]
        results.extend(process_batch(model, tokenizer, batch, args.mode, max_length))
    results = [postprocess_sentence(result, mode=args.mode) for result in results]
    
    # Write results to output file
    if args.output_mode == "txt":        
        with open(args.output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"{result}\n")

    elif args.output_mode == "tsv":
        segmentnr_base_name = args.input_file.split("/")[-1].split(".")[0]
        output = pd.DataFrame(columns=["segmentnr", "original", "analyzed"])
        segmentnrs = [f"{segmentnr_base_name}-{i}" for i in range(len(results))]
        output["segmentnr"] = segmentnrs
        output["original"] = lines
        output["analyzed"] = results
        output.to_csv(args.output_file, sep="\t", index=False)

    print(f"Results written to {args.output_file}")

if __name__ == "__main__":
    main()
