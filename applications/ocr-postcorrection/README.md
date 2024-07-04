# ByT5-Sanskrit Analyzer 

This repository holds code for OCR postcorrection inference based on the ByT5-Sanskrit model as described in this paper: XXX  

## Data Source


The pretraining data was taken from the [Sangraha dataset](https://huggingface.co/datasets/ai4bharat/sangraha) by AI4Bharat.


## Requirements for Inference

- Python 3.6+
- PyTorch
- Transformers
- tqdm
- pandas

You can install these requirements with pip: `pip install torch transformers tqdm pandas`.  

### Inference Arguments (run_inf.py)

- `--input-file`: Path to the input file containing Sanskrit text (required)
- `--output-file`: Path to the output file (required)
- `--batch-size`: Batch size for processing (optional, default: 20)

### Inference Example

`python run_inf_dp.py --input-file toy-data/ocr-toy.txt --output-file toy-data/ocr-toy-corrected.txt`

## Inference Output

