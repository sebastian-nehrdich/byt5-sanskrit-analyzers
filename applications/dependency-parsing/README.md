# ByT5-Sanskrit Analyzer 

This repository holds code for dependency parsing inference based on the ByT5-Sanskrit model as described in this paper: XXX  

## Data Source

The training data for this model has been taken from the Digital Corpus of Sanskrit (DCS):

[http://www.sanskrit-linguistics.org/dcs/](http://www.sanskrit-linguistics.org/dcs/)

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

`python run_inf_dp.py --input-file toy-data/dp-toy.conllu --output-file toy-data/dp-output.txt`

## Inference Output

The inference output has the surface form, shortened label and arc on a single line per sentence. This needs to be further parsed to create proper conllu output. 
