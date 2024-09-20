# Welcome to ByT5-Sanskrit

This repository contains inference scripts for ByT5-Sanskrit analyzers developed by Sebastian Nehrdich and Oliver Hellwig. The analyzer capabilities can be accessed in an interactive application at [dharmamitra.org](http://dharmamitra.org). A publication describing the tools is accepted at EMNLP 2024. The finetuning data for these models is taken from the [DCS](http://www.sanskrit-linguistics.org/dcs/).  

## Models
The pretrained base model is available here: [Huggingface link](https://huggingface.co/buddhist-nlp/byt5-sanskrit)
The finetuned multitask model: [Huggingface link](https://huggingface.co/chronbmm/sanskrit5-multitask)

## Repository Structure

Our project is organized into three main directories. Currently, we make the applications/ section available, we will add the training and data sections in the future. 

- `applications/`: Contains various inference scripts for applying our model.

## Getting Started

For instructions how to run the individual downstream applications, see the README.md files in the subfolders under applications/. 

## Citation
If you like our work and use it in your research, feel free to cite the paper:
```
@inproceedings{
nehrdichetal2024,
title={One Model is All You Need: ByT5-Sanskrit, a Unified Model for Sanskrit {NLP} Tasks},
author={Nehrdich, Sebastian and Hellwig, Oliver and Keutzer, Kurt},
booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
year={2024},
}
```
