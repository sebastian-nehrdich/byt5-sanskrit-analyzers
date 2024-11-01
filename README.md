# ByT5-Sanskrit Analyzers

This repository contains inference scripts for ByT5-Sanskrit analyzers developed for our EMNLP submission. The analyzer capabilities can be accessed in an interactive application at [dharmamitra.org](http://dharmamitra.org). The finetuning data for these models is taken from the [DCS](http://www.sanskrit-linguistics.org/dcs/).  

## Python Package
For those of you who do not want to run the model locally but just need a quick, working solution in python there is [this package](https://pypi.org/project/dharmamitra-sanskrit-grammar/).  

## Models
The pretrained base model is available here: [Huggingface link](https://huggingface.co/buddhist-nlp/byt5-sanskrit)  
The finetuned multitask model: [Huggingface link](https://huggingface.co/chronbmm/sanskrit5-multitask)

## Repository Structure

Our project is organized into three main directories. Currently, we make the applications/ section available, we will add the training and data sections in the future. 

- `applications/`: Contains various inference scripts for applying our model.

## Getting Started

For instructions how to run the individual downstream applications, see the README.md files in the subfolders under applications/. 


## Citation
The preprint is available on [arxiv](https://arxiv.org/abs/2409.13920). 
If you like our work and use it in your research, feel free to cite the paper:
```
@inproceedings{
nehrdichetal2024,
title={One Model is All You Need: ByT5-Sanskrit, a Unified Model for Sanskrit {NLP} Tasks},
author={Nehrdich, Sebastian and Hellwig, Oliver and Keutzer, Kurt},
booktitle={Findings of the 2024 Conference on Empirical Methods in Natural Language Processing},
year={2024},
}
```
