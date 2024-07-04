from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelpath = "chronbmm/sanskrit-byt5-dp"

model = AutoModelForSeq2SeqLM.from_pretrained(modelpath).to(device)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

def process_batch(sentences, batch_size, max_length):
    all_results = []
    for i in tqdm(range(0,len(sentences), batch_size)):
        current_sentences = sentences[i:i+batch_size]

        encoded = tokenizer(current_sentences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
        generated_tokens = model.generate(**encoded, max_length=max_length, num_beams=5, length_penalty = 1.5, repetition_penalty= 1.5, )
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_results += result
    return all_results