from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

model_name = "chronbmm/sanskrit5-multitask"
max_length = 512
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def process_batch(batch, mode):
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