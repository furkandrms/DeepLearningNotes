from datasets import load_dataset
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)


model_checkpoint = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# This example presents a simple approach to tokenization and training. More advanced methods may be required in real applications.
for sample in pretrained_dataset.take(100):  
    inputs = tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)


def predict(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)

    predicted_tokens = [tokenizer.decode(tok, skip_special_tokens=True) for tok in predictions]

    return predicted_tokens

# Örnek metin girdisi, `[MASK]` tokeni ile birlikte
input_text = "Chocolate is the best [MASK] treat."

predicted_output = predict(input_text, model, tokenizer)

print("Predicted Output:", predicted_output)

