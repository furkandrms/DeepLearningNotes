from datasets import load_dataset
pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)

from transformers import AutoModelForMaskedLM, AutoTokenizer

model_checkpoint = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# This example presents a simple approach to tokenization and training. More advanced methods may be required in real applications.
for sample in pretrained_dataset.take(100):  # Örneğin ilk 100 örneği alın
    inputs = tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

import torch

def predict(input_text, model, tokenizer):
    # Girdi metnini modelin beklediği formata dönüştür
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Modeli eğitim modundan çıkar (Dropout vb. etkileri kaldırır)
    model.eval()

    # Tahmin yaparken gradyan hesaplamasını devre dışı bırak
    with torch.no_grad():
        outputs = model(**inputs)

    # Modelin çıktılarından tahminleri elde et
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Token ID'lerini gerçek metne çevir
    predicted_tokens = [tokenizer.decode(tok, skip_special_tokens=True) for tok in predictions]

    return predicted_tokens

# Örnek metin girdisi, `[MASK]` tokeni ile birlikte
input_text = "Chocolate is the best [MASK] treat."

# Tahmin fonksiyonunu kullanarak modelin tahminini al
predicted_output = predict(input_text, model, tokenizer)

# Tahmini yazdır
print("Predicted Output:", predicted_output)

