import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_model(model_path):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def classify_text(text, model, tokenizer):
    device = next(model.parameters()).device
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

    label_map = {0: 'feedback', 1: 'inquiry', 2: 'refund'}
    return label_map[predicted.item()]

def main():
    model_path = 'model/distilbert_classifier.pth'
    model = load_model(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    while True:
        text = input("Enter text to classify (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        prediction = classify_text(text, model, tokenizer)
        print(f"Classification: {prediction}")

if __name__ == "__main__":
    main()
