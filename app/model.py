from transformers import BertTokenizer, BertModel

model_name = "bert-base-uncased"

def load_model():
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return model, tokenizer

def process_message(message, model, tokenizer):
    # Simulate processing: Replace with actual model inference
    response = "Processed: " + message
    return response
