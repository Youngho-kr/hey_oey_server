import openai
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get('openai.api_key')
model_id = os.environ.get('model_id')

# 텍스트 분류 모델 로드
tokenizer = AutoTokenizer.from_pretrained("Falconsai/intent_classification")
model = AutoModelForSequenceClassification.from_pretrained("Falconsai/intent_classification")

# 문장 분류 함수
def classify_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    labels = ["긍정", "경청", "공감"]
    return labels[predictions.item()]

# 텍스트 생성 및 레이블링 함수
def generate_and_classify(user_input):
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an AI model designed to provide psychological counseling to parents raising young children."},
            {"role": "user", "content": user_input}
        ]
    )

    response = completion.choices[0].message.content
    label = classify_intent(response)
    return response, label

# 예시 입력
while(True):
    user_input = input("입력 :")
    if(user_input == "quit"): break
    response, label = generate_and_classify(user_input)
    print(f"Chatbot: {response}")
    print(f"Intent: {label}")