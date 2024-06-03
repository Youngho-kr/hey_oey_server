import openai
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
openai.api_key = os.environ.get('openai.api_key')
model_id = os.environ.get('model_id')

# KoELECTRA 모델 로드
tokenizer_koelectra = AutoTokenizer.from_pretrained("yshyeonn/hey-oey-open")
model_koelectra = ElectraForSequenceClassification.from_pretrained("yshyeonn/hey-oey-open")

# 문장 분류 함수
def classify_intent(sentence):
    inputs = tokenizer_koelectra(sentence, return_tensors="pt")
    outputs = model_koelectra(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    labels = ["긍정", "경청", "공감"]
    return labels[predictions.item()]

# 대화 이력 초기화
conversation_history = [
    {"role": "system", "content": "You are an AI model designed to provide psychological counseling to parents raising young children."}
]

# 텍스트 생성 및 레이블링 함수
def generate_and_classify(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation_history
    )
    response = completion.choices[0].message['content']
    conversation_history.append({"role": "assistant", "content": response})
    
    label = classify_intent(response)
    return response, label

if __name__ == "__main__":
    while True:
        user_input = input("입력: ")
        if user_input.lower() == "quit":
            break
        response, label = generate_and_classify(user_input)
        print(f"Chatbot: {response}")
        print(f"Intent: {label}")
