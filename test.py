from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 모델과 토크나이저 로드
MODEL_NAME = "eunjin/kogpt2-finetuned-wellness"
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

def chatbot(question, max_length=128):
    # 입력 받은 질문을 토큰화하고, 모델이 답변을 생성하도록 합니다.
    input_ids = tokenizer.encode(question, return_tensors='pt')
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,  # 빔 서치를 사용하여 다양한 가능성 탐색
        no_repeat_ngram_size=2,  # 반복 n-gram 크기 제한
        early_stopping=True,  # 빔 서치 조기 중단
        top_k=50,  # 확률 높은 top 50 단어만 고려
        top_p=0.95  # 확률 누적 합이 0.95를 넘지 않는 단어들을 고려
    )
    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return reply

# 예시 질문으로 챗봇 테스트
question = "요즘 너무 우울해"
print("질문:", question)
print("답변:", chatbot(question))
