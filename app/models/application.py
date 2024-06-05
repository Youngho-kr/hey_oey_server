import os
import torch
import numpy as np

from gensim.models import FastText

from scenario import health
from kochat.app import KochatApi
from kochat.data import Dataset
from kochat.loss import CRFLoss, CenterLoss
from kochat.model import intent, entity
from kochat.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer
from huggingface_hub import hf_hub_download
# 환경 변수에서 토큰 가져오기
hf_token = os.getenv("HF_TOKEN")

# 토큰이 설정되어 있지 않으면 오류 메시지 출력
if not hf_token:
    raise ValueError("Hugging Face token not set in environment variables")

# 모델 초기화
repo_name = "darami819/capston1_closed_domain"
clf_model_path = hf_hub_download(repo_id=repo_name, filename="DistanceClassifier.pth", use_auth_token=hf_token)
rcn_model_path = hf_hub_download(repo_id=repo_name, filename="EntityRecognizer.pth", use_auth_token=hf_token)
gensim_embedder_path = hf_hub_download(repo_id=repo_name, filename="GensimEmbedder.gensim", use_auth_token=hf_token)
npy_file_path_1 = hf_hub_download(repo_id=repo_name, filename="GensimEmbedder.gensim.wv.vectors_ngrams.npy", use_auth_token=hf_token)
npy_file_path_2 = hf_hub_download(repo_id=repo_name, filename="GensimEmbedder.gensim.trainables.vectors_ngrams_lockf.npy", use_auth_token=hf_token)
dataset = Dataset(ood=False)

# 모델 로드 시도
try:
    emb_model = FastText.load(gensim_embedder_path)
    # print("Gensim model loaded successfully.")
    # print(emb_model)

    emb = GensimEmbedder(model=emb_model)
    # print("Embedder initialized successfully.")
    # print(emb)
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 실패: {e}")


# DistanceClassifier 로드
clf_model = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
)
clf_model.model.load_state_dict(torch.load(clf_model_path, map_location=torch.device('cpu')))

# EntityRecognizer 로드
rcn_model = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)
rcn_model.model.load_state_dict(torch.load(rcn_model_path, map_location=torch.device('cpu')))


# KochatApi 설정
kochat = KochatApi(
    dataset=dataset,
    embed_processor=(emb, False),  # 학습하지 않음
    intent_classifier=(clf_model, False),
    entity_recognizer=(rcn_model, False),
    scenarios=[health]
)

def main():
    uid = "test_user"
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response = kochat.request_chat(uid, user_input)  # 사용자 입력을 처리하고 응답 생성
            print("Bot:", response)
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
