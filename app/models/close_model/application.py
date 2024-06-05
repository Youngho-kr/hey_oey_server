# 필요한 라이브러리를 불러옵니다.
import os
import torch
import numpy as np

from gensim.models import FastText
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# 환경 변수에서 토큰 가져오기
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    raise ValueError("Hugging Face token not set in environment variables")

# 모델 초기화
repo_name = "darami819/capston1_closed_domain"
clf_model_path = hf_hub_download(repo_id=repo_name, filename="DistanceClassifier.pth", use_auth_token=hf_token)
rcn_model_path = hf_hub_download(repo_id=repo_name, filename="EntityRecognizer.pth", use_auth_token=hf_token)
gensim_embedder_path = hf_hub_download(repo_id=repo_name, filename="GensimEmbedder.gensim", use_auth_token=hf_token)

# 필요한 추가 파일
npy_file_path_1 = hf_hub_download(repo_id=repo_name, filename="GensimEmbedder.gensim.wv.vectors_ngrams.npy", use_auth_token=hf_token)
npy_file_path_2 = hf_hub_download(repo_id=repo_name, filename="GensimEmbedder.gensim.trainables.vectors_ngrams_lockf.npy", use_auth_token=hf_token)

from kochat.app import KochatApi
from kochat.data import Dataset
from kochat.loss import CRFLoss, CenterLoss
from kochat.model import intent, entity
from kochat.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer
from scenario import health

# 데이터셋 및 모델 로드
dataset = Dataset(ood=False)

try:
    emb_model = FastText.load(gensim_embedder_path)
    emb = GensimEmbedder(model=emb_model)
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 실패: {e}")

# Classifier 모델 로드
clf_model = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
)
clf_model.model.load_state_dict(torch.load(clf_model_path, map_location=torch.device('cpu')))

# EntityRecognizer 모델 로드
rcn_model = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)
rcn_model.model.load_state_dict(torch.load(rcn_model_path, map_location=torch.device('cpu')))

# Kochat API 인스턴스 생성
kochat = KochatApi(
    dataset=dataset,
    embed_processor=(emb, False),
    intent_classifier=(clf_model, False),
    entity_recognizer=(rcn_model, False),
    scenarios=[health]
)

def request_chat(message, uid="test_user"):
    try:
        return kochat.request_chat(uid, message)
    except Exception as e:
        print(f"오류 발생: {e}")
        return None
