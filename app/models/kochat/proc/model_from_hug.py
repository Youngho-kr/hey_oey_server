import torch
# from kochat_config import download_model_from_hub, load_model_from_hub_link
from .distance_classifier import DistanceClassifier
from .entity_recognizer import EntityRecognizer
from kochat.model import intent, entity
from kochat.loss import CRFLoss, CenterLoss
from kochat.data import Dataset

dataset = Dataset(ood=False)


from huggingface_hub import hf_hub_download
def download_model_from_hub(repo_name, filename):
    """Hugging Face Hub에서 파일을 다운로드하고 경로를 반환합니다."""
    return hf_hub_download(repo_id=repo_name, filename=filename)

# def load_model_from_hub_link(model, file_path):
#     """
#         주어진 경로에서 모델 상태 딕셔너리를 로드하여 모델 객체에 적용합니다.
#
#         Args:
#         model (torch.nn.Module): 상태 딕셔너리를 적용할 모델 객체.
#         file_path (str): 모델 상태 딕셔너리 파일의 경로.
#
#         Returns:
#         torch.nn.Module: 상태가 업데이트된 모델 객체.
#         """
#     # 파일 경로에서 모델 상태 딕셔너리를 로드합니다.
#     model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
#     return model
class ModelLoader:
    _instance = None

    def __new__(cls, repo_name):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.clf_model = None
            cls._instance.rcn_model = None
            cls._instance.load_model_from_hug(repo_name)
            print("done")
        return cls._instance

    def load_model_from_hug(self, repo_name):
        clf_model_path = download_model_from_hub(repo_name, "DistanceClassifier.pth")
        rcn_model_path = download_model_from_hub(repo_name, "EntityRecognizer.pth")

        print(clf_model_path)
        print(rcn_model_path)

        # 임시 레이블 딕셔너리
        dummy_labels = {
            "0": 0, "PART": 1, "ISSUE": 2
        }

        dummy_intent_model = intent.CNN(dataset.intent_dict)
        print("AA")
        dummy_entity_model = entity.LSTM(dataset.entity_dict)
        print("BB")

        dummy_intent_loss = CenterLoss(dataset.intent_dict)
        print("CC")

        dummy_entity_loss = CRFLoss(dataset.entity_dict)
        print("DD")


        self.clf_model = DistanceClassifier(model=dummy_intent_model, loss=dummy_intent_loss)
        print("EE")

        self.clf_model.model.load_state_dict(torch.load(clf_model_path, map_location=torch.device('cpu')))
        print("GG")

        self.rcn_model = EntityRecognizer(model=dummy_entity_model, loss=dummy_entity_loss)
        print("FF")

        self.rcn_model.model.load_state_dict(torch.load(rcn_model_path, map_location=torch.device('cpu')))
        print("HH")


    @property
    def clf(self):
        if not self.clf_model:
            raise ValueError("Classifier model is not loaded")
        return self.clf_model

    @property
    def rcn(self):
        if not self.rcn_model:
            raise ValueError("Entity recognizer model is not loaded")
        return self.rcn_model
