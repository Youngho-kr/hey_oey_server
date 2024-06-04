from torch import Tensor
from torch import nn
from torchcrf import CRF

from kochat.decorators import entity
from kochat.loss.base_loss import BaseLoss

import logging

# Set up logging
# logging.basicConfig(level=logging.DEBUG)
@entity
class CRFLoss(BaseLoss):

    def __init__(self, label_dict: dict):
        """
        Conditional Random Field를 계산하여 Loss 함수로 활용합니다.

        :param label_dict: 라벨 딕셔너리
        """

        super().__init__()
        self.classes = len(label_dict)
        # self.crf = CRF(self.classes, batch_first=True)
        self.crf = CRF(self.classes)


    def decode(self, logits: Tensor, mask: nn.Module = None) -> list:
        """
        Viterbi Decoding의 구현체입니다.
        CRF 레이어의 출력을 prediction으로 변형합니다.

        :param logits: 모델의 출력 (로짓)
        :param mask: 마스킹 벡터
        :return: 모델의 예측 (prediction)
        """
        # logits의 크기를 확인하여 시퀀스 길이가 0인 경우 빈 리스트를 반환
        # if logits.size(0) == 0 or logits.size(1) == 0:
        #     return []
        #
        #
        # logits = logits.permute(0, 2, 1)
        # return self.crf.decode(logits, mask)
        batch_size, seq_length, num_tags = logits.size()
        # logging.debug(f"logits size: {logits.size()}")

        # if mask is not None:
        #     logging.debug(f"mask size: {mask.size()}")

        # batch_size, seq_length, num_tags = logits.size()
        # logging.debug(f"logits size: {logits.size()}")
        # if mask is not None:
        #     logging.debug(f"mask size: {mask.size()}")
        #
        # # 시퀀스 길이가 0인 경우 빈 리스트를 반환
        # if seq_length == 0:
        #     logging.debug("Sequence length is 0, returning empty list.")
        #     return []
        #
        # logits = logits.permute(0, 2, 1)  # (batch_size, num_tags, seq_length)
        #
        # decoded = []
        # for i in range(batch_size):
        #     seq_logits = logits[i]
        #     if mask is not None:
        #         seq_mask = mask[i]
        #         seq_length = int(seq_mask.sum().item())  # 마스크된 시퀀스 길이 계산
        #         logging.debug(f"Sequence length after masking: {seq_length}")
        #         if seq_length == 0:
        #             decoded.append([])
        #             continue
        #     else:
        #         seq_length = logits.size(2)  # 마스크가 없을 경우 시퀀스 길이
        #
        #     seq_logits = seq_logits[:, :seq_length]  # 유효한 시퀀스 길이만큼 자르기
        #     if seq_length == 0:
        #         decoded.append([])
        #     else:
        #         try:
        #             logging.debug(f"Decoding sequence of length {seq_length}")
        #             decoded.append(self.crf.decode(seq_logits.unsqueeze(0))[0])  # 배치 크기 1로 디코딩
        #         except IndexError as e:
        #             logging.error(f"IndexError during decoding: {e}")
        #             decoded.append([])
        #
        # print("decode : ", decoded)
        # return decoded
        logits = logits.permute(0, 2, 1)
        return self.crf.decode(logits, mask)

    def compute_loss(self, label: Tensor, logits: Tensor, feats: Tensor, mask: nn.Module = None) -> Tensor:
        """
        학습을 위한 total loss를 계산합니다.

        :param label: label
        :param logits: logits
        :param feats: feature
        :param mask: mask vector
        :return: total loss
        """

        logits = logits.permute(0, 2, 1)
        logging.debug(f"Computing loss with logits size: {logits.size()}, label size: {label.size()}")
        if mask is not None:
            logging.debug(f"mask size: {mask.size()}")
        log_likelihood = self.crf(logits, label, mask=mask, reduction='mean')
        return - log_likelihood  # nll loss
