# Copyright 2020 Kochat. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
# from collections import Callable
from copy import deepcopy
from random import randint
from kochat.decorators import data
from data_manager import DataManager  # 데이터 관리 클래스 임포트

import os

# DataManager 인스턴스 생성 예
data_manager = DataManager('data/response/health.csv')

@data
class Scenario:

    def __init__(self, intent, scenario=None):
        self.intent = intent
        self.scenario, self.default = self.__make_empty_dict(scenario)

    def __make_empty_dict(self, scenario):
        default = {}
        for k, v in scenario.items():
            if len(scenario[k]) > 0:
                default[k] = v # 디폴트 딕셔너리로 일단 빼놓고
                scenario[k] = [] # 해당 엔티티의 리스트를 비워둠
            else:
                default[k] = []
                # 디폴드 없으면 리스트로 초기화

        return scenario, default

    def __check_entity(self, entity: list, text: list, dict_: dict) -> dict:
        """
        문자열과 엔티티를 함께 체크해서
        딕셔너리에 정의된 엔티티에 해당하는 단어 토큰만 추가합니다.

        :param text: 입력문자열 토큰(단어) 리스트
        :param entity: 엔티티 리스트
        :param dict_: 필요한 엔티티가 무엇인지 정의된 딕셔너리
        :return: 필요한 토큰들이 채워진 딕셔너리
        """
        # print("check entity : ",text, entity)
        for t, e in zip(text, entity):
            for k, v in dict_.items():
                if k.lower() in e.lower():
                    v.append(t)
                    # print("v is ", v)
                    # print("text is ",t)
        # print(dict_)
        return dict_

    def __set_default(self, result):
        for k, v in result.items():
            if len(result[k]) == 0 and len(self.default[k]) != 0:
                # 디폴트 값 중에서 랜덤으로 하나 골라서 넣음
                result[k] = \
                    [self.default[k][randint(0, len(self.default[k]) - 1)]]

            result[k] = ' '.join(result[k])
        return result

    def apply(self, entity, text):
        scenario = deepcopy(self.scenario)
        result = self.__check_entity(entity, text, scenario)
        # print(scenario)
        # print("check entity 하고 나서 result :",result)
        result = self.__set_default(result)
        # print("set default 하고 나서 result :",result)
        required_entity = [k for k, v in result.items() if len(v) == 0]
        # print("required_entity :", required_entity)
        upper_required_entity = []
        for i in required_entity:
            upper_required_entity.append(i.upper())

        # print("required_entity :", upper_required_entity)

        if len(required_entity) == 0:
            # print("0일 때")
            return {
                'input': text,
                'intent': self.intent,
                'entity': entity,
                'state': 'SUCCESS',
                'answer': data_manager.get_info(self.intent, result)
            }
        else:
            if 'ISSUE' in upper_required_entity:
                # print("ISSUE가 필요해")
                return {
                    'input': text,
                    'intent': self.intent,
                    'entity': entity,
                    'state': 'REQUIRE_' + '_'.join(upper_required_entity),
                    'answer': None
                }
            elif 'PART' in upper_required_entity:
                # print("뭔가 인자가 더 필요하지만 part니까 무시할게")
                return {
                    'input': text,
                    'intent': self.intent,
                    'entity': entity,
                    'state': 'SUCCESS',
                    'answer': data_manager.get_info(self.intent, result)
                }
            else:
                # print('else')
                return {
                    'input': text,
                    'intent': self.intent,
                    'entity': entity,
                    'state': 'REQUIRE_' + '_'.join(upper_required_entity),
                    'answer': "추가 정보 필요"
                }
