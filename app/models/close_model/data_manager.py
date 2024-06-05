# 파일 이름: data_manager.py
import pandas as pd

class DataManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_data()

    def load_data(self):
        # 데이터를 로드하는 로직
        # 예: JSON 파일, CSV 파일, 데이터베이스 등
        # 실제 파일 또는 데이터베이스로부터 데이터를 로드하는 코드
        data = pd.read_csv(self.filepath)
        data['entity_key1'] = data['entity_key1'].str.strip()  # 공백 제거
        data['entity_value1'] = data['entity_value1'].str.strip()
        data['entity_key2'] = data['entity_key2'].str.strip()
        data['entity_value2'] = data['entity_value2'].str.strip()
        self.data = data

    def get_info(self, intent, entities):
        # 데이터프레임에서 조건에 맞는 데이터를 조회
        # entities는 dict 형태이며, 이를 이용해 조건에 맞는 행을 필터링합니다.
        # print("entities :",entities)

        # Intent를 기반으로 첫 번째 필터링
        filtered_data = self.data[self.data['intent'] == intent]
        # print("intent data :", filtered_data)

        if entities['part'] != '':
            filtered_data = filtered_data[(filtered_data['entity_value1'] == entities['part']) | (filtered_data['entity_value1'] == 'O')]
            filtered_data = filtered_data[(filtered_data['entity_value2'] == entities['issue'])]
        else:
            filtered_data = filtered_data[(filtered_data['entity_value2'] == entities['issue'])]

        # print("filtering :", filtered_data)

        # 결과가 있으면 첫 번째 행의 'response' 컬럼 값을 반환, 없으면 기본 메시지를 반환
        if not filtered_data.empty:
            # print("결과가 있는 것 같네요")
            return filtered_data.iloc[0]['response']
        else:
            return "해당 정보를 찾을 수 없습니다."
