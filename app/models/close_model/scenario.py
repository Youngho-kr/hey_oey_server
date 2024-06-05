"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from kocrawl.dust import DustCrawler
# from kocrawl.weather import WeatherCrawler
from kochat.app import Scenario
# from data_manager import DataManager
# from new_api import custom_api
# from kocrawl.map import MapCrawler


# data_manager = DataManager('data/response/health.csv')
# df = pd.read_csv('data/response/health.csv')
# print(df.columns)  # 모든 열 이름을 출력하여 'part' 열이 있는지 확인

health = Scenario(
    intent='health',
    # api=custom_api,
    scenario={
        'part': [],
        'issue': [],
    }
)