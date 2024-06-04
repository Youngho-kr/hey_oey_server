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


from kochat.app.scenario import Scenario


class ScenarioManager:

    def __init__(self):
        self.scenarios = []

    def add_scenario(self, scen: Scenario):
        if isinstance(scen, Scenario):
            self.scenarios.append(scen)
            # print("Scenario added")
        else:
            raise Exception('시나리오 객체만 입력 가능합니다.')

    def apply_scenario(self, intent, entity, text):
        # print(text)


        for scenario in self.scenarios:
            # print(scenario.intent, intent)
            # print("아으아아ㅡ아ㅡ아")
            if scenario.intent == intent:
                return scenario.apply(entity, text)

        return {
            'input': text,
            'intent': intent,
            'entity': entity,
            'state': 'FALLBACK',
            'answer': None
        }
