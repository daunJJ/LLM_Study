## Prompt Tuning

### Prompt Tuning 아키텍쳐

- **Prompt Tuning의 핵심 idea** : fine-tuning 과정시 pre-train이 끝난 파라미터 w_0를 고정하고 각 Task에 적합한 정보로 Fine-Tuning 가능한 Embedding을 input 앞에 추가
    - task에 적합한 파라미터로 input앞에 임베딩 벡터 추가
    - 임베딩 벡터는 역전파 그래디언트 디센트로 학습되어 task에 적합해짐

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/8d01a924-2d21-4985-8232-17d89e43486a" width="400"/>
        
- **프롬프트 튜닝**
    - Continuous한 소프트 프롬프트
    - 입력 계층을 포함하여 앞에 추가된 prefix 활성화로 역전파하여 학습
    - "접두사 튜닝"(prefix-tuning)에 대한 간소화
        - prefix-tuning은 모든 trasformer의 layer앞에 prefix를 추가해야 해서 학습해야하는 파라미터의 양이 더 많다
    - 전체 사전 훈련된 모델을 동결하고 입력 텍스트 앞에 추가로 k개의 조정 가능한 임베딩 토큰만을 허용
    
- **디자인 이슈(하이퍼 파라미터)**
    - **초기 임베딩 프롬프트 벡터**
        1. 무작위 초기화
        2. 자주 사용되는 단어
        3. target Y에 등장하는 단어
    - **임베딩 토큰의 개수(프롬프트의 길이)**
        - 프롬프트가 짧을수록 조정해야 하는 새로운 파라미터가 적으므로, 잘 수행되는 최소 길이를 찾으려고 함

### 기존 연구와의 비교

- **다른 접근법과의 비교**
    - 접두사 튜닝(prefix-tuning)
        - 모든 트랜스포머 계층에서 앞에 추가되는 일련의 접두사를 학습
        - 따라서 학습 해야 하는 파라미터의 양이 많다
    - P-tuing
        - 학습 가능한 연속적인 프롬프트가 사람의 디자인을 기반으로 한 패턴을 사용하여 임베디드 입력 전체에 교차
        - 앞. 뒤로 프롬프트를 추가해야 하기 때문에 복잡하고 파라미터가 조금 더 많을 수 있음
    - 프롬프트 튜닝
        - 임베디드 입력에 앞에 추가되는 단일 프롬프트 표현을 사용
        - input 앞에 한 번만 프롬프트를 추가하면 돼서 파라미터 양이 적다
        
- **요약**
    - 기존의 pre trained 파라미터는 고정
    - input x 임베딩 앞에 학습 가능한 continuous 임베딩을 추가하여 각 task에 맞게 파인튜닝
