## Prefix-Tuning 기법

- 핵심 idea

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/6c65481a-b2a7-4001-a9b7-755c76847b41" width="400"/>

    - fine-tuning 과정시 pre-train이 끝난 파라미터 w_0를 고정하고 Prefix-tuning 세팅의 새로운 파라미터를 추가하여 학습 → 각 서브 태스크 별로 prefix만 재학습
- 소개
    - 언어 모델의 매개변수를 고정시키고 작은 연속적인 작업별 벡터만을 최적화
    - 이후 토큰이 해당 접두사에 ‘가상 토큰’처럼 접근할 수 있게 함
    - 학습 가능한 가상의 벡터를 추가하는데, 이는 LLM의 output에 대한 힌트를 제공..?
    - LoRA처럼 모듈식이라, pre-trained된 LM을 fix해둔 뒤 prefix만 서브 태스크에 맞게 학습하여 각각 처리
- 다른 경령화된 세부 조정 방법
    - 어댑터 튜닝: 사전 훈련된 언어 모델의 레이어 사이에 추가적인 작업별 레이어를 삽입
    - 인컨텍스트 러닝 또는 프롬프팅: 작업 입력에 대한 몇 가지 예시를 붙여 LM에서 출력을 생성
- 장점
    - prefix만 교체하는 방식으로 하나의 LM은 한 번에 여러 작업을 지원
    - 하나의 배치에서 여러 사용자/작업의 예시를 처리하는 것도 가능
    - full fine-tuning보다 1000배(0.1%) 적은 파라미터 사용
- 학습 상황
     <img src="hhttps://github.com/daunJJ/LLM_Study/assets/109944763/25e46b6f-98b9-442e-b7b2-a9823015d92a1" width="500"/>
     
    - Autoregressive model의 경우 prefix를 하나만 붙이고
    - Encoder, Decoder 모델의 경우 각각 하나씩 붙임
- prefix-tuning
     <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/7a56e250-c4af-41ec-8f05-0858fb5a133c" width="200"/>

    - 지시사항을 연속적인 단어 임베딩(continous word embedding)으로 최적화
    - prefix-tuning은 접두사의 모든 레이어를 최적화
    - 언어 모델 매개변수 φ는 고정되어 있고, 접두사 매개변수 θ만이 훈련 가능한 매개변수
    - P새타 행렬을 재매개변수화(reparameterization)하여 성능 개선
- prefix-tuning의 하이퍼 파라미터
    - prefix length에 따라 성능 달라짐
- 단점
    - 원래 input의 앞에 prefix를 concat하여 넣는 것이기 때문에 input 토큰 길이 제한의 제약을 받음
- 정리

     <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/a243b07e-f633-4c46-8e9e-757bd7dd4502" width="450"/>
