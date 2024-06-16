## [Overview] GPT-1,2,3 정리
|  | GPT-1 | GPT-2 | GPT-3 |
| --- | --- | --- | --- |
| 학습 데이터셋 | bookCopus | web-text(Reddit) | common crawl |
| 파라미터  | 1.17억 | 15억 | 1750억 |
| 특징  | 대용량 데이터로 Unsupervised Pre-train 후 개별 Task에 Supervise fine-tuning을 하였더니 작은 지도학습 데이터셋으로도 성능이 좋음을 증명  | 별도의 fine-tuning 과정 없이도 Unsupervised Pre-training 만으로도 다양한 task에서 성능이 좋음을 증명 (zero-shot learning)  | 별도의 fine-tuning 과정 없이 Unsupervised Pre-training 후 In-context learning을 수행하여 예시의 개수가  많아질수록 성능이 좋아짐을 증명 |

## GPT-3 (2020) 논문 리뷰

### Abstract

- 소수의 예시(few-shot examples)를 사용한 In-context Learning으로 성능을 크게 향상
- 이전의 작업별 fine-tuning과 경쟁력을 갖춤
- 1750억 개 파라미터로 이전의 LM 보다 10배 많음

### Introduction

- 이전 접근법의 한계는 작업별 미세 조정이 필요하다는 것
    - 레이블된 학습 데이터의 필요는 언어 모델의 적용 가능성을 제한함
    - 모델이 좁은 훈련 분포에 지나치게 특화되어 일반화되지 못함
    - 수집이 어려운 지도 학습을 위한 감독 데이터 셋을 필요로 함
- 메타 학습: 언어모델이 Gradient Discent로 학습을 다시 하는 것이 아닌, 새로운 패턴 인식 능력을 개발
    - 이를 문맥 내 학습(Incontext Learning)을 통해 시도
    - 사전 학습된 언어 모델의 텍스트 입력을 작업 명제의 한 형태로 사용
- Incontext learning의 종류
    - 제로샷: 지시만
    - 원샷: 한 가지 예시
    - 퓨샷: 몇 가지 예시
- GPT-3 논문이 In-context Learning을 최초로 증명
- GPT-3를 3가지 조건에서 평가
    - 퓨샷 학습
    - 원샷 학습
    - 제로샷 학습
- 예시 개수가 높아질수록 성능이 좋아짐

### Approach

- Fine-tuning에서 Prompt Engineering으로 컨셉 변경

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/0e1d7447-e4f3-40d4-9827-410b3e652185" width="400"/>
    
- 모든 작업마다 새로운 대규모 데이터셋이 필요하다는 FT의 단점을 극복
- FS(Few Shot)는 가중치 업데이트 없이 추론 시점에 작업의 몇 가지 시연을 조건으로 제공받는 설정
    - 특정 작업 데이터에 대한 필요성이 크게 줄어듬
- 1S(One Shot)은 작업의 자연어 설명과 함께 오직 하나의 시연만 허용된다는 점이 다름
    - 인간에게 전달되는 방식과 가장 밀접
- 0S(Zero Shot)은 시연이 사용되지 않으며 오직 작업을 설명하는 자연어 지시문만 제공
- 퓨삿 결과들이 state-of-the-art 파인튜닝 모델에 조금 뒤처지는 정도

### Model and Architectures

- GPT-2와 동일한 모델과 아키텍쳐
- 1억 2500만개 파라미터 → 1750억 개 파라미터로 3개의 크기 등급에서 8가지 다른 크기의 모델 훈련
    - n_params: 훈련 가능한 매개변수의 총 수
    - n_layers: 총 레이어 수
    - d_model: bottleneck 레이어의 유닛 수
    - d_head: attention head의 차원
    - n_ctx=2048 토큰의 context windows

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/d0d15cab-bfa5-4796-ab47-5bec267bd6ba" width="400"/>

    - 이중 가장 큰 모델이 일반적인 GPT-3 모델

### Training Dataset

- 약 1조 단어에 달하는 common crawl 데이터셋
    - 고품질 참조 코퍼스 범위와의 유사성을 기반으로 필터링
    - 데이터셋 내외에서 문서 수준에서 퍼지 중복 제거
    - 알려진 고품질 참조 코퍼스를 훈련 혼합물에 추가

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/0fb662f8-9c0e-4337-97a8-ebdca7960794" width="400"/>

- 훈련 세트가 테스트 세트에 발견되는 데이터 오염을 줄이기 위해, 모든 벤치마크의 개발 및 테스트 세트와의 중복을 찾아 제거하려고 시도

### Evaluation

- 개발 세트에서 K = 10 ~ 100개의 예시로 실험 후 가장 좋은 값을 테스트 세트에서 실행
- P(완성|맥락) / P(완성|답변 맥락)을 계산 

### Result

GPT-3를 오픈 벤치마크 데이터셋에 대해서

개별 데이터셋에 대해 fine-tuning하지 않고 전체 대규모 데이터셋에 대해 pre-traning 후 few-shot 예제만 몇 가지 주어지는 접근법으로 

기존의 state-of-the-art finetuning 모델과 비슷한 성능을 보임을 증명
