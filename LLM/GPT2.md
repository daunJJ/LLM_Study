## [Overview] GPT-1,2,3 정리
|  | GPT-1 | GPT-2 | GPT-3 |
| --- | --- | --- | --- |
| 학습 데이터셋 | bookCopus | web-text(Reddit) | common crawl |
| 파라미터  | 1.17억 | 15억 | 1750억 |
| 특징  | 대용량 데이터로 Unsupervised Pre-train 후 개별 Task에 Supervise fine-tuning을 하였더니 작은 지도학습 데이터셋으로도 성능이 좋음을 증명  | 별도의 fine-tuning 과정 없이도 Unsupervised Pre-training 만으로도 다양한 task에서 성능이 좋음을 증명 (zero-shot learning)  | 별도의 fine-tuning 과정 없이 Unsupervised Pre-training 후 In-context learning을 수행하여 예시의 개수가  많아질수록 성능이 좋아짐을 증명 |

## GPT-2(2019) 논문 리뷰

### **Intro**

**GPT-1과 비교** 

- GPT-1이 취한 방법론
    - unsupervised pre-training과 supervised fine-tuning을 진행하는 semi-supervised 방법론
- GPT-2가 취한 방법론
    - unsupervised pre-training만 수행하고, 개별 task에 맞게 fine-tuning하는 과정을 없앰

**GPT-2의 아키텍쳐**

- GPT-1과 크게 다르지 않지만, 파라미터의 규모만 확장
    - GPT-1(1.17억) → GPT-2(15억) → GPT-3(1750억)

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/f484c686-2a20-48f8-8e47-02c3bb479adc" width="350"/>

**핵심 Contribution**

- WebText와 같은 양질의 대량 데이터로 LM Pre-traning을 진행할 경우, Supervised Learning 없이 Zero-shot 환경에서 다양한 자연어 처리 Task해결이 가능함을 보여줌
    - 각 Task에 맞게 Supervised Learning(GPT-1)할 필요가 없음을 보여주는 범용적인 LM

### **Abstract**

- (질의 응답, 번역, 독해, 요약과 같은) 자연어 처리 작업은 일반적으로 특정 작업에 대한 데이터셋에서 지도 학습을 통해 접근
- 수백만 개의 웹페이지로 구성된 WebText에서 학습될 때, 언어 모델이 지도 학습 없이 다양한 자연어 처리 작업을 수행할 수 있음을 보여줌
- 제로샷 작업 전송(zero-shot task transfer) 방식
- 모델의 용량이 큰 것이 필수적 → 여전히 webText에 언더피팅되어 있음

### Introduction

- 과거 ML 시스템 접근 방식은 단일 도메인 데이터셋에서만 작업함으로써 일반화 부족
- 궁극적으로 모든 것에 유능한 LM을 만듦으로써, (수동으로 훈련 데이터셋을 생성하고 라벨을 붙여 지도학습 시키는) 좁은 분야의 LM을 만들 필요성을 없앰
- GPT-2는 제로샷 환경에서 파아미터나 구조 변경 없이도 수행 가능한 LM

### Approach

- 접근 방식의 핵심은 언어 모델링
- 심볼 시퀀스에 대한 결합 확률을 조건부 확률의 곱으로 인수분해

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/d4ea3da7-9533-49ae-be76-3ed50ce6ee28" width="250"/>
    
- 동일한 입력에 대해 다른 작업을 수행할 수 있어야 하므로 P(output | input, task)를 모델링해야 함
- Task에 대한 Description 문장을 맨 처음에 넣어준다
    - 예시: (한글로 번역, 여어, 한글)의 연속
    - 예시: (질문에 답변, 지시어, 질문, 답변)의 연속
- 충분한 용량을 가진 언어 모델은, 스스로 예측을 위해 자연어 시퀀스에서 보여진 작업을 추론하고 수행하기 시작할 것이라는 추측

### Training Dataset

- 품질에 중점을 둔 새로운 웹 스크랩 생성
    - 카르마 점수가 3점보다 높은 Reddit 4천 5백만개 스크랩
- Dragnet과 Newspaper 콘텐츠 추출기의 조합으로 HTML 응답에서 텍스트 추출
- 중복 제거와 휴리스틱 기반 정리 후 800만개 문서 (40GB 텍스트)
- 테스트 평가 작업과 겹치는 문제로 인해 학습에서 Wikipedia 문서 제거

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/7cbc3669-468d-4018-8936-ed9ccab3b523" width="350"/>

### Input Representation

- 일반적인 언어 모델(LM)은 어떤 문자열이든 확률을 계산할 수 있어야 함
- 현재 대규모 LMs에는 소문자화, 토큰화, 어휘 외 토큰과 같은 전처리
단계가 포함되어 있어, 모델링 가능한 문자열의 공간을 제한
- Byte Pair Encoding (BPE): 빈번한 단어 수준 입력과 드물게 나타나는 문자 수준 입력 사이를 효과적으로 보간

### Model

- Transformer
- GPT-1과 구조적인 모델은 거의 동일
    - 레이어 정규화 이동 및 추가
    - 모델 깊이와 잔여 레이어의 누적을 고려한 수정된 초기화: 잔여 레이어의 가중치를 1/square root(잔여 레이어의 수)로 스케일링
    - 어휘 (Vocab size) 50257개로 확장
    - 컨텐스트 크기를 512 → 1024로 토큰 확장

### Experiments

- 서로 다른 크기의 4가지 언어 모델을 훈련시키고 벤치마크
- 가장 작은 모델은 GPT-1 파라미터 수와 동일
- 두번째로 작은 모델은 BERT의 가장 큰 모델 파라미터 수와 동일
- 우리가 일반적으로 GPT-2라고 부르는 모델은 가장 큰 모델
- 그러나 모든 모델은 여전히 WebText를 충분히 학습하지 못함 → 파라미터 크기를 늘리면 학습률이 더 높아질 것이라는 의미

**Language Modeling**

- WebText 언어 모델(LM)이 제로샷 도메인 전환에서 어떻게 수행되는지 이해하는 데 관심
- 제로샷 설정에서 8개의 테스트 데이터셋 중 7개에서 state-of-the-art 성능 달성

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/72f8ed2b-76b9-4556-971c-19366070222d" width="400"/>


**작업**

- Children’s Book Test(CBT) 데이터셋
    - 생략된 단어에 대한 10개의 가능한 선택 중 어느 것이 옳은지 예측하는 작업
- LAMBADA 데이터셋
    - 텍스트의 장거리 의존성을 모델링하는 시스템의 능력을 테스트
    - 적어도 50개의 토큰 컨텍스트가 필요한 문장의 마지막 단어를 예측하는 것
- Winograd Schema Challenge 데이터셋
    - 텍스트의 모호성을 해결하는 능력을 측정함으로써 시스템의 상식적인 추론 능력을 측정
- Reading Comprehension (독해) 작업
    - Conversation Question Answering 데이터셋(CoQA)은 7개 다른 도메인으로부터의 문서와 문서에 대한 질문자와 응답자 사이의 자연어 대화 쌍
- Summarization (요약) 작업
    - CNN 및 Daily Mail 데이터셋(Nallapati et al., 2016)에서 GPT-2의 요약 능력을 테스트
- Translation (번역) 작업
    - WMT-14 프랑스어-영어 테스트 세트
    - GPT-2가 언어를 다른 언어로 번역하는 방법을 배우기 시작했는지를 테스트
    - 언어 모델을’영어 문장 = 프랑스어’ 문장 형식의 예제 쌍의 컨텍스트에 조건을 부여하고, 최종적으로’영어 문장 =’ 라는 프롬프트 이후에 우리는 모델에서 탐욕적인 디코딩을 사용하여 샘플링하고, 처음 생성된 문장을 번역으로 사용
- Question Answering(질의응답) 작업
    - Natural Questions 데이터셋 으로 테스트
    - 언어 모델의 컨텍스트는 데이터셋의 짧은 답변 스타일을 추론하는데 도움이 되는 예제 질문 답변 쌍으로 구성

### Generalization vs Memorization

- Memorization : 오버피팅되어 데이터를 모델이 그대로 암기하는 현상
    - 학습과 테스트 간의 중복이 존재하면 모델의 일반화 성능이 과다 보고됨
- 범용적인 성능에 대해, 메모라이제이션 문제인지를 확인해 봄
    - → 테스트 데이터가 학습 데이터에 얼마나 많이 나타나는지 WebText에서 확인해 봄
- 메모라이제이션 확인 과정
    - WebText 훈련 세트 토큰의 8-그램(8-grams)을 포함하는 블룸 필터를 만듦
    - 문자열은 소문자 영숫자 단어만 포함하도록 정규화되었고, 구분자로 단일 공백이 사용
    - 블룸 필터를 사용하여, 주어진 데이터셋에 대해 해당 데이터셋의 8-그램 중 WebText 훈련 세트에도 있는 비율을 계산
    - 일반 LM 데이터셋의 테스트 세트는 WebText 훈련과 1-6%의 중복을 가지며, 중복의 평균은 3.2%
    - 그러나 데이터셋이 자체 훈련 분할과 더 큰 중복을 가지며,평균 중복은 5.9% (그러니 WebText의 중복은 적은 편)

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/2ee55418-5aae-43ac-b2c4-f15dd15d33c9" width="400"/>

- 따라서, WebText에선 Memorization 문제 발생 확률이 상대적으로 낮다
- n-gram 중복 기반의 중복 제거를 중요한 검증 단계 및 정상성 확인으로 사용하는 것을 권장

### Discussion

- 비지도 작업 학습이 지도 학습의 한계를 넘는, 탐구할 가치 있는 추가적인 연구 영역이라는 것을 제안
- WebText 언어 모델의 제로샷 성능을 연구
- 독해에선 GPT-2의 성능은 제로샷 환경에서 지도학습 기준선과 경력이 있었지만, 요약과 다른 작업에서는 아직 기본적인 수준
- 언어 모델들은 충분한 용량이 있을 때만 기존 성능을 능가하기 시작
    - 성능의 상한이 어디에 있는지는 명확하지 않음

### Conclusion

- 충분히 크고 다양한 데이터셋에 대해 큰 언어 모델이 훈련되면, 이는 많은 도메인과 데이터셋에 걸쳐 잘 수행할 수 있게 됨
- GPT-2는 제로샷(zero-shot) 세팅으로 8개의 테스트된 언어 모델링 데이터셋 중 7개에서 state-of-the-art 성능을 달성
- 충분히 다양한 텍스트 말뭉치에서 훈련된 대용량 모델이 명시적인 지도학습 없이도 놀라운 양의 작업을 수행하는 방법을 배우기 시작한다는 것을 제안
