## [Overview] GPT-1,2,3 정리
|  | GPT-1 | GPT-2 | GPT-3 |
| --- | --- | --- | --- |
| 학습 데이터셋 | bookCopus | web-text(Reddit) | common crawl |
| 파라미터  | 1.17억 | 15억 | 1750억 |
| 특징  | 대용량 데이터로 Unsupervised Pre-train 후 개별 Task에 Supervise fine-tuning을 하였더니 작은 지도학습 데이터셋으로도 성능이 좋음을 증명  | 별도의 fine-tuning 과정 없이도 Unsupervised Pre-training 만으로도 다양한 task에서 성능이 좋음을 증명 (zero-shot learning)  | 별도의 fine-tuning 과정 없이 Unsupervised Pre-training 후 In-context learning을 수행하여 예시의 개수가  많아질수록 성능이 좋아짐을 증명 |

## GPT-1(2018)

**Overview**

- generative pre-training을 통해서 대량의 corpus 데이터를 학습시킨 뒤 개별 task에 fine-tuning
- 단일 모델로 여러 개의 sub task를 수행 가능하다는 것을 발견
- 이미지처리에서 유행하던 <대량의 데이터로 pre-training한 단일 모델 → 개별 task에 맞게 fine-tuning>을 NLP에 도입한 최초의 논문

**Architecture**

- Transformer 디코더를 12depth 쌓음
- 분류, 함의 인식, 유사성, 다중 선택 등의 서로 다른 Task들은 원하는 Input형태가 다름 → Task Specific Transformation을 거쳐 동일한 모델에 넣을 수 있도록 Input 변형 작업을 거침

**Parameter**

- GPT-1(1.17억) → GPT-2(15억) → GPT-3(1750억)

## GPT-1(2018) 논문 정리

### **Intro**

- GPT는 Generative Pre-trained Transformer의 약자
- GPT-1은 OpenAI에서 2018년에 개발한 최초의 GPT모델
- 기반 논문: Improving language understanding by generative pre-training.

### **핵심 Contribution**

- 하나의 모델로 다양한 자연어처리 Task에 대해서 state-of-the-art 성능을 보여주는 모델
    - 대규모 unlabeled 데이터셋을 이용한 Pre-Training
    - 개별 Task 에 맞게 Fine-Tuning

### **Abstract**

- GPT-1은 특정 task를 목적으로 하지 않고, 일반적인 NLP 모델로
task-agnostic(초경험적인 것의 본체는 인식 불가능)하게 학습
- unlabeled 텍스트 묶음 데이터는 풍부한 반면, 각각의 개별 task에 대해서 labeled된 데이터는 희소
- GPT-1은 generative pre-training로 먼저 학습하고 개별 task에 대해서 fine-tuning할 경우 각각의 개별 task에 대해 큰 성능 갭을 보여주면서 state-of-the-art를 개선할 수있음을 증명
- 각각의 task에 최적화된 형태의 모델과 많은 튜닝이 들어간 모델들을 이기고 12개의 subtask중에서 9개에서 state-of-the-art 성능

### **자연어 처리 Task**

- Text  Entailment  (텍스트 함의인식)
    - 텍스트간의 의미관계를 파악하는 문제영역
    
    - 비교하는 텍스트를 Premise p와 hypothesis h
    - 두 텍스트 간의 관계에 대한 정답 레이블은 entailment (함의), contradiction(모순) or neutral(중립)
- Semantic Similarity
    - 두 문장이 동일한 의미를 가지고 있는지를 판단하는 NLP 문제영역
    - 정답 레이블은 same, not same 이거나 두문장간의 유사도 점수를 예측
- Reading Comprehension
    - 텍스트를 읽고 텍스트 안의 내용을 이해하는지를 테스트하는 NLP 문제영역
- Commonsense Reasoning
    - (인간이 매일 마주하는 일상적인 상황의 유형과 본질을 추측하는 인간과 같은) 상식 능력을 평가하는 문제영역
    - 물리적 대상의 특성, 분류학적 속성 및 사람들의 의도에 대한 판단이 포함
- Sentiment Analysis
    - 텍스트에 포함되어 있는 감정 상태를 맞추는 문제영역
- Linguistic Acceptability
    - 텍스트가 문법적으로(grammatical) 올바른지를 테스트하는 NLP 문제영역

### **Introduction**

- unlabeled 텍스트에서 학습하기
    - subtask에 대한 supervised learning 과정을 완화하려면 raw text로부터 효율적으로 학습하는 방법이 NLP에서 필수적
    - unlabeled 데이터로 모델을 학습시킬 수 있다면 많은 시간과 비용이 소요되는 labelling 작업을 건너 뛸 수 있음
    - 완전히 supervised learning을 대체하지 않더라도 supervised learning을 수행할 때 효율적인 feature로 활용할 수 있음
- unlabeled 텍스트에서 학습이 어려운 이유
    - 어떤 손실함수가 다양한 task에 효율적으로 transfer 할 수 있는 목적함수인지 불분명
    - general tsak에 대해 학습된 표현을 각각의 target task에 맞게 변형하는 과정이 불분명
- GPT-1이 취한 방법론
    - unsupervised pre-training과 supervised fine-tuning을 진행하는 semi-supervised 방법론
    - GPT-1은 대량의 unlabeled 텍스트 데이터에 접근 가능하고, 수동으로 labelling된 target task에 맞는 데이터가 있는 상황을 가정
    - 다양한 Task에 대해 잘 동작하고 Long-Term context를 잘 기억하는 것이 검증된 Transformer 모델을 사용
    - 모델 구조 변경을 최소화하기 위해서 모든 task를 구분 기호(delimiter token)를 포함한 하나의 연속된 텍스트로 변경해서 학습

### **Framework**

- 트레이닝 과정의 2가지 stage
    1. 첫번째 stage는 대량의 텍스트로 높은 범용성의 language model을 학습시킴
    2. 두번째 stage는 labeled 데이터를 이용해서 각각의 task에 맞게 fine-tuning을 진행
- Unsupervised pre-training
    - unlabeled 토큰 corpus 𝒰 = {𝑢!, … , 𝑢"}, 에서 표준 적인 language modelling 목적함수인 아래 수식을 최대화하는 방향으로 학습

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/c5c97d75-166d-40cf-8edc-09dc187932b1" width="300"/>

    - k는 context window 크기, P는 학습된 뉴럴 네트워크의 파라미터 Θ에 기반한 conditional probability
    - 학습을 위해 Transformer의 변형인 multi-layer Transformer decoder를 사용
- Supervised fine-tuning
    - 𝐿1손실함수를 통해서 unsupervised pre-training을 마친 후 각 target task에 맞게 supervised learning을 진행
    - labeled 데이터셋 C는 인풋 토큰들의 시퀀스 𝑥1, 𝑥2, . . , 𝑥m과 label y로 구성
    - 인풋을 pre-trained 모델에 넣고, 마지막 transformer block의 활성값(activiton)인 ℎ 을 얻고 이는 마지막 linear layer 파라미터인 𝑊와 함께 target y를 예측

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/85c1593f-5b26-414d-b8e0-68d5646b6293" width="300"/>

    - Supervised Learning에서 최대화하려는 목적함수는 다음과 같음

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/5b5135c1-26ba-4dd3-99c1-493b213daf60" width="260"/>

    - 또한 fine-tuning 과정에서 아래와 같이 Language Modelling 목적함수를 추가적인 Auxiliary 목적함수로 지정해 줄 경우 다음과 같은 장점을 얻을 수 있음

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/cbf21cdb-b17f-4e62-a64d-782af3dcd755" width="240"/>
 
        1. Supervised 모델의 일반화(generalization) 성능을 향상
        2. 수렴 속도를 향상
- 학습을 위한 Transformer 모델 구조

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/66a48be3-74b5-40a5-83b7-0d54b7ca26481" width="150"/>

    - Input: Text 임베딩 + Positional 임베딩
    - Output: LM의 Prediction 확률값 + Sub Task에 대한 정답을 예측

### **Task -specific input transformations**

- Input의 변환이 필요한 이유
    - 몇몇 task, 예를 들어, text classification 같은 경우, 아무런 수정없이 이전에 언급한 방식으로 바로 fine-tuning을 진행
    - question answering, text entailment와 같은 task는 정렬된 문장 쌍 (sentence pairs)나 (document, question, answer)의 triplets로 구성
    - 따라서 pre-trained 모델이 연속된 텍스트에 대해 학습되었으므로 이런 task에 적용하기 위해서는 약간의 변환 과정이 필요
- GPT-1은 traversal-style approach 취함
    - 각각의 task 특성에 맞게 구조화된 input을 pre-trained 모델 구조가 처리할 수 있는 연속된 텍스트로 변경
    - 모든 변환은 임의로 초기화된 (randomly initialized)과 시작 토큰(start token) 종료 토큰(end token) < 𝑠 >, < 𝑒 > 을 추가
- 개별 Sub Task에 대해 변환 과정 살펴보기
    - Textual  entailment
        - premise 𝑝와 hypothesis ℎ를 delimiter token ($)를 사이에 넣고 이어줌
    - Similarity
        - 2가지 문장(A,B) 의 순서가 중요하지 않기 때문에, 모델이 Order에 대해 잘못된 지식을 학습하지 않도록 2가지 쌍을 만듦
        - A-B 한 쌍, B-A 한 쌍
    - Question Answering and Commonsense Reasoning
        - QA: document-question-answer
            - answer의 후보군이 많으므로 후보군 마다 쌍을 만듦
        
        <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/72b7e535-9a3e-4e6d-aa51-e11c0a68ed4b" width="400"/>


### **Experiment(실험 환경)**

- Unsupervised pre-training
    - BookCorpus 데이터셋을 사용
    - long-term information을 잘 학습할 수 있도록 연속적인 긴 텍스트로 구성
- Model specifications
    - 모델은 original Transformer에 기반
    - 12개의 레이어를 가진 masked self-attention heads를 포함한 decoder-only transformer를 사용
    - Position-wise feed-forward 네트워크는 3072 차원의 Inner states를 사용
    - 최대 learning rate가 2.5e-4인 Adam optimizer를 사용
    - learning rate는 0에서 첫 2000 step까지 선형적으로 증가시키고, 0까지 cosine schedule로 annealing
    - 512개의 토큰으로 구성된 연속된 sequence를 랜덤하게 64개씩 뽑아 minibatch를 구성하고 100번의 epoch동안 학습
    - BPE(Byte Pair Encoding)을 이용하여 4만번의 merge를 통해 vocabulary 생성
        - Residul Embedding과 attention의 dropout은 0.1
    - L2 정규화 텀 이용
    - 활성화 함수에 Gaussian Error Linear Unit(GELU) 이용
- Fine-tuning details
    - unsupervised pre-training 과정에서 사용한 하이퍼파라미
    터를 다시 사용
    - classifier에 dropout rate는 0.1을 적용
    - 대부분의 Task 에 learning rate 6.25e-5와 32의 batch size를 사용
    - 대부분의 경우 3 epochs의 빠른 fine-tuning으로 충분

### Supervised fine-tuning 성능

- Natural Language Inference

    <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/86982a50-e44a-46e8-848f-693593cb6b55" width="330"/>
  
- Question answering and commonsense reasoning

    <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/11a3cf15-4a89-437d-a6ba-a7099e14a2e5" width="330"/>

- Semantic Similarity & Classification

    <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/3d3834a5-2606-4c95-95b0-362738271909" width="330"/>
