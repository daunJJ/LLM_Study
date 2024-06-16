## Chain-of-Thought Prompting

**Overview**

- Chain-of-Thought Prompting의 핵심 idea : 프롬프트에 중간 설명과정을 넣어줘서 기존에 LLM이 잘 풀지 못했던 산술연산, 상식추론, 상징 추론 과정을 잘 풀게 만듦
    
    <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/941e5e9a-24c2-4efd-aaeb-b3b99554382d" width="400"/>

**Abstract**

- 사고의 연쇄 생성(chain of thought)이 큰 언어 모델의 복잡한 추론(complex reasoning) 수행 능력을 크게 향상시키는 방법을 탐구
- 사고의 연쇄 프롬프팅(chain-of-thought prompting)이라는 단순한 방법을 통해 이러한 추론 능력이 자연스럽게 나타나는 방법을 보여줌

**Introduction**

- 모델 크기만 확대하는 것만으로는 산술, 상식, 기호적 추론과 같은 도전적인 작업에서 높은 성능을 달성하는 데 충분하지 않았음
    1. 산술적 추론(arithmetic reasoning) 기법: 최종 답변으로 이어지는 자연어 근거를 생성함으로써 혜택을 얻음
    2. 큰 언어 모델의 프롬프팅을 통한 문맥 내 몇 번의 학습(few-shot learning via prompting)이라는 흥미로운 전망을 제공
  
  ->  새로운 작업에 대한 별도의 언어 모델 체크포인트를 미세 조정하는 대신, 모델에 몇 가지 입력-출력 예시를 통해 작업을 시연하여 "프롬프트" 할 수 있음

- 그러나 제한 사항 존재
    1.  고품질의 근거로 큰 데이터 세트를 만드는 것은 일반적인 기계 학습에서 사용되는 간단한 입력-출력 쌍보다 훨씬 복잡
    2. 몇 번의 프롬프팅 방법(few-shot prompting)은 추론 능력을 필요로 하는 작업에서 잘 작동하지 않음 
- <입력(input), 사고의 연쇄(chain of thought), 출력(output)>으로 구성된 프롬프트가 주어진 추론 작업에 대한 몇 번의 프롬프팅(few-shot prompting) 수행 능력을 탐구
- few-shot 예제로 chain of thought 사용
  
    <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/ee0b2e1d-406e-46ce-8f4e-109a07ef48a4" width="400"/>
    
- 산술(arithmetic), 상식(commonsense) 및 기호적 추론(symbolic reasoning) 벤치마크에서의 경험적 평가를 제시
- (chain-of-thought prompting)이 표준 프롬프팅보다 우수한 성능을 보이며 때로는 눈에 띄게 우월함

**Chain-of-Thought Prompting**

- 자신의 사고 과정(문제를 중간 단계로 분해하고 각 단계를 해결한 후 최종 답변을 제공하는 것), 논문의 목적은 언어 모델에 비슷한 사고의 연쇄를 생성하는 능력을 부여하는 것
- 사고의 연쇄 프롬프팅은 언어 모델에서의 추론을 촉진하기 위한 접근법
    1. 모델이 여러 단계의 문제를 중간 단계로 분해할 수 있게 해줌
    2. 모델의 행동에 대한 해석 가능한 창을 제공하여 어떻게 특정한 답에 도달했는지 제안하며, 추론 경로가 어디서 잘못되었는지 디버깅할 기회를 제공
    3. 수학 단어 문제, 상식 추론, 기호 조작과 같은 작업에 사용될 수 있음
    4. 사고의 연쇄 시퀀스의 예시를 몇 번의 프롬프팅 예제에 포함시킴으로써 쉽게 유도

**Experimental Setup**

- 다섯 가지 수학 단어 문제 벤치마크를 고려
    -  수학 단어 문제의 GSM8K 벤치마크 (Cobbe 등, 2021)
    -  다양한 구조의 수학 단어 문제 SVAMP 데이터셋 (Patel 등, 2021)
    -  다양한 수학 단어 문제의 ASDiv 데이터셋 (Miao 등, 2020)
    -  대수학 단어 문제의 AQuA 데이터셋
    -  MAWPS 벤치마크 (Koncel-Kedziorski 등, 2016)
- 사고의 연쇄 프롬프팅(Chain-of-thought prompting)
    - 프롬프팅을 위한 8개의 몇번의 프롬프트 예제와 함께 사고의 연쇄를 수동으로 구성
    - AQuA를 제외한 모든 벤치마크에 대해 이 8개의 사고의 연쇄 예제 한 세트를 사용( AQuA는 자유 응답 대신 다중 선택이기 때문)
- 언어 모델(Language models)
    - GPT-3
    - LaMDA
    - PaLM
    - UL2 20B
    - Codex

**Results**

- 사고의 연쇄 프롬프팅이 모델 규모의 출현하는 능력(emeregent ability)임을 보임 = 대규모 LLM 에서만 발생
    - 작은 모델의 성능에 긍정적인 영향을 미치지 않으며, 약 100B 매개변수의 모델과 함께 사용될 때만 성능 향상을 가짐
- 사고의 연쇄 프롬프팅은 더 복잡한 문제에 대해 더 큰 성능 향상
- GPT-3 175B와 PaLM 540B를 통한 사고의 연쇄 프롬프팅은 일반적으로 라벨링된 훈련 데이터셋에서 작업 특정 모델을 미세 조정하는 기존의 최고 기술과 유리하게 비교

**Conclusion**

- 산술, 기호, 및 상식 추론와 같은 어려운 추론 문제에 대하여 사고의 연쇄 프롬프팅(chain-of-thought prompting)을 통해 few shot 예제를 보여주면, 대규모 LLM에 대해 극적인 성능 향상을 불러온다
- 그러나, 프롬프트를 수동으로 작성해야 하기 때문에 예제 생성이 어렵다
