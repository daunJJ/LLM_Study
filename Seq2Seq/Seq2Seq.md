## Seq2Seq Learning

### **Input/Output**

- Input: sequence of items(단어, 문장, 이미지의 feature 등)
- Output: 다른 sequence of items
    - 입, 출력의 아이템의 개수는 같을 필요 X

### **Main Idea**
 <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/5d01e3ea-0661-4a44-bc78-8fd72ae02178" width="360"/>

- Encoder : 입력 정보(sequence of items) → 처리 → Context Vector → 저장
- Decoder: 압축된 정보(Context Vector) → 처리(item by item) → 출력

### **Encoder-Decoder**

- 사용 모델:
    - Encoder와 Decoder에는 RNN이 쓰임
- 작동 방식:
    - Encoder에 입력이 하나씩(단어 구분) 들어올 때마다 hidden state가 한번씩 업데이트
    - 가장 마지막 hidden state가 context vector 가 되어 Decoder로 전달
    - Decoder는 해당 hidden state로 하나씩(단어 구분) Output 생성
- 문제점:
    - 가장 마지막 hidden state는 뒤쪽 아이템의 영향력이 커지고, 앞 쪽 아이템의 영향력이 적어짐 ⇒ 장기 의존성 문제
    - LSTM과 GRU가 개발됨 ⇒ 장기 의존성을 완벽하게 해결하는 것은 아님
    - 이를 해결하기 위해 Attention 등장

### **Attention**

- 원리:
    - 모델이 각각의 Input sequence 중 현재의 Output Item이 주목해야하는 part에 가중치를 주어 해당 part에 집중하도록 함
- 종류:
    - Bahadanau attention
        - 어텐션 스코어 자체를 학습하는 뉴럴 네트워크 모델
    - luong attention
        - current state와 과거 hidden state의 유사도를 측정하여 어텐션 스코어를 만듦 (따로 학습 X)
    - 성능이 비슷하여, 따로 학습 시키지 않는 attention인 Luong이 사용하기 더 좋음

### **Attetion in Seq2Seq**

 <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/34aab46c-e27a-4a20-b779-18dcccf8e240" width="450"/>

- 전체 hidden state를 디코더에 넘겨줘서 디코더에 더 많은 정보를 주게 됨
- 디코더는 필요한 hidden state를 취사 선택하여 가중치를 부여해 사용
- 디코더는 output을 생성하기 전에 추가적인 작업 수행

     <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/809b2eda-e776-4845-b6eb-cc815bd9fd90" width="300"/>
    
    1. Encoder hidden state를 모두 확인 (각각의 hidden state는 해당하는 단어의 영향력을 가장 크게 받음) <4x3>
    2. 각각의 hidden state에 가중치(스코어) 부여 = 어텐션 스코어 <1 x 3 >
        
        = 인코더 hidden state와 디코더 hidden state를 내적한 값 (디코더에 미치는 영향력이 크면 스코어 커짐) 
        
    3. softmax를 이용하여 스코어를 정규화 <1 x 3> 
    4. 각각의 Input hidden state 벡터를 softmax 스코어와 결합 = 해당 step의 Context Vector <4 x 1>
    5. Context Vector 와 디코더의 hidden state를 Concat 
    
    ⇒ 이를 FeedForward 뉴럴 네트워크에 거치면 최종 Output 
    
