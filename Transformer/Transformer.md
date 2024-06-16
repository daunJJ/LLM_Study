## 트랜스포머 이전의 딥러닝

### 딥러닝 기반 기계 번역 발전 과정

- 21년 기준 최신 고성능 모델들은 Transformer 아키텍처를 기반으로 함
    - GPT: 트랜스포머 디코더 사용
    - BERT: 트랜스포머 인코더 사용

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/ee13f02a-f493-4f76-8864-6193e652b2f1" width="450"/>
    

### 기존 Seq2Seq 모델들의 한계점

- 기존 Seq2Seq 모델들의 한계점
    - context vector v (고정된 크기)에 소스 문장의 정보 압축
        
        ⇒ 병목(bottleneck)이 발생하여 성능 하락의 원인이 됨

        <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/986a91e3-448d-4ddb-a84d-e191289972b5" width="450"/>

    - [인코더]: 단어들이 입력될때마다 이전 단어들의 정보를 가지는 hidden state 값을 업데이트 ⇒ 마지막 단어의 hidden state는 전체 문장을 대표하는 context vector로서 사용
    - [디코더]: context vector로부터 hidden state 업데이트 ⇒ 앞출력 단어가 들어올 때마다 hidden state 업데이트 ⇒ eos(end-of-sequence)가 나올 때까지 반복
- 발전) 디코더가 context vector를 매번 참고하게 함
    - RNN을 거치며 정보가 손실되는 정보를 줄이므로 성능이 향상됨
        
        ⇒ 다만 여전히 소스 문장을 하나의 벡터에 압축해야 함

        <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/d4f2b6cd-a218-4ac0-9120-e583d678ad6d1" width="450"/>        

### Seq2Seq with Attention

- [문제 상황] 하나의 context vector가 소스 문장의 모든 정보를 가지고 있어야 하므로 성능 저하
- [해결 방안] 매번 소스 문장에서의 출력 전부를 입력으로 받음
    - 최신 GPU의 많은 메모리와 빠른 병렬 처리로 가능
- Seq2Seq 모델에 어텐션 매커니즘 사용
    - 입력 시퀀스의 매 단어마다의 hidden state를 전부 디코더에 전달
    - h와 w의 행렬곱으로 에너지를 생성 = 어떤 단어에 가중치를 둘지 결정하는 역할
  
      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/3eb6c08c-687b-4d5f-b98a-ebadc08e17c0" width="400"/>    

### Seq2Seq with Attention : 디코더

- 디코더에서 출력 단어를 만드는 방법

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/106dabf3-f273-4f43-b442-dd47c10ecd061" width="450"/>

    - i: 현재의 디코더가 처리 중인 인덱스
    - j: 각각의 인코더 출력 인덱스
    - 에너지: 매번 디코더가 출력 단어를 만들 때마다 모든 j를 고려함 = 인코더의 모든 출력을 고려함
        - Si-1: 디코더가 이전에 출력한 단어를 만들기 위해 사용한 hidden state
        - hj: 인코더 파트의 hidden state
        
        ⇒ 디코더 파트에서 이전에 출력했던 정보(=S)와 인코더의 모든 출력 값(=h)을 비교하여 Energy를 구함 = 어떤 h값과 가장 많은 연관성을 가지는 지를 구하기 위한 용도 
        
    - 가중치: 에너지 값에 Softmax를 취한 값 = 비율적으로 어떤 h와 연관성이 큰지를 구함
    - Ci: 가중치 값을 실제 h값과 곱하여 가중치가 반영된 인코더 출력 결과를 더함 → 인코더의 입력으로 사용
- 어텐션 가중치를 통해 각 출력이 어떤 입력 정보를 참고했는지 알 수 있음

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/1676ad27-e452-4bab-bafc-6ed708a9ab1f" width="450"/>    

## 트랜스포머 기본 구조

### Tranformer

- 딥러닝 기반 자연어 처리 네트워크에서 핵심이 되는 논문
    - 논문명: Attention is All you need
- RNN이나 CNN을 필요로 하지 않음
    - 대신 Positional Encoding 사용 ⇒ 순서에 대한 정보를 주는 역할
- 인코더와 디코더로 구성
    - Attention 과정을 여러 레이어에서 반복
    - 인코더의 역할: 입력 시퀀스의 주요 특징을 추출하여, 이를 압축된 형태의 벡터로 표현 = Context vector
    - 디코더의 역할: 인코더가 생성한 고차원 표현을 입력으로 받아, 목표 시퀀스를 생성 = 출력 값
 
      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/e37b8746-08e3-4766-87b1-ed4bc0df09dd" width="200"/>

      Incoder와 Decoder를 Nx번 반복 
    

### 트랜스포머 동작 원리: 입력 값 임베딩

- Embedding
    - 단어 → Continuous한 실수 벡터로 표현
    - 문장 → Embedding Matrix로 표현

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/06418ac0-f449-415d-ad19-4ea6c1b5f9c5" width="450"/>    

      embedding dimension은 정하기 나름 
    
- Positonal Encoding
    - RNN을 사용하지 않으려면 위치 정보를 포함하고 있는 임베딩 사용
    - Input Embedding Matrix와 같은 크기(dim)에 위치 정보를 가지는 Positional Encoding을 더하여 각 단어의 순서 정보를 나타냄
      
      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/fb1f70e4-ec3b-4393-9d1b-6e4ddca934e0" width="300"/>        

### 트랜스포머 동작 원리: 인코더

- 임베딩이 끝난 이후에 어텐션 진행
    - 각 단어들의 연관성을 학습 = 문맥에 대한 정보를 학습
- 잔여 학습(Residual Learning) 사용
    - 특정 layer를 건너 뛰어서 입력할 수 있도록 연결
    - 기존 정보를 입력하면서 + 추가적으로 잔여된 부분만 학습
       
      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/578b2d9d-2d55-4a82-aa21-60b320712139" width="300"/>

- 어텐션(Attention)과 정규화(Normalization) 과정 반복
    - 각 layer는 서로 다른 파라미터를 가짐

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/e6e516be-47e0-45ac-a978-1f7a4fbd4afd" width="300"/>        

### 트랜스포머 동작 원리: 인코더와 디코더

- 동작 방식
    - 입력 값 → 여러 개의 인코더 레이어를 거침
    - 인코더의 마지막 출력 → 여러 개의 모든 디코더 레이어에 입력되어 최종 출력 생성

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/28f65731-e173-41d1-a4a2-89fa25781a0a" width="450"/>

    - 하나의 디코더 레이어에서는 2개의 어텐션 사용
        - 1번째 어텐션: 서로의 출력 단어들이 어떤 가중치를 가지는지 구함 = self attention
        - 2번째 어텐션: 각각의 출력 단어가 인코더의 정보(소스 문장에서의 정보)와 어떤 가중치를 가지는지 구함 = incoder-decoder attention
    - 마지막 인코더 레이어의 출력은 모든 디코더 레이어에 입력됨 
 
      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/c3308755-9d39-4bda-8832-8a928cd4afe4" width="400"/>

- 특징
    - RNN을 사용하지 않으면서 인코더와 디코더를 다수 사용한다는 점
        - 입력 단어 자체가 연결되어서 한번에 입력 → 한번의 어텐션
        - 즉 위치 정보를 여러 개의 hidden state를 거치지 않고도 한번에 입력 가능
    - <eos>가 나올 때까지 디코더 이용
        - 디코더는 RNN과 마찬가지로 단어 개수만큼 사용해야 함

### 트랜스포머 동작 원리: 어텐션 (Attention)

- Scaled Dot-Product Attention

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/8803f1dc-7636-419f-b23d-fdd65dfe7293" width="450"/>

    - 쿼리: 물음의 주체
    - 키: 질문을 당하는 대상
        - 예시: I (쿼리) / am a teacher (키)
        - 각각의 키에 대해 어텐션 스코어를 구하는 방식
    - 값: 곱해지는 대상
        - 어텐션 스코어를 value와 곱해서 attention value가 됨
- Multi-Head Attention
    - 입력 문장에 대한 V, K, Q가 h개의 서로 다른 어텐션 컨셉을 수행하도록 함
    - 다양한 특징을 학습 가능
- 특징
    - Attention의 입력 값과 출력 값의 dimention을 같도록 함 ⇒ 인코더, 디코더 레이어를 중첩해서 사용 가능
    - 모든 Muti-head Attention의 동작 방식는 같고, Q, K, V만 달라짐
- 실제 수학식

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/0c2d0129-921d-4332-9c54-b529fa6ba0a8" width="450"/>    

### 트랜스포머의 동작 원리: Attention (행렬)

1) 각 단어에 대해 쿼리, 키 , 값을 생성 

- 임베딩 차원: 원본 논문(512차원) / 예시(4차원)
- 예시 head(h)가 2차원

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/c6f9d4b9-45a2-44e2-b2d8-5b91cf1eb544" width="450"/>

- 2차원의 쿼리, 키, 값을 만들기 위해 (4x2)차원의 가중치 행렬을 곱함

2) 하나의 쿼리 단어 → 전체 키와 행렬곱 → 어텐션 Energy (dk)

3) softmax를 거쳐 쿼리에 대한 각 Key의 가중치를 구함 (72,,) 

4) 가중치 * value 를 모두 더해 최종 Attention value를 구함 

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/bae405c5-15f8-45e5-8f47-3657408595cd" width="450"/>

- 전체 예시
    - 문장에 대해 쿼리, 키 , 값을 생성 (임베딩차원:4, head:2)
      
      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/669bb45a-f97a-406a-9f72-42eada374044" width="450"/>

    - 쿼리와 키 행렬을 곱하여 Attention Energy 행렬 구함 (단어들의 연관성을 나타냄)
    - Softmax를 거쳐서 각 행(쿼리)에 대한 키의 확률 값을 계산 = 가중치 = Attention Score
    - 가중치와 value 곱하여 최종 Attention 생성

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/b8a2dae2-831c-4d17-b57f-80cc534aa0cf" width="450"/>

- 마스크 행렬(mask matrix): 음수 무한 값을 넣어 특정 단어를 무시하도록 함

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/5b6b23ea-f9ee-48a1-bef0-07dd77619949" width="450"/>    

### 트랜스포머의 동작 원리: Multi-Head Attention

- 각 head 마다 쿼리, 키, 값을 만들어서 Attention을 수행

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/5f4e8c70-a2ee-48a6-a169-0db8d89b5f37" width="450"/>

- head 마다의 attention value을 concat하면 처음 임베딩 차원과 동일한 벡터가 생성됨
- dmodel x dmodel 크기의 가중치 행렬과 곱하면 Multihead Attention의 결과가 됨

<img src="https://github.com/daunJJ/LLM_Study/assets/109944763/f8f7668e-00e6-4164-b5db-097638167cd0" width="450"/>    

### 트랜스포머의 동작 원리: 어텐션의 종류

<img src="https://github.com/daunJJ/LLM_Study/assets/109944763/93b792f8-7f08-4984-84cd-25669a17e9db" width="450"/>

- Encoder Self-Attention
    - 각 단어가 서로에게 어떤 연관성을 가지는지 구함
    - 전체 문장에 대한 표현을 학습
- Masked Decoder Self-Attention
    - 각 출력 단어가 앞쪽에 등장한 출력 단어를 참고할 수 있도록 함
- Encoder-Decoder Attention
    - 쿼리는 디코더에 있고 키, 값은 인코더에 있음
    - 각 출력 단어(쿼리)가 입력 단어(키, ) 중 어디에 가중치를 둘지 구함

### 트랜스포머의 동작 원리: Self-Attention

- Self-Attention은 인코더와 디코더 모두에서 사용됨
- 매번 입력 문장에서 각 단어가 다른 어떤 단어와 연관성이 높은 지 계산

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/0b28150d-f4ee-4392-8b1d-018d51d70183" width="450"/>

    it은 tree와 연관이 높음 
    

### 트랜스포머 핵심 요약

- 트랜스포머는 RNN이나 CNN을 사용하지 않고, Positional Encoding을 사용함
- 인코더와 디코더로 구성되며, Attention 과정을 여러 레이어에서 반복함
- BERT와 가튼 향상된 네트워크에서도 채택되고 있음

### 트랜스포머의 동작 원리: Positional Encoding

- 주기 함수 공식을 사용하여 Positional Encoding 수행

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/3d773b1c-7674-425e-9349-424fa42a44d1" width="450"/>

    - pos: 각 단어의 번호
    - i: 각 단어에 대한 임베딩 값의 위치
- 주기성을 학습할수만 있다면 sin, cos이 아닌 다른 함수 사용 가능
    
    = 위치에 대한 임베딩을 따로 학습할수만 있으면 됨 (⇒ 트랜스포머 이후 별도의 임베딩 레이어를 사용하기도 함) 
    
- 각 단어에 대한 임베딩 및 인코딩 과정

  <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/4e0e93c7-dafd-4b48-bfa5-4b890253f777" width="450"/>

    ⭕: (1, 4) 
    
    - 입력 단어의 임베딩과 동일한 dimension을 가지는 위치 인코딩값을 더함

      <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/0fb8bb78-eaea-42b1-b681-ebfc534e9875" width="450"/>
