## Alpaca 모델 리뷰

: Llama 1 7B 모델을 지시어 튜닝을 이용하여 Fine Tuning한 오픈소스 경량 LLM 모델

### 데이터

- text-davinci-003(GPT) 모델을 이용하여 self-instruct된 52K의 명령 수행 시연
    
    - [Alpaca Data Format]
    1. input 없음 
       - “instruction”:  “주요한 3가지 색은?”,
       - “input”: “”,
       - “output”:  “빨강, 파랑, 노랑”
    2. input 있음
       - “instruction”:  “동물과 식물을 분류해줘”,
       - “input”: “고양이, 해바라기, 코끼리, 느티나무”,
       - “output”:  “고양이: 동물, 해바라기: 식물 .. ”
  

### 주의점

- text-davinci-003을 기반하여 만든 명령어 데이터로 OpenAI와 경쟁하는 모델 개발 금지
- 적절한 안전조치를 설계하지 않아 일반적인 용도로 배포하기 위한 준비가 되지 않음

### Training recipe

1. Llama 와 같은 강력한 사전 훈련된 언어모델을 이용
2.  self-instruct 논문을 기반하여 기존의 강력한 언어 모델로 자동으로 명령 데이터 생성
    - self instruct seed set으로부터 175개의 인간이 작성한 명령-출력 쌍 사용
    - seed set을 text-davinci-003에 예시로 사용하여 52K의 명령 생성
    - 이를 허깅페이스의 훈련 프레임워크를 사용하여 Llama 1 7B모델을 supervised fine-tuning

→ 8개의 80GB GPU에서 3시간, $100 소요

### 성능 평가

- 5명의 인간 평가를 수행
- text-davinci-003과 Alpaca 7B 사이의 눈가림 페어 비교를 수행
    
    → 두 모델이 매우 유사한 성능을 가짐
    
- 작은 모델 크기와 제한된 데이터를 고려할 때 놀라운 성과

### 단점

- Alpaca의 답변은 일반적으로 ChatGPT보다 짧음
    
    → text-davinci-003으로 생성한 명령어 자체가 짧기 때문
    
- 환영, 독성, 스테레오 타입과 같은 결함 존재
- 잘못된 정보를 퍼뜨릴 수 있음

### 장점

- 훈련 방법을 공개하여 추가 위험 방지
- 연구의 확장성 제공
- 적은 비용으로 LLM을 훈련 시킬 수 있다는 것을 실현

### 배포

- OpenAI의 콘텐츠 중재 API를 사용하여 해로운 컨텐츠를 필터링
- 모든 모델 출력에 워터마크를 추가하여 Alpaca모델의 출력인지를 감지할 수 있게 함
- 비영리 목적으로만 사용 가능
