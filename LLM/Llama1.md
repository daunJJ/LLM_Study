## Llama1 논문 리뷰

### 특징

- [오픈소스](https://brunch.co.kr/@bumgeunsong/15)로 공개된 LLM 중 가장 강력한 성능

### 소개

- 더 많은 파라미터가 더 나은 성능을 이끌어낼 것
- 주어진 컴퓨팅 예산으로는 더 많은 데이터로 훈련된 더 ‘작은’ 모델이 최상의 성능
- 또한 더 오래 훈련된 작은 모델이 큰 모델보다 추론에 더 저렴함

⇒ 더 많은 토큰에 대해 더 작은 모델을 사용하여 제한된 추론 예산에서 최상의 성능을 달성하는 일련의 언어 모델을 훈련하자! 

### Pre-training Data

: 공개적으로 사용 가능하고 오픈 소싱과 호환되는 데이터만 사용

- CommomCrawl
    - [CCNet 파이프라인](https://arxiv.org/abs/1911.00359)을 이용해 전처리
        - [fastText](https://wikidocs.net/22883) 선형 분류기를 이용해 영어가 아닌 페이지 제거
            - **FastText**
                - 단어를 글자 단위 n-gram의 구성으로 취급
                    - apple → n=3 <ap, app, ppl, ple, le>
                - 내부 단어를 통해 모르는 단어와의 유사도를 계산할 수 있다는 장점
                    - birthplace → birth, place
                - 빈도가 적은 단어여도 다른 단어의 n-gram과 겹치면 높은 임베딩 벡터값을 가질 수 있다는 장점
                - 오타나 맞춤법이 틀린 단어도 높은 임베딩
        - [n-gram](https://wikidocs.net/21692) 언어모델을 이용해 저품질 콘텐츠 필터링
            - **n-gram**
                - 통계적 언어 모델 (SLM) 중 하나
                    - 문장에 대한 확률을 조건부 확률로 계산
                - n개의 연속적인 단어 이후에 나올 예측
                    - 이때 n-1개의 단어만을 고려
                     <img src="https://github.com/daunJJ/LLM_Study/assets/109944763/3a21cc13-15f8-4a82-98fa-47f03327105d" width="360"/>
                $$
                P(w\text{|boy is spreading}) = \frac{\text{count(boy is spreading}\ w)}{\text{count(boy is spreading)}}
                $$
                    
                - 문제점
                    - 희소문제
                    - n을 선택하는 것은 trade-off 문제
                        - n이 클수록 희소문제 심각해짐
        - 위키피디아를 참고하는 페이지와 무작위로 샘플링된 페이지 분류
- C4
    - 중복 제거와 언어 식별 단계 포함
    - CCNet과 차이점: 품질 필터링
        - 구두점의 존재나 웹페이지에 있는 단어 수 같은 [휴리스틱](https://blog.naver.com/mosfnet/222712926777)에 의존
                    
- GitHub
    - 휴리스틱을 기반으로 라인 길이나 영숫자 문자의 비율에 따라 저품질 파일 필터링
    - 정규 표현식을 사용하여 헤더와 같은 보일러플레이트 제거
        - 보일러플레이트
            
            BoilerPlate코드: 모든 코드를 작성하기 위해 항상 필요한 부분
            
            - Import : 필요한 코드를 불러들이는 부분
            - Component : 현 페이지를 구현하는 코드
            - StyleSheet : 페이지의 객체를 꾸미기 위한
            - styleExport : 현 Javascript 코드를 타 Javascript에서 접근하기 위한 부분
    - 중복 제거
- Wikipedia
    - 하이퍼링크, 댓글, 다른 형식의 보일러플레이트 제거
- Gutenberg and Books
    - 90% 이상의 내용이 중복되는 책 제거
- ArXiv(논문)
    - 참고 문헌 제거
    - 주석 제거
    - 논문 일관성을 높이기 위해 사용자가 작성한 정의와 매크로를 인라인으로 확장
- Stack Exchange(과학 웹사이트)
    - HTML 태그 제거
    - 답변 점수가 높은 순으로 정렬

### Tokenizer

- 바이트 페어 인코딩 (BPE) 알고리즘을 사용하여 토큰화
  -  BPE 알고리즘
    : 빈도가 높은 문자열을 짧은 바이트 쌍으로 나타내는 방식 (압축 용도)
    1. 각 문자의 빈도수 계산
    2. 가장 빈도가 높은 문자 쌍을 새로운 단위로 합침
    3. 더 이상 합칠 수 없을 때까지 과정2 반복 
    예) aaabdaaabac 에서 aaa를 Z로 변경
    
    
- [SentencePiece](https://wikidocs.net/86657)의 구현을 사용
- 모든 숫자를 개별 숫자로 분리
- 알려지지 않은 UTF-8 문자를 분해하기 위해 바이트를 대체로 사용
- 최종적으로 1.4조개의 토큰
- 대부분의 훈련 데이터에서 각 토큰은 훈련 중 한 번만 사용
    - 위키피디아와 Books만 예외적으로 2번의 Epoch

### Architecture

- 트랜스포머 아키텍처 기반
- 4가지 사이즈의 모델 구현

### Optimizer

- AdamW 옵티마이저를 사용
    - b1 = 0.9, b2 = 0.95
- 코사인 학습률 일정을 사용
    - 최종 학습률이 최대 학습률의 10%
- 가중치 감쇠 : 0.1
- 그래디언트 클리핑: 1.0

### 훈련 환경

- 65B 파라미터 모델을 훈련할 때, 80GB의 RAM을 가진 2048개의 A100 GPU 사용
- 초당 약 380 토큰 처리 → 1.4조 토큰이면 약 21일

### 주요 결과

: 특정 작업에서 다른 모델과 비교

- Common Sense Reasoning
    - 크기가 10배 작음에도 대부분의 벤치만크에서 GPT-3를 능가
- Closed-book Question Answering
    - GPT3, 친칠라보다 가장 강력한 성능
- Reading Comprehension
    - GPT3을 능가하며 PaLM과 경쟁력 있음
- Mathmatical reasoning
    - 수학 데이터에서 미세조정되지 않았음에도 불구하고 수학 데이터로 미세조정된 Minerva를 능가
- Code generation
    - 다른 일반 모델보다 높은 성능
- Massive Multitask Language Understanding
    - PasLM이 가장 강력하나 비슷한 성능
- Evolution of performance during training
    - 토큰이 많을수록 강력해짐

### 편향성, 독성, 진실성

- 독성
    - 비독성에 가까움
- 편향성
    - 다른 모델과 비교하여 크게 편향되지 않음
- 진실성
    - 잘못된 답을 할 가능성 높음

### 결론

- Llama-13B가 GPT-3를 능가하면서도 10배 이상 작다
- 공개된 데이터만을 사용하여 최첨단 성능을 달성할 수 있다
- 규모를 키우면 지속적인 성능 향상이 가능하다
