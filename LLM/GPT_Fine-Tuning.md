## OpenAI API로 GPT Fine-Tuning 해보기

**Fine-Tuning을 위한 Prompt 구성**

- “role” : “system” -> chat봇의 특성을 정의 
- “role” : “user” -> 사용자가 입력할 프롬프트를 설정
- “role” : “assistant” -> 챗봇의 대답을 설정    

**OpenAI 라이브러리 install** 

```python
pip install openai
```

**학습 데이터 upload**

- 생성한 프롬프트 jsonl 파일을 업로드(최소 10개 이상의 prompt 필요)

```python
import os
import openai
openai.api_key = "OPEN_API_KEY"
openai.File.create(
  file=open("mydata.jsonl", "rb"),
  purpose='fine-tune'
)
```

**업로드 한 File Id를 확인**

```python
openai.File.list()
```

**Fine-Tuning 작업**

```python
openai.FineTuningJob.create(training_file="여러분의_File_Id", model="gpt-3.5-turbo")
```

**Fine-Tuning이 끝난 모델에 Inference를 진행**

- "role": "system"에 따라 답변 스타일이 달라짐
    - 건방진 자아를 가지는 챗봇 요청
    
    ```python
    completion = openai.ChatCompletion.create(
      model="모델_id",
      messages=[
        {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
        {"role": "user", "content": "What's the capital of France?"}
      ]
    )
    print(completion.choices[0].message)
    ```
    
    - 도움을 주려는 자아를 가진 챗봇 요청
    
    ```python
    completion = openai.ChatCompletion.create(
      model="모델_id",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
      ]
    )
    print(completion.choices[0].message)
    ```
    

**Fine-Tuning 작업 취소하기**

```python
openai.FineTuningJob.cancel("여러분의_job_id")
```

## GPT Fine-Tuning을 위한 형태로 KorQuad 데이터셋 정제하기

**KorQuAD 데이터 불러오기**

```python
import json

with open('KorQuAD_v1.0_dev.json', 'r') as f:
    dev_data = json.load(f)

dev_data = [item for topic in dev_data['data'] for item in topic['paragraphs'] ]
```

**Q:A 형태로 정제**

```python
refined_dict = {}
for original_dict in dev_data:
  for entry in original_dict['qas']:
      question = entry['question']
      answers = [ans['text'] for ans in entry['answers']][0]
      refined_dict[question] = answers

print(refined_dict)
```

{'임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?': '1989년 2월 15일', '1989년 6월 30일 평양축전에 대표로 파견 된 인물은?': '임수경', '임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배된 연도는?': '1989년', '임종석을 검거한 장소는 경희대 내 어디인가?': '학생회관 건물 계단', …}

**Fine-Tuning을 위한 Prompt 형태로 구성** 

```python
results = []

for question, answer in final_refined_dict.items():
    message = {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
    results.append(message)

results
```

[{'messages': [{'role': 'system', 'content': ''},
{'role': 'user', 'content': '임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?'},
{'role': 'assistant', 'content': '1989년 2월 15일'}]},
{'messages': [{'role': 'system', 'content': ''},
{'role': 'user', 'content': '1989년 6월 30일 평양축전에 대표로 파견 된 인물은?'},
{'role': 'assistant', 'content': '임수경'}]}]

**jsonl 파일로 저장 및 다운로드**

```python
with open('korquad_data.jsonl', 'w', encoding='utf-8') as file:
    for res in results:
        file.write(json.dumps(res, ensure_ascii=False) + '\n')
    
from google.colab import files
files.download('korquad_data.jsonl')
```

**inference 시 주의할 점**

- Llama와 달리 인자 고정이 불가능해서 동일한 프롬프트를 입력해도 결과(Output)가 달라질 수 있음
