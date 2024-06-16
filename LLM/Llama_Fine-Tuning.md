## KorQuad 데이터셋에 맞게 Llama Fine-Tuning

### 프로젝트 소개

- 프로젝트 목적
    - **특정 좁은 분야(narrow) 의 지식에 있어서**는 현존 최강의 LLM(GTP-4)보다 **강력한 Fine-Tuning 된 LLM**을 만들어보자
- 전이학습
    - 사전 학습된 Neural Networks의 파라미터를 새로운 Task에 맞게 fine-tuning
    - 자연어 처리, 혹은 컴퓨터 비전 영역에서 광범위하게 사용
- KorQuad 데이터셋
    - Quad: 스탠포드 대학에서 만든 **Question & Answer 쌍으로 구성된 데이터셋**
    - KorQuad: Quad의 한국어 버전
    - Question과 Answer를 위한 문맥 Paragraph가 주어지고 해당 Paragraph에 있는 정보를 토대로 Answer를 맞추는 형태의 데이터셋
    - 전체 데이터는 1,560 개의 Wikipedia article에 대해 10,645건의 문단과 66,181개의 질의응답쌍
        - Context, Question, Answer, answer_start가 담김 → 동일한 Context에 대해 여러 개의 QA가 있을 수 있음

### KorQuad 데이터셋 정제

- dev_data 형태
    
    {qas: [ {answers : [], id : , question : }, {answers : [], id : , question : } … ],
    
    context: }
    
    ```python
    {'qas': [{'answers': [{'text': '1972년', 'answer_start': 0}],
        'id': '5912341-7-0',
        'question': '테드 번디가 워싱턴 대학교를 졸업한 해는?'},
       {'answers': [{'text': '워싱턴 대학교', 'answer_start': 6}],
        'id': '5912341-7-1',
        'question': '테드번디가 1972년 졸업한 대학교는?'},
       {'answers': [{'text': '로스 데이비스', 'answer_start': 177}],
        'id': '5912341-7-2',
        'question': '워싱턴 주 공화당 의장을 지낸 사람은?'},
       {'answers': [{'text': '다니엘 J. 에반스', 'answer_start': 30}],
        'id': '6599035-7-0',
        'question': '1972년 번디는 대학교를 졸업하고 누구의 재선 캠페인에 합류했나?'},
       {'answers': [{'text': '앨버트 로젤리니', 'answer_start': 85}],
        'id': '6599035-7-1',
        'question': '번디는 누구를 따라다니며 선거 유세 연설을 녹음해서 에반스 캠프팀이 분석하도록 도왔나?'},
       {'answers': [{'text': '로스 데이비스', 'answer_start': 177}],
        'id': '6599035-7-2',
        'question': '에반스가 재선에 성공한 후 번디는 누구의 사무보조원으로 고용됐나?'}],
      'context': '1972년 워싱턴 대학교를 졸업한 후에 번디는 주지사 다니엘 J. 에반스의 재선 캠페인에 합류했다. 대학생인 것처럼 행세하며 에반스의 적수인 전 주지사 앨버트 로젤리니를 따라다니면서 미행했고, 선거 유세 연설을 녹음해서 에반스 캠프팀이 분석하도록 도왔다. 에반스가 재선에 성공한 뒤에 번디는 워싱턴 주 공화당 의장인 로스 데이비스에게 사무보조원으로 고용됐다. 데이비스는 번드를 좋게 회상했고 번디를 "똑똑하며 진취적이며... 그리고 체제 신봉자"라고 묘사했다. 1972년 초기에 번디는 별로 좋지 못한 로스쿨 입학 시험 성적에도 불구하고 퓨젯 사운드 대학교와 유타 대학교 로스쿨 입학허가를 받았다. 번디는 에반스, 데이비스 그리고 다수의 워싱턴 대학교 심리학과 교수진에게 추천서를 받은 것이 강점이었다.'}
    ```
    
- qas 안에 ‘quesiton’과 ‘answer’만 파싱하여 → {’question’: ‘answer’} 딕셔너리로 만듬
- prompt 생성
    - 학습 시에는 아래 형태로 학습을 시키지만
    - 나중에 test할 때는 response를 빼고 앞부분만 input으로 넣어줌
    
    ```python
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {question} ### Response: {answer}"
    ```
    
- dict의 key가 ‘text’이고 value에 prompt가 들어간 형태로 정제해야 라이브러리 사용 가능
    
    ```python
    prompt_dict = {}
    prompt_dict['text'] = prompt
    ```
    
- **ChatGPT를 이용해서 10개씩 추가 문장을 생성 (Augmentation)**
    
    - [ChatGPT에 다음과 같이 지시]
      - "prompt_dict의 한 question 문장" 위 문장을 Data Augmentation하기 위해서 뜻은 같지만 비슷한 문장 10개를 만들어줘
    
    - Augmentation_dict에 key는 ‘원본 문장’, value는 ‘만들어진 문장 리스트’ 형태로 저장
- Augmentation한 문장(Q)에 A를 달아서 최종적으로 final_prompt_list 생성
    
    ```python
    final_prompt_list = []
    for idx, (question, answer) in enumerate(refined_dict.items()):
      if idx >= num_items:
        break
      print(question)
      prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {question} ### Response: {answer}"
      print(idx, prompt)
      prompt_dict = {}
      prompt_dict['text'] = prompt
      final_prompt_list.append(prompt_dict)
      # Data Augmentation
      auagmentated_questions = augmentation_dict[question]
      print(len(auagmentated_questions))
      for each_question in auagmentated_questions:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {each_question} ### Response: {answer}"
        print(idx, prompt)
        prompt_dict = {}
        prompt_dict['text'] = prompt
        final_prompt_list.append(prompt_dict)
    ```
    

### Llama2를 KorQuad 데이터셋에 맞게 fine-tuning 하기

- 허깅페이스에서 제공하는 [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) 이용
    - 복잡한 코드를 작성하지 않고 AutoTrain 커멘드로 간단하게 Llama2를 fine-tuning 할 수 있게 도와주는 라이브러리
- 하이퍼 파라미터 지정
    
    ```python
    !autotrain llm --train \
        --project-name "llama2-korquad-finetuning-da" \  # 학습 과정이 저장될 폴더명
        --model "TinyPixel/Llama-2-7B-bf16-sharded" \  # (허깅페이스에 올라온 모델 중) pretraining 모델을 어떠한 모델을 쓸지 
        --data-path "korquad_prompt_da" \  # fine-tuning에 사용할 데이터 경로 (해당 폴더 안에 데이터 넣기)
        --text-column "text" \  # text가 존재하는 key를 지정 
        --peft \  # peft(LoRA) 사용 여부 
        --quantization "int4" \  # int4 사용 여부 
        --lr 2e-4 \
        --batch-size 8 \
        --epochs 40 \
        --trainer sft \  # supervise finetuning
        --model_max_length 256  # max: 4096, 프롬프트 토큰의 개수 지정 -> 길수록 오래 걸림 
    ```
    
- 완료되면 llama2-korquad-finetuning-da 폴더 안에 check point 파일 생성됨
    
    = llama2 7B 모델이 korquad 데이터셋에 맞게 fine-tuning 되어 저장됨 
    

### **KorQuad 데이터셋에 맞게 fine-tuning 된 Llama 2의 성능 측정하기**

- AutoTrain Advanced를 불러오고 pytorch 업데이트
- fine-tuning이 끝난 check-point 파일을 업로드 후 zip 압축 해제
- config 지정 후 모델 불러오기
    
    ```python
    model_id = "TinyPixel/Llama-2-7B-bf16-sharded"  # 사용할 모델명 
    #peft_model_id = "./llama2-korquad-finetuning/checkpoint-208"   # Data Augmentation 적용 x
    peft_model_id = "./llama2-korquad-finetuning-da/checkpoint-480"   # Data Augmentation 적용 o
    # checkpoint 파일 path 지정 
    
    config = PeftConfig.from_pretrained(peft_model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype="float16",
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.eval()
    ```
    
- test를 위해서 prompt 포맷을 맞춤
    - 원래 prompt에서 ‘response’ 앞까지만 지정
    
    ```python
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "
    
    # do_sample : False -> 항상 가장 높은 확률값을 가진 단어로 예측 (=똑같은 input에 대해서는 항상 같은 결과를 output)
    def gen(x):
        q = prompt % (x,)
        gened = model.generate(
            **tokenizer(
                q,
                return_tensors='pt',
                return_token_type_ids=False
            ).to('cuda'),
            max_new_tokens=128,
            early_stopping=True,  
            do_sample=False,  
        )
        return tokenizer.decode(gened[0]).replace(q, "")
    ```
    
- gen(”질문”)으로 성능 test
- 예측 성능 비교
    
    ```
    # GPT-4(ChaptGPT) : 11/20
    # Fine-Tuend Llama 2(Data Augmentation 적용 x) : 15/20
    # Fine-Tuend Llama 2(Data Augmentation 적용 o) : 20/20
    ```
    
    - **Augmentation을 적용한 fine-tuning된 Llama2**의 성능이 가장 좋음!
        
        = **특정 좁은 분야(narrow) 의 지식에 있어서 GPT보다 강력한 fine-tuning된 LLM을 만들 수 있다**
