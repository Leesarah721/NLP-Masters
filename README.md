# 청약 S c a n n e r (NLP 마스터즈)

# 1. 프로젝트 소개

<aside>
💡

### **프로젝트 주제: 청약 관련 문서 요약 및 질의응답 챗봇**

- **목표**: PDF 문서를 요약하고, 해당 문서를 바탕으로 사용자의 질문에 답변하는 챗봇 개발.
- **기술**: LangChain을 활용하여 텍스트 분석 및 요약.
- **기능**:
    1. 사용자가 PDF를 업로드하고, 요약된 내용을 받을 수 있음.
    2. 챗봇을 통해 PDF의 내용에 대한 질문에 답변 제공.

### 핵심 타겟

- 청약에 관한 복잡한 문서, 약관, 정책 설명서 등을 요약하여 빠르게 이해할 수 있도록 지원합니다
</aside>

## 1-1. RAG 소개

![image.png](image.png)

## **1-2. 순서도**

![순서도.png](%25EC%2588%259C%25EC%2584%259C%25EB%258F%2584.png)

## **1-3.** 역할분담

<aside>
💡

**각 파트가 독립적으로 진행될 수 있도록 하여,  협업 효율성을 높였습니다!**

### **파트 A: 데이터 준비 및 전처리**

- **Load**: 다양한 파일 형식(PDF, Word 등)을 처리하고 예외 처리 기능 개발.
- **Split**: PDF를 섹션, 챕터, 페이지별로 분할하고 텍스트로 변환하여 저장.

### **파트 B: 데이터 임베딩 및 저장**

- **Embed**: 텍스트 데이터를 임베딩 벡터로 변환하고 최적화된 임베딩 모델 선택.
- **Store**: 벡터 데이터베이스에 임베딩 데이터 저장, 인덱스 생성 및 최적화.

### **파트 C: 질문 및 검색**

- **Question**: 사용자 질문 입력 UI 설계 및 검증 로직 개발.
- **Retrieve**: 질문을 벡터화하여 벡터 데이터베이스에서 관련 데이터 검색.

### **파트 D: 응답 생성 및 통합**

- **Prompt**: 검색된 데이터를 기반으로 LLM에 전달할 프롬프트 생성.
- **LLM**: LLM 호출 및 API 결과 처리.
- **Answer**: LLM 결과를 적합한 형태로 가공하고 추가 질문 처리.
</aside>

## **1-4. 사용툴**

<aside>
💡

- **백엔드** : python 3.10
- **프론트엔드** : Streamlit
- **버전 관리** : Git
- **프레임워크** : Langchain
- **IDE** : Visual Studio Code
</aside>

# 2. 프로젝트 개발내용

## 2-1.  PDF 파일 로더

![image.png](image%201.png)

**✅ 청약 공고문에서 PDF 파일 로더의 중요성**

- **표 정보의 정확한 추출**
    
    청약 공고문은 많은 표 형태의 정보를 포함하고 있으며, 이 정보를 정확하게 추출하고 레이아웃을 잘 따오는 것이 중요함.
    
- **정확한 분류와 구조화**
    
    표 및 데이터가 잘 분류되고 구조화되어야, 나중에 검색 시 필요한 정보에 빠르게 접근할 수 있음. 잘 정리된 데이터는 검색과 분석을 용이하게 만듦.
    
- **검색 효율성 향상**
    
    청약 공고문 내 정보가 체계적으로 분류되어 있으면, 검색할 때 필요한 정보를 쉽게 찾을 수 있어 사용자 경험을 크게 향상시킴.
    

**✅ 사용한 로더:  `PyMuPDFLoader`**
`pdfplumber`가 가장 높은 정확도를 보였지만 속도가 느리다는 점을 발견했으며, **`PyMuPDFLoader`**가 속도와 정확도 면에서 모두 적합하다고 판단했습니다.

## 2-2.  텍스트 전처리

**✅ 불용어 제거** 

- **nltk**
    - `nltk`는 한글 불용어 파일을 제공하지 않기 때문에, 한글 텍스트의 불용어 제거에 적합하지 않음.
- **kiwi**
    - `kiwi`는 **StopWords**를 통해 불용어를 제공하며, **tokenizer**로 단어를 분리 후 **join**을 통해 불용어를 제거하고 분리된 단어들을 다시 조합.
    - 이때, 한글을 제외한 다른 문자들은 모두 `replace` 처리됨.
    
    **➡️ 특수문자와 불용어 제거 문제**
    
    - 해당 PDF는 **특수문자**를 기준으로 내용이 나누어지는데, `kiwi`로 불용어 처리 및 특수문자 제거를 진행한 뒤 임베딩을 수행하면 **엉뚱한 대답**을 하는 경향이 있음.
    - 이는 특수문자를 중요한 구분자로 사용하는 문서에서 불용어 제거 및 특수문자 제거가 오히려 의미를 왜곡시키기 때문.
    
    **➡️ 결론**
    
    - 결국, **불용어 제거**를 하지 않기로 결정.
    - 특수문자를 기준으로 문서가 나누어지는 구조에서는 불용어 처리나 특수문자 제거가 텍스트의 의미를 흐트러뜨릴 수 있으므로, 이를 제거하지 않고 원본 텍스트를 그대로 사용하는 것이 더 나은 결과를 도출할 수 있음.
    

**✅ 텍스트 분할**

| 분할 도구 | 기준 | 장점 | 단점 | 사용 사례 |
| --- | --- | --- | --- | --- |
| RecursiveCharacter
TextSplitter | 구분자 
계층적 처리 | 구조와 문맥 보존 | 재귀적 처리로 
속도가 느릴 수 있음 | 문서, 긴 텍스트를 
문맥 손실 없이 처리 |
| Character
TextSplitter | 문자 수 | 간단하고 빠름 | 문맥 손실 가능 | 텍스트를 일정 길이로 
단순 분할 |
| Token
TextSplitter | 토큰 수 | 모델 입력 
제한과 일치 | 특정 토크나이저에 종속 | LLM 입력 준비 |
| Sentence
TextSplitter | 문장 | 문맥 보존 | 문장 길이에 따른 
제한 초과 가능 | 문장 단위로 정보를 
나누고 싶을 때 |
| Paragraph
TextSplitter | 단락 | 큰 의미 단위 보존 | 긴 단락에 대한 
추가 처리 필요 | 문서의 자연스러운 
흐름 유지 |
| FixedSize
Splitter | 고정 크기 | 균일한 크기로 
분할 가능 | 문맥 및 
구조 손실 가능 | 단순히 데이터를 
균등하게 나누어야 할 때 |
- **사용 도구**: `RecursiveCharacterTextSplitter`
- **설정**:
    - `chunk_size=1000`: 각 청크의 최대 크기를 1000자로 설정.
    - `chunk_overlap=200`: 청크 간 중복 영역을 200자로 설정하여 문맥 보존.
    - **커스텀 구분자**: `splits_text = ["■", "▣"]`를 설정하여 PDF 내에서 해당 기호를 기준으로 문맥을 나누도록 지정.

## 2-3. 임베딩

 **✅0. 로컬 환경에서의 임베딩 환경 구현**

- 일반적으로, OpenAI API 키를 활용하면 쉽게 해결 가능하지만,

 **비용 효율성**과 **데이터 프라이버시**를 위해 **무료 버전의 로컬 모델**을 활용 가능 하게 구현

**✅ 1. CPU vs GPU**

 **CPU 기반 임베딩의 성능 저하**

- 대용량 PDF 처리 시 CPU 임베딩이 느리고 시스템 응답 속도가 떨어짐.
- GPU 전환 초기에는 오히려 성능이 감소하는 문제 발견.

 **GPU 최적화 작업**

1. GPU 메모리 기반 동적 배치 크기 조정 기능 추가:
    - GPU 메모리 크기에 따라 `Batch Size`를 자동 조정.
2. 벡터 정규화(`normalize_embeddings=True`) 적용:
    - 벡터 품질 향상 및 처리 속도 최적화.
3. 새로운 모델 도입:
    - 기존 OpenAI 임베딩 모델에서 `jhgan/ko-sbert-nli`로 교체하여, 한국어에 최적화된 경량화 모델 사용.

**<CPU vs GPU성능 비교>**

![image.png](image%202.png)

1. CUDA 사용 1: GPU 기본 설정 사용
2. CUDA 사용 2: Batch_size 최적화, 임베딩 정규화

**✅ 2. 임베딩 모델 비교**

| 모델 | 주요 특징 | 장점 | 단점 | 용도 | 언어 | 핵심 기술 |
| --- | --- | --- | --- | --- | --- | --- |
| jhgan/ko-sbert-nli | - 한국어 문장 유사도 및 의미적 관계 분석에 강함 | - 문장 유사도 계산 및 의미적 관계 분석에서 뛰어난 성능 | - 문장 수준에서만 사용 가능, 다른 자연어 처리 작업에는 제한적 | - 문장 임베딩, 유사도 계산, 자연어 추론(NLI) | 한국어 | - SBERT 모델 기반, 의미적 텍스트 유사도 및 관계 분석에 최적화 |
| OpenAI Embeddings | - GPT 모델 기반 임베딩 생성- 텍스트의 의미를 깊이 있게 반영- API 기반으로 쉽게 호출 가능 | - 고품질 성능: 텍스트의 문맥을 정확하게 이해하고 의미를 잘 반영- 사용 용이성: API를 통해 쉽게 사용할 수 있음- 다국어 지원: 다양한 언어를 지원 | - 비용: 사용량에 따라 높은 비용 발생- 인터넷 의존: 클라우드 기반으로 인터넷 연결 필수- 커스터마이징 제한: 모델의 세부 조정 불가 | - 텍스트 임베딩 생성- 문서 검색, 질문 답변, 텍스트 유사도 계산 등 다양한 NLP 작업 | 여러 언어 지원 (영어 포함) | Transformer 기반GPT 모델 기반 임베딩 생성 |
| models/embedding-001 | - OpenAI의 GPT-3 모델을 기반으로 한 텍스트 임베딩- 텍스트 유사도 및 문서 검색에 최적화 | - 빠르고 효율적: 빠른 속도로 임베딩을 생성- 비용 효율성: 상대적으로 저렴하고 효율적인 결과 제공- 다양한 응용 가능: 문서 검색, 유사도 계산 등 다양한 작업에 활용 | - 비용: 사용량에 따라 비용 발생- 인터넷 의존: 클라우드 기반으로 항상 인터넷 연결 필요- 커스터마이징 제한: 직접 모델을 수정하거나 최적화할 수 없음 | - 텍스트 임베딩 생성- 유사도 계산, 검색 시스템, 질문 응답 시스템 등 | 영어 주로 사용, 다양한 언어 가능 | TransformerGPT-3 기반 임베딩 생성 |
| nlpai-lab/KoE5 | - 한국어 자연어 이해 및 생성 성능 향상 | - 한국어 텍스트 처리에 뛰어난 성능 | - 특정 작업에 최적화되어 있어 범용성에서 제한적일 수 있음 | - 텍스트 분류, 문서 임베딩, 질의응답, 텍스트 유사도 | 한국어 | - BERT 기반
Fine-tuning된 모델 |
| hkunlp/instructor-larger | - Instruction-tuned 모델 (대화형 응답 및 텍스트 생성에 강함) | - 다양한 작업을 처리할 수 있는 멀티태스크 학습 | - 한국어에 최적화된 성능은 다소 부족할 수 있음 | - 텍스트 생성, 질의응답, 대화형 AI 시스템 | 한국어, 영어 | - Instruction-tuning 기반, 멀티태스크 학습 |
| sentence-transformers/all-MiniLM-L6-v2 | - 경량화된 MiniLM 모델 기반- 빠르고 효율적인 텍스트 임베딩 생성- 로컬 실행 가능 | - 빠른 처리 속도: 경량화된 모델로 빠른 임베딩 생성- 로컬 실행: 서버 없이 로컬에서 실행 가능- 메모리 효율성: 적은 리소스로 높은 성능 발휘 | - 정확도 제한: 더 큰 모델보다 낮은 정확도- 메모리 제약: 메모리가 부족한 경우 성능 저하 가능- 큰 데이터셋 처리에 제약 있을 수 있음 | - 문장 간 유사도 계산- 텍스트 검색, 분류 작업에 최적화 | 영어, 다국어 (영어 최적화) | MiniLM(경량화된 Transformer) 기반 기술 |

**`nlpai-lab/KoE5`  vs `jhgan/ko-sbert-nli`
<임베딩 모델 `nlpai-lab/KoE5`  vs `jhgan/ko-sbert-nli` 성능 비교>**

![image.png](image%203.png)

**⇒ 경량화 모델(`jhgan/ko-sbert-nli`)로 전환하여 GPU 사용 필요성을 줄이고 CPU 기반 효율성 확보.**

**<`jhgan/ko-sbert-nli` 모델로 로드 유형만 변경 시 성능 비교>**

![image.png](image%204.png)

**⇒ 결론**

**모델에 따라서 GPU 로드가 실용적일 수 도 있다.**

**하지만,  본 모델의 경우 경량화 모델이라 GPU 로드의 필요성이 없다.**

## 2-4. Vectorstore

### ****🔺**`Vectorstore` 를 사용하는 이유**?

<aside>
💡

 텍스트 데이터를 벡터화하여 고차원 공간에 임베딩합니다. 
이를 통해 단어의 의미를 수학적으로 표현할 수 있으며, 
벡터 공간 내에서 **유사도 기반 검색**을 할 수 있게 합니다!

</aside>

### **✅ InMemoryVectorStore**

💡 **LangChain** 라이브러리에서 제공하는 **벡터 스토어**(Vector Store) 중 하나로, 데이터를 메모리 내에서 관리하는 단기 저장소로, 서버 종료 시 데이터가 사라지기 때문에 지속성 부족 문제와 데이터가 디스크에 저장되지 않는 단점이 있음. 또한, **성능**과 **지속성**을 고려할 때 `FAISS` 나**`ChromaDB`** 같은 다른 벡터 저장소를 사용하는 것이 더 적합하다고 판단.

### **✅ ChormaDB**

💡 벡터 검색을 위한 **오픈소스 데이터베이스**로, 다양한 **모델과 호환** 가능하며, 벡터와 메타데이터를 함께 저장하고 검색할 수 있음. 벡터 검색 및 메타데이터 검색을 동시에 처리할 수 있다는 이점이 있지만 정확도 면에서는 다소 떨어짐.

### **✅ FAISS**

💡FAISS는 **Facebook**에서 개발한 유명한 **오픈소스 라이브러리리**로 대규모 데이터셋에 대해 고속으로 근사 최근접 이웃 검색을 수행할 수 있는 최적화된 라이브러리입니다.  **속도**와 **검색 성능** 면에서 매우 유리하며, 대규모 데이터셋에 대해 고속으로 근사 최근접 이웃 검색을 수행할 수 있는 최적화된 라이브러리로, 분할된 문서를 저장하고 검색하는 데 효과적이기 때문에 **FAISS**를 벡터 저장소로 선택하는 것이 바람직함.

따라서, **FAISS**를 사용하여 분할된 문서를 벡터 저장소에 저장하고 검색하는 방식이 최적의 해결책이라고 결정

## 2-5. Retriever

### ****🔺 **Retriever?**

<aside>
💡

Retriever는 질문을 이해하고 관련 문서를 찾는 역할을 하며, 

그 후 Generator가 그 정보를 바탕으로 자연스러운 답변을 생성합니다. 이 단계에서 

Retriever의 정확성과 속도가 RAG 시스템의 성능을 크게 좌우합니다.

</aside>

- **Multi-Query Retriever**는 질문을 여러 개의 쿼리로 분리하여 다양한 관점에서 검색하고
- **Self-Querying Retriever**는 검색 과정에서 자동으로 새로운 쿼리를 생성해 반복적으로 개선합니다.
- **Parent-Document Retriever**는 부모-자식 관계를 기반으로 문서 간의 관계를 고려하여 검색
- **Ensemble Retriever**는 여러 가지 검색 방식을 결합하여 최상의 결과를 도출합니다.
- **Time-weighted Vector Retriever**는 시간이 중요한 경우 최신 정보를 우선적으로 처리하는 방식
- **Long Context Reorder**는 긴 문서에서 문맥을 잘 반영하기 위해 문서를 재조정

### **✅ Multi-Querying Retriver**

💡 하나의 질의(Query)를 다양한 방식으로 변형하여 
 여러 개의 질의를 생성하고, 이를 통해 검색 또는 정보 추출 성능을 향상시키는 방법

![image.png](image%205.png)

**🔹기대효과:** 

- **애매모호한 질문의 구체화**
    - 사용자가 입력한 모호하거나 불분명한 질문을 더 명확하고 구체적인 질문들로 변환하여 정보 검색 및 답변의 정확도를 높임.
- **다양한 관점에서의 정보 제공**
    - 하나의 질문에 대해 다양한 관점을 반영한 질의를 생성하여, 사용자가 놓쳤을 수 있는 중요한 정보나 대안을 탐색할 수 있도록 도움.

**🔹문제점:** 

- **질의와 관련 없는 생성 질의**
    - 생성된 여러 개의 질의가 원래 질의의 의도를 제대로 반영하지 못하고, 관련성이 떨어지는 질의가 생성되는 문제가 발생.
- **상이한 답변 제공**
    - Multi-Query Retriever가 생성한 여러 질의로 인해, 동일한 원질의에 대해 일관성 있는 답변이 제공되지 않고, 매번 다른 내용의 답변이 생성되는 문제가 발생.

### **✅ Ensemble Retriver**

💡 **두 가지 이상의 검색 방법**을 결합하여 **검색 성능을 향상**시키는 기법입니다. 이 코드에서는 **Dense Retriever**와 **Sparse Retriever**를 결합하여 앙상블 검색기를 구현하고 있습니다.

![image.png](image%206.png)

- **`dense_retriever` (밀집 검색기)**:
    - **FAISS** **벡터 기반** 검색 방식을 사용하는 **Dense Retriever**는 **밀집 벡터**를 기반으로 한 검색기입니다. 주로 **Embedding 모델**을 사용하여 텍스트를 벡터로 변환하고, 이 벡터들을 이용하여 검색합니다.
    - `search_type= similarity` 으로 검색 방법을 지정
- **`sparse_retriever` (희소 검색기)**:
    - **BM25**는 전통적인 **TF-IDF** 기반의 검색 기법 중 하나로, **희소 벡터**(단어 빈도 기반의 벡터)를 사용하여 문서와 질의 간의 유사도를 계산합니다.

- **`Ensemble Retriever`**:
    - `fetch_k = 4` **최초 검색**에서 **최대 4개의 문서**를 가져오도록 설정
    - 앙상블 방식에 따라 `weights=[0.5, 0.5]`로 두 검색기의 결과를 통합합니다
    - `k = 3`최종적으로 **최상위 3개의 문서**를 검색 결과로 반환 선택하여 최종 결과로 제공합니다.

## 2-6. Prompt

💡 모델에게 원하는 작업을 지시하거나, 필요한 정보를 요청할 수 있습니다
      모델이 무엇을 해야 하는지 정확히 인식하도록 돕습니다.
      이에 따른 결과나 답변을 생성하게 됩니다.

```python
    prompt = PromptTemplate.from_template(
        """
            PDF 문서에서 주어진 질문에 대한 답변을 제공하는 프롬프트입니다.
            주어진 문서는 텍스트와 표로 구성되어 있습니다.
            표 항목도 있으므로 표 항목은 주변 텍스트를 확인하고 답변해주세요.
            숫자로 이루어진 표는 되도록 합계쪽을 읽고 대답해주세요
            주어진 문서를 참고하여 질문에 대한 자세한 답변을 제공하고, 
            관련된 경우 목차를 포함하여 추가적인 지침을 제공합니다.
            답변을 찾지 못한 경우 유사한 답변을 찾아 대답해주세요.
            답변은 너무 길지 않게 요약해서 한국어로만 대답해주세요.
            
            #Contents:
            {context}

            #Question:
            {question}

            #Answer:
        """
    )
```

## 2-7. ChainModel

- **ConversationalRetrievalChain**:
    
    대화형 상호작용과 지속적인 문맥 유지가 필요한 시스템에 적합.
    
- **RetrievalQAWithSourcesChain**: 
신뢰성과 출처 제공이 중요한 QA 시스템에 유용.
- **MapReduceDocumentsChain**: 
대량의 문서를 병렬로 처리해 효율적으로 종합 정보 생성.
- **StuffDocumentsChain**: 
여러 문서를 단순히 합쳐 하나의 응답으로 생성하는 데 적합.
- **DocumentQuestionAnsweringChain**: 
단일 문서를 기반으로 질문에 빠르고 정확하게 답변.

### **주요 차이점 요약**

- **문맥 유지**: ConversationalRetrievalChain만 문맥을 유지.
- **출처 제공**: RetrievalQAWithSourcesChain이 출처 반환.
- **데이터 크기**: MapReduce는 대규모 데이터, Stuff는 소규모 데이터에 적합.
- **응답 범위**: Stuff와 DocumentQA는 단일 문서에 집중.

## 2-8. LLM 모델

### **✅ Groq API**

- Groq API는 **GroqChip**이라는 AI 하드웨어와 관련된 API로, 고성능 AI 모델 실행을 위한 **하드웨어 가속화**를 제공합니다.
- 주로  AI 모델 추론 속도를 높이는 데 사용됩니다.
- 다국어 LLM모델이어 한글 + 외국어로 답변 생성
- 병합되어있는 표는 아직 인식 한계
- 사용한 모델 : “mixtral-8x7b-32768”, ”llama3-8b-8192”

### **✅** GPT Turbo 3.5

- **GPT-3.5 Turbo**는 OpenAI가 개발한 GPT-3.5 모델의 최적화된 버전입니다.
- **속도 및 효율성**: GPT-3.5 Turbo는 GPT-3의 기본 버전보다 더 빠르고 비용 효율적인 모델로, 특히 실시간 응답이 중요한 애플리케이션에서 유리합니다.
- **성능**: GPT-3.5는 다양한 작업에서 좋은 성능을 보이며, 텍스트 생성, 요약, 번역, 질문 응답 등 여러 분야에서 활용됩니다.
- **파라미터 수**: GPT-3.5 모델은 GPT-3보다 적은 파라미터를 가질 수 있지만, 여전히 강력한 성능을 제공합니다. 정확한 파라미터 수는 공개되지 않았습니다.
- **주요 특징**:
    - 더 빠른 처리 속도와 낮은 비용을 제공.
    - GPT-3에 비해 더 많은 문맥을 처리할 수 있으며, 실용적이고 경제적인 선택지로 주로 사용됩니다.

### **✅** Gemini Flash 1.5

- **Gemini Flash 1.5**는 Google의 DeepMind에서 개발한 최신 언어 모델인 Gemini 시리즈의 일환으로, Gemini 1.5는 특히 큰 모델과 성능을 자랑합니다.
- **Gemini 1.5**는 Google의 최신 LLM로, 이전의 **LaMDA** 모델을 개선하고 확장한 버전입니다. Gemini 모델은 텍스트 생성, 번역, 요약, 질의 응답 등 다양한 자연어 처리 작업을 수행합니다.
- **특징**:
    - Gemini Flash는 이전 모델들보다 더 빠르고, 고급 기능을 갖추고 있습니다.
    - Google의 검색 엔진, 클라우드 서비스, Google Assistant 등 다양한 제품에 적용될 수 있습니다.
    - Gemini 모델은 보다 효율적이고 강력한 성능을 제공하여, 다양한 NLP 작업에서 뛰어난 결과를 보입니다.

# 3. 개선 및 보안할점

### **개선할 점**

- **표 인식 문제**: 복잡한 표를 인식하지 못함. OCR과 병합 표 데이터 처리 추가 예정.
- **불완전한 답변**: 내용이 잘리거나 일부만 답변하는 경향 있음.
- **LLM 최적화**: 현재 API로 구동 중, 로컬화 또는 sLLM 사용 검토 필요.
- **이미지 기반 PDF 처리**: 이미지로 구성된 PDF는 읽을 수 없음.
- **프롬프트 최적화**: 현재 프롬프트는 최적이 아니므로 추가 최적화 필요.
- **검색 정확도**: 유사한 질문에 대한 답변 정확도를 개선해야 함.

### **보안할 점**

- **검색 정확도 보완**: 비슷한 질문을 했을 때에도 더 정확한 답변을 제공하도록 보안 필요.
