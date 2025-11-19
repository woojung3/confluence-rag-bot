# 주제별 페르소나 AI 챗봇

이 프로젝트는 `topics.json` 파일에 정의된 여러 주제(문서 세트)를 기반으로, 선택된 주제에 맞는 페르소나로 질문에 답변하는 RAG(검색 증강 생성) 챗봇입니다. FastAPI를 기반으로 구축되었으며, Confluence 문서를 활용하여 지식 기반을 구축합니다.

## 주요 기능 및 특징

*   **컨플루언스 데이터 추출**: Confluence 문서를 Markdown 파일로 로컬에 추출합니다.
*   **데이터 전처리 및 분할 (Chunking)**: 추출된 문서를 의미 있는 단위로 분할하고, 각 조각(Chunk)에 출처 정보(metadata)를 저장합니다.
*   **텍스트 벡터화 (임베딩)**: 텍스트 조각들을 임베딩 모델(`ko-sroberta-multitask`)을 사용해 벡터로 변환합니다.
*   **벡터 데이터베이스 저장**: 변환된 벡터들을 FAISS를 사용하여 로컬 벡터 DB로 저장합니다.
*   **검색 및 답변 생성 (RAG)**: 사용자 질문을 벡터로 변환하여 DB에서 유사 문서를 검색하고, 검색된 문서와 질문을 조합하여 Google Gemini LLM을 통해 답변을 생성합니다.
*   **주제별 페르소나 챗봇**: `topics.json` 파일을 통해 챗봇의 페르소나, 참조 문서 경로, 시스템 프롬프트를 유연하게 관리하며, 주제별로 독립적인 FAISS DB를 구축합니다.
*   **동적 UI 및 백엔드**: `topics.json`의 주제 목록을 기반으로 UI 메뉴를 동적으로 구성하고, 선택된 주제에 따라 RAG 체인을 동적으로 로드하여 답변을 제공합니다.

## 기술 스택

*   **프로그래밍 언어**: Python
*   **핵심 프레임워크**: LangChain
*   **API 서버**: FastAPI
*   **UI**: HTML, CSS, JavaScript
*   **임베딩 모델 (로컬)**: `ko-sroberta-multitask`
*   **벡터 DB (로컬)**: FAISS
*   **LLM (답변 생성)**: Google Gemini
*   **설정 관리**: JSON (주제별 페르소나 정의)

## 사전 준비

1.  **Python**: Python 3.9 이상 버전이 설치되어 있어야 합니다.
2.  **API 키 및 환경 변수**: Google Gemini API 키와 Confluence 기본 URL이 필요합니다.
    -   `rag_chatbot` 디렉토리 안에 `.env` 파일을 생성하고 아래 내용을 채워야 합니다.

        ```
        # .env
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        CONFLUENCE_BASE_URL="https://your-domain.atlassian.net"
        ```
    - `CONFLUENCE_BASE_URL`은 답변 출처에 Confluence 검색 링크를 생성할 때 사용됩니다.

## 설치 및 설정

1.  **가상환경 생성 및 활성화**:
    프로젝트의 독립적인 실행 환경을 위해 가상환경을 생성하고 활성화합니다.
    ```bash
    # macOS / Linux (프로젝트 루트에서 실행)
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **필요 라이브러리 설치**:
    `requirements.txt` 파일에 명시된 모든 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

## 챗봇 설정 및 데이터베이스 구축

### 1. 주제(페르소나) 정의

챗봇이 답변할 주제와 페르소나는 `rag_chatbot/topics.json` 파일에서 관리합니다. 이 파일을 열어 챗봇의 페르소나, 참조할 문서 경로, 그리고 시스템 프롬프트를 필요에 맞게 수정하거나 추가할 수 있습니다.

-   **`name`**: UI에 표시될 페르소나 이름
-   **`path`**: 참조할 문서들의 루트 폴더 경로 (`output/` 기준)
-   **`prompt`**: 해당 페르소나에 주입될 시스템 프롬프트

### 2. 데이터베이스 구축

챗봇이 문서를 검색하고 이해할 수 있도록, `topics.json`에 정의된 각 주제별로 벡터 데이터베이스(FAISS)를 생성해야 합니다. 이 과정은 최초 한 번, 또는 문서 내용이 변경될 때마다 실행해야 합니다.

```bash
# 프로젝트 루트 디렉토리에서 실행
.venv/bin/python3 build_db.py
```
이 명령을 실행하면 `faiss_indexes`라는 폴더가 생성되며, 이 안에 각 주제별 벡터 데이터베이스 파일들이 저장됩니다.

## 애플리케이션 실행

### 1. 개발용 실행

아래 명령어를 실행하면 개발용 서버가 시작됩니다. 코드 변경 시 서버가 자동으로 재시작됩니다.
```bash
uvicorn rag_chatbot.main:app --reload
```
서버가 시작되면 웹 브라우저에서 `http://127.0.0.1:8000` 주소로 접속하여 챗봇을 사용할 수 있습니다.

### 2. 서버 배포용 실행 (tmux 사용)

`tmux`를 사용하면 터미널 세션을 유지한 채 백그라운드에서 애플리케이션을 실행할 수 있습니다. 이는 서버(예: Mac Mini)에서 터미널 접속을 종료한 후에도 챗봇이 계속 동작하게 할 때 유용합니다.

1.  **새로운 tmux 세션 생성**:
    ```bash
    tmux new -s rag_chatbot_session
    ```
    (`rag_chatbot_session`은 원하는 세션 이름으로 변경 가능합니다.)

2.  **애플리케이션 실행**:
    `tmux` 세션 내에서 다음 명령어를 실행하여 챗봇 서버를 시작합니다.
    ```bash
    uvicorn rag_chatbot.main:app --host 0.0.0.0 --port 8000
    ```
    (`--host 0.0.0.0` 옵션으로 외부 접속을 허용합니다.)

3.  **tmux 세션에서 분리 (Detach)**:
    `Ctrl+b`를 누른 후 `d`를 눌러 현재 `tmux` 세션에서 분리합니다. 이제 터미널을 닫아도 애플리케이션은 백그라운드에서 계속 실행됩니다.

4.  **tmux 세션에 다시 연결 (Reattach)**:
    언제든지 다음 명령어로 실행 중인 세션에 다시 연결할 수 있습니다.
    ```bash
    tmux attach -t rag_chatbot_session
    ```

5.  **tmux 세션 종료**:
    `tmux` 세션 내에서 `exit`를 입력하거나, `Ctrl+b`를 누른 후 `x`를 눌러 세션을 종료할 수 있습니다. 이 경우 챗봇 애플리케이션도 함께 종료됩니다.

    또는, 세션에 연결하지 않은 상태에서 다음 명령어로 세션을 강제 종료할 수 있습니다.
    ```bash
    tmux kill-session -t rag_chatbot_session
    ```

## Confluence 문서 추출 가이드

이 문서는 Confluence 페이지를 Markdown 형식으로 내보내는 방법을 설명합니다.

### 사용법

Confluence 페이지와 모든 하위 페이지를 내보내려면 다음 명령을 사용하십시오. Atlassian 사용자 이름(이메일)과 API 토큰을 묻는 메시지가 표시됩니다.

```bash
.venv/bin/confluence-markdown-exporter pages-with-descendants <page-url> <output-path>
```
**참고**: `confluence-markdown-exporter` 패키지는 프로젝트 루트의 `requirements.txt`를 통해 설치됩니다.

### 예시

```bash
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089320505/1.1.' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089615431/OKR' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089517115/1.3.' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089549921/1.5.' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089517138/1.7.' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089386001/2.0.' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2571239425/EU25+Product+Plan' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2619343702/2024+AutoCrypt+PKI-V2X' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/1819934721/2025+AutoCrypt+PKI-Vehicles' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2571141121/2025' --output-path ./output/ &&
.venv/bin/confluence-markdown-exporter pages-with-descendants 'https://auto-jira.atlassian.net/wiki/spaces/camlab/pages/2089549827/3.' --output-path ./output/
```

## 추출된 문서 정제 (매크로 제거)

Confluence에서 문서를 추출한 후, FAISS 데이터베이스 구축 전에 불필요한 Confluence 매크로 관련 텍스트나 HTML `details` 태그를 제거하여 데이터를 정제해야 합니다. 이를 위해 `exporter/remove_macros.sh` 스크립트를 사용합니다.
Confluence 문서 추출이 완료되어 `exporter/output/` 경로에 Markdown 파일들이 생성된 후에 실행해야 합니다.

**사용법**:

```bash
./exporter/remove_macros.sh output/
```

## 프로젝트 구조

```
.
├── requirements.txt          # Python 의존성 목록
├── output/                   # Confluence 문서 추출 공간
├── build_db.py               # FAISS DB 구축 스크립트
├── faiss_indexes/            # 주제별 FAISS 벡터 데이터베이스 저장소
│   ├── 기술_규정_도우미/
│   ├── 연구소_생활_도우미/
│   └── ...
└── rag_chatbot/              # 챗봇 애플리케이션 소스 코드
    ├── .env                  # 환경 변수 설정 파일
    ├── main.py               # FastAPI 애플리케이션 메인 파일
    ├── topics.json           # 챗봇 주제 및 페르소나 정의 파일
    └── static/               # 웹 UI 정적 파일 (index.html 등)
```
