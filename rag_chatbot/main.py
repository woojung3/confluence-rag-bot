import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import ConversationChain
from google.api_core.exceptions import ResourceExhausted
import time
from langchain_classic.memory import ConversationBufferMemory
from starlette.concurrency import run_in_threadpool

# --- Application Setup ---
app = FastAPI()

# --- Configuration and Global State ---
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
LOG_FILE = BASE_DIR / "chat_log.log"
INDEXES_DIR = BASE_DIR.parent / "faiss_indexes"
TOPICS_CONFIG_PATH = BASE_DIR / "topics.json"

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global cache for loaded resources
llm = None
embeddings = None
topics_config = {}
vector_store_cache = {}
chat_logger = None

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    topic: str
    mode: str = "rag_single_turn" # Add mode with a default
    history: List[Dict[str, str]] = []

# --- Helper Functions ---
def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    return re.sub(r'[^a-zA-Z0-9\u3131-\uD79D_.-]', '_', name)

def get_confluence_search_url(source_filename: str) -> str:
    """Generates a Confluence search URL."""
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    if not base_url:
        return None
    from urllib.parse import quote
    # Assuming the source_filename is the page title
    page_title = Path(source_filename).stem
    return f"{base_url}/wiki/search?text={quote(page_title)}"

# --- FastAPI Events ---
@app.on_event("startup")
def startup_event():
    """Load all necessary models and configurations at startup."""
    global llm, embeddings, topics_config, chat_logger
    
    # 1. Setup Logger
    chat_logger = logging.getLogger("chat")
    chat_logger.setLevel(logging.INFO)
    # Prevent duplicate handlers if uvicorn reloads
    if not chat_logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "data": %(message)s}',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        )
        handler.setFormatter(formatter)
        chat_logger.addHandler(handler)

    # 2. Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found in .env file")

    # 3. Load LLM
    print("Loading Generative AI model...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    
    # 4. Load Embedding Model
    print("Loading embedding model...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 5. Load Topics Configuration
    print("Loading topics configuration...")
    if TOPICS_CONFIG_PATH.exists():
        with open(TOPICS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
            for topic in topics_data:
                sanitized_name = sanitize_filename(topic["name"])
                topics_config[topic["name"]] = {
                    "prompt": topic["prompt"],
                    "index_path": INDEXES_DIR / sanitized_name
                }
    else:
        raise RuntimeError(f"Topics config file not found at {TOPICS_CONFIG_PATH}")
    
    print("Startup complete.")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serve the main index.html file."""
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/topics")
async def get_topics():
    """Return the list of available topic names."""
    return list(topics_config.keys())

@app.get("/info")
async def get_info():
    """Return metadata about the DB."""
    if INDEXES_DIR.exists():
        mod_time = INDEXES_DIR.stat().st_mtime
        return {"index_last_updated": datetime.fromtimestamp(mod_time).isoformat()}
    return {"index_last_updated": None}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests for both RAG and conversational modes."""
    topic_name = request.topic
    if topic_name not in topics_config:
        raise HTTPException(status_code=404, detail="Topic not found")

    response = await handle_rag_chat(request, topic_name)
    
    # Log the interaction
    log_data = {
        "topic": topic_name,
        "mode": request.mode, # Use request.mode
        "query": request.query,
        "answer": response.get("answer")
    }
    chat_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    return response

async def handle_rag_chat(request: ChatRequest, topic_name: str):
    """Handles RAG-based chat."""
    config = topics_config[topic_name]
    
    if topic_name not in vector_store_cache:
        print(f"Loading vector store for topic: {topic_name}")
        index_path = config["index_path"]
        if not index_path.exists():
            raise HTTPException(status_code=500, detail=f"Index for topic '{topic_name}' not found.")
        try:
            vector_store_cache[topic_name] = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")

    vectorstore = vector_store_cache[topic_name]
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    formatted_history = []
    if request.mode == "rag_multi_turn":
        # Initialize memory for multi-turn RAG
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        # Populate memory with past messages
        for message in request.history:
            if message['role'] == 'user':
                memory.chat_memory.add_user_message(message['content'])
            elif message['role'] == 'assistant':
                memory.chat_memory.add_ai_message(message['content'])
        formatted_history = memory.load_memory_variables({})["history"]
    # If mode is "rag_single_turn", formatted_history remains empty, effectively ignoring past conversation.

    base_prompt = config["prompt"]
    
    rag_instructions = [
        "RAG로 찾아낸 컨텍스트(context) 내용 중 답변과 관련이 깊은 Mermaid 차트가 있다면, 해당 차트의 원본 코드를 답변에 반드시 포함시켜 주세요. 만약 컨텍스트에서 Mermaid 차트를 찾지 못했다면, 절대로 스스로 차트를 만들지 마세요.",
        "답변에 사용된 컨텍스트의 출처를 답변 하단에 '--- 출처 ---' 섹션을 만들고, 각 출처를 목록 형태로 명확하게 제시해주세요. 출처가 여러 개일 경우 모두 포함해야 하며, 링크가 걸려 있어야 합니다."
    ]

    # '소장 옵시디언' 토픽의 경우, 출처 기재를 요청하지 않음
    if topic_name == "소장 옵시디언":
        rag_instructions.pop()

    final_template = (
        base_prompt +
        "\n\n" +
        "\n\n".join(rag_instructions) +
        """

---
**이전 대화:**
{history}
---
**컨텍스트:**
{context}
---
**질문:**
{input}
---
**답변:**
"""
    )

    prompt = PromptTemplate(
        input_variables=["history", "context", "input"],
        template=final_template
    )

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get('source', '출처 불명')
            url = get_confluence_search_url(source)
            if url:
                source_link = f"[{source}]({url})"
            else:
                source_link = source
            formatted_docs.append(f"- 출처: {source_link} (Raw Source: {source}, URL: {url})\n- 내용: {doc.page_content}")
        return "\n\n".join(formatted_docs)

    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "input": RunnablePassthrough(), "history": lambda x: formatted_history}
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        # Run the potentially blocking LLM invocation in a separate thread
        answer = await run_in_threadpool(rag_chain.invoke, request.query)
        return {"answer": answer}
    except ResourceExhausted as e:
        chat_logger.error(f"Google Gemini API quota exceeded: {e}")
        raise HTTPException(
            status_code=429,
            detail="Google Gemini API 할당량(쿼터)이 초과되었습니다. 잠시 후 다시 시도하거나, 관리자에게 문의하여 할당량을 늘려주세요."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG chain invocation: {e}")



# --- To run the app: uvicorn rag_chatbot.main:app --reload ---
