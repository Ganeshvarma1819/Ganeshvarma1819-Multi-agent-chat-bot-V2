import os
import random
import asyncio
import re
import json
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# Other Libraries
import httpx
from transformers import pipeline

# LangChain & AI Model Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain Agent Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain import hub

# Logging
import logging
from logging_config import setup_logger

# Setup logging
setup_logger("api")
logger = logging.getLogger("api")

# Load environment variables
load_dotenv()

# --- GLOBAL VARIABLES ---
vector_store_global = None
llm = None
embeddings = None
translator = None
CHROMA_DB_PATH = "./chroma_db"

# --- PYDANTIC MODELS ---
class AskRequest(BaseModel):
    question: str
    session_id: str = Field(default_factory=lambda: f"session_{random.randint(1000, 9999)}")

# --- FASTAPI APP & STARTUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global vector_store_global, translator, llm, embeddings
    logger.info("Server starting up...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, stream=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store_global = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    try:
        translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
    except Exception as e:
        logger.error(f"Failed to load NLLB model: {e}. Falling back to Helsinki-NLP model.")
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ine")
    logger.info("All models and databases loaded. Server is ready!")

# --- PROMPTS AND AGENTS ---

# ✅ FINAL, STRICTER SYNTHESIS PROMPT
synthesis_prompt = PromptTemplate(
    template="""You are an expert editor and AI assistant. Your primary goal is to answer the user's question by synthesizing the provided context into a **single, cohesive, and completely non-repetitive paragraph.**

**CRITICAL RULES:**
1.  **ZERO REDUNDANCY:** If you see the same fact, figure, or detail mentioned multiple times in the context, you MUST state it **ONLY ONCE** in your final answer. Synthesize all supporting sources into a single citation for that fact.
2.  **NO BROKEN SENTENCES:** Your final output must be a well-written, grammatically correct paragraph. Do not include sentence fragments or stitch text together awkwardly.
3.  **SYNTHESIZE, DO NOT LIST:** Weave the information together logically. Do not simply list the facts you find.
4.  **CITE ACCURATELY:** Cite every piece of information using the format (Source X, Page Y).
5.  **USE CONTEXT ONLY:** Do not use any information not present in the context below.

<context>
{context}
</context>

<question>
{question}
</question>

Here is the single, synthesized, and non-repetitive paragraph that answers the user's question:
""",
    input_variables=["context", "question"],
)


intent_classifier_prompt_template = "Classify the user's question: 'mathematical', 'informational', or 'conversational'. Return ONLY the category name. USER QUESTION: {question} CATEGORY:"

def build_code_agent(llm):
    tools = [PythonREPLTool()]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- HELPER FUNCTIONS ---
def batch_translate(text: str, model_pipeline) -> str:
    if not text.strip(): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*\((Source\s\d+,\sPage\s\d+)\)', '', text)
    text = re.sub(r'[\*\#]', '', text)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences: return ""
    try:
        translations_raw = model_pipeline(sentences, src_lang="eng_Latn", tgt_lang="tel_Telu", max_length=512)
        return " ".join([t['translation_text'] for t in translations_raw])
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        return "[Translation failed]"

async def brave_search(query, limit=3):
    pass

# --- ORCHESTRATOR ---
async def orchestrate_and_stream(question: str, session_id: str, send_chunk_cb):
    try:
        global llm, vector_store_global
        logger.info(f"Session: {session_id}, Question: '{question}'")

        intent_classifier_chain = PromptTemplate.from_template(intent_classifier_prompt_template) | llm | StrOutputParser()
        intent = (await intent_classifier_chain.ainvoke({"question": question})).strip().lower()
        logger.info(f"Intent classified as: '{intent}'")

        full_english_answer = ""
        agent_label = "Knowledge Agent"

        if "informational" in intent:
            retriever = vector_store_global.as_retriever(search_type="similarity", search_kwargs={'k': 7})
            docs = await retriever.ainvoke(question)

            if not docs:
                full_english_answer = "I could not find any relevant information in my documents to answer your question."
                await send_chunk_cb("english", agent_label, full_english_answer, final=False)
            else:
                unique_sources = sorted(list(set(doc.metadata.get("source") for doc in docs if doc.metadata.get("source"))))
                source_map = {source: f"Source {i+1}" for i, source in enumerate(unique_sources)}

                context_parts = []
                for doc in docs:
                    source_name = source_map.get(doc.metadata.get('source'), 'Unknown Source')
                    page_num = doc.metadata.get('page', 'N/A')
                    context_parts.append(f"Source: {source_name}, Page: {page_num}\nContent: {doc.page_content}")
                
                formatted_context = "\n\n---\n\n".join(context_parts)

                search_meta_info = f"**🔍 Search Results:** Found {len(docs)} relevant document chunks from {len(unique_sources)} source(s)."
                await send_chunk_cb("metadata", "System", search_meta_info, final=False)
                
                synthesis_chain = synthesis_prompt | llm | StrOutputParser()
                async for chunk in synthesis_chain.astream({"context": formatted_context, "question": question}):
                    full_english_answer += chunk
                    await send_chunk_cb("english", agent_label, chunk, final=False)

        elif "mathematical" in intent:
            agent_label = "Math Agent"
            code_agent = build_code_agent(llm)
            result = await run_in_threadpool(code_agent.invoke, {"input": question})
            full_english_answer = result.get("output", "Could not calculate a result.")
            await send_chunk_cb("english", agent_label, full_english_answer, final=False)

        elif "conversational" in intent:
            agent_label = "Conversational Agent"
            full_english_answer = await (llm | StrOutputParser()).ainvoke(question)
            await send_chunk_cb("english", agent_label, full_english_answer, final=False)
            
        else:
            full_english_answer = "I can only answer informational, mathematical, or conversational questions at this time."
            await send_chunk_cb("english", "System", full_english_answer, final=False)

        logger.info("Translating final answer to Telugu...")
        telugu_answer = await run_in_threadpool(batch_translate, full_english_answer, translator)
        await send_chunk_cb("telugu", "అనuvad ఏజెంట్", telugu_answer, final=False)
        await send_chunk_cb("english", agent_label, "", final=True)
        logger.info("Orchestrator finished successfully.")

    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        await send_chunk_cb("english", "System", f"An error occurred: {e}", final=True)

# --- FASTAPI ENDPOINT ---
@app.post("/ask_multi")
async def ask_multi(request: AskRequest):
    session_id, question = request.session_id, request.question
    send_queue = asyncio.Queue()
    async def send_sse_chunk(lang, agent, content, final=False):
        payload_type = "metadata" if lang == "metadata" else "agent_token"
        await send_queue.put({"type": payload_type, "lang": lang, "agent": agent, "content": content, "final": final})
    async def stream_generator():
        asyncio.create_task(orchestrate_and_stream(question, session_id, send_sse_chunk))
        try:
            while True:
                item = await send_queue.get()
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("final"):
                    break
        except asyncio.CancelledError:
            logger.info("Stream cancelled by client.")
        finally:
            logger.info("Closing stream generator.")
    return StreamingResponse(stream_generator(), media_type="text/event-stream")