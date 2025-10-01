# api.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# --- Global constants ---
CHROMA_DB_PATH = "./chroma_db"

# --- Deduplication Wrapper ---
class DeduplicatedChroma(Chroma):
    """Wrapper around Chroma to prevent duplicate documents being inserted."""

    def add_documents(self, documents, **kwargs):
        if not documents:
            return []

        # Fetch existing documents
        existing_texts = set()
        if self._collection.count() > 0:
            all_existing = self._collection.get(include=["documents"])
            existing_texts = set(all_existing.get("documents", []))

        # Filter new docs
        new_docs = [doc for doc in documents if doc.page_content not in existing_texts]

        if not new_docs:
            print("⚠️ No new unique documents to add (all were duplicates).")
            return []

        print(f"Deduplication: {len(new_docs)} unique docs (skipped {len(documents) - len(new_docs)} duplicates).")
        return super().add_documents(new_docs, **kwargs)


# --- Initialize FastAPI app ---
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_vectorstore():
    """Initialize the Chroma vector store with deduplication wrapper."""
    return DeduplicatedChroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    )


def get_llm():
    """Initialize Google Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.2, convert_system_message_to_human=True
    )


@app.post("/ask")
async def ask_question(request: Request):
    """Endpoint to handle user questions."""
    body = await request.json()
    query = body.get("question", "")

    if not query:
        return {"error": "Question is required."}

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Use the following context to answer the question.
        If the answer is not available in the context, say you don't know.

        Context:
        {context}

        Question: {question}
        """
    )

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    result = qa.invoke({"query": query})
    answer = result["result"]
    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

    return {"answer": answer, "sources": sources}


@app.post("/stream")
async def stream_answer(request: Request):
    """Endpoint for streaming answers (like chat)."""
    body = await request.json()
    query = body.get("question", "")

    if not query:
        return {"error": "Question is required."}

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    async def event_stream():
        prompt = f"Answer this question using retrieved context:\n\n{query}"
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(event_stream(), media_type="text/plain")
