from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine
import uvicorn
import uuid
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

app = FastAPI(title="Mistral RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    search_documents: bool = True

class NewSessionRequest(BaseModel):
    title: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    sources: List[str] = []
    search_used: bool = True

class ChatSession(BaseModel):
    session_id: str
    title: str
    created_at: str
    messages: List[ChatMessage] = []
    updated_at: str

rag_engine = RAGEngine()
PERSISTENCE_FILE = "chat_sessions.json"

def load_sessions() -> Dict[str, ChatSession]:
    if os.path.exists(PERSISTENCE_FILE):
        try:
            with open(PERSISTENCE_FILE, 'r') as f:
                data = json.load(f)
                return {sid: ChatSession(**sdata) for sid, sdata in data.items()}
        except Exception as e:
            print(f"Error loading sessions: {e}")
    return {}

def save_sessions(sessions: Dict[str, ChatSession]):
    try:
        with open(PERSISTENCE_FILE, 'w') as f:
            json.dump({sid: s.dict() for sid, s in sessions.items()}, f, indent=2)
    except Exception as e:
        print(f"Error saving sessions: {e}")

chat_sessions = load_sessions()

def generate_session_title(first_message: str) -> str:
    return first_message[:30] + "..." if len(first_message) > 30 else first_message

# --- START OF FIX: SMART QUERY ROUTING ---

def needs_document_search(message: str) -> bool:
    """
    A simple router to decide if a query needs to search local documents.
    This is a basic implementation. More advanced versions could use another LLM call
    or a fine-tuned classifier model.
    """
    # Keywords that suggest the user is asking about the document content.
    # Make this specific to your documents.
    doc_specific_keywords = [
        "manoj", "chauhan", "cv", "resume", "experience", "project", "skill",
        "role", "company", "enterprise", "digital transformation", "initiatives"
    ]
    
    # Convert message to lowercase to make the check case-insensitive.
    lower_message = message.lower()
    
    # If any keyword is found, it likely needs a document search.
    for keyword in doc_specific_keywords:
        if keyword in lower_message:
            print(f"Router: Keyword '{keyword}' found. Routing to RAG search.")
            return True
            
    # Otherwise, assume it's a general knowledge question.
    print("Router: No specific keywords found. Routing to direct LLM call.")
    return False

# --- END OF FIX ---


@app.get("/")
async def root(): return {"message": "API is running"}

@app.post("/sessions/new")
async def create_new_session(request: NewSessionRequest):
    session_id, now = str(uuid.uuid4()), datetime.now().isoformat()
    chat_sessions[session_id] = ChatSession(
        session_id=session_id, title=request.title or "New Chat",
        created_at=now, updated_at=now, messages=[]
    )
    save_sessions(chat_sessions)
    return {"session_id": session_id, "title": request.title or "New Chat"}

@app.get("/sessions")
async def get_all_sessions():
    sessions_list = sorted(
        [s.dict() for s in chat_sessions.values()],
        key=lambda x: x["updated_at"], reverse=True
    )
    return {"sessions": sessions_list}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        save_sessions(chat_sessions)
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id
        now = datetime.now().isoformat()

        if not session_id or session_id not in chat_sessions:
            session_id = str(uuid.uuid4())
            chat_sessions[session_id] = ChatSession(
                session_id=session_id, title=generate_session_title(request.message),
                created_at=now, updated_at=now, messages=[]
            )
        
        session = chat_sessions[session_id]
        user_message = ChatMessage(
            role="user", content=request.message, timestamp=now,
            search_used=request.search_documents
        )
        session.messages.append(user_message)
        
        # --- MODIFIED: Use the router logic ---
        # The frontend toggle acts as an override. If the user unchecks it, we respect that.
        # Otherwise, our smart router decides.
        should_search = request.search_documents and needs_document_search(request.message)

        if should_search:
            rag_response = rag_engine.generate_response(request.message)
            answer, sources = rag_response["answer"], rag_response.get("sources", [])
        else:
            answer, sources = rag_engine.direct_response(request.message), []
        # --- END OF MODIFICATION ---
        
        assistant_message = ChatMessage(
            role="assistant", content=answer, timestamp=datetime.now().isoformat(),
            sources=sources, search_used=should_search
        )
        session.messages.append(assistant_message)
        
        if len(session.messages) == 2: session.title = generate_session_title(request.message)
        session.updated_at = datetime.now().isoformat()
        save_sessions(chat_sessions)
        
        return {
            "session_id": session_id, "message": answer,
            "sources": sources, "history": [msg.dict() for msg in session.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)