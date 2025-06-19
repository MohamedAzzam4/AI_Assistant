# main.py - Full FastAPI Backend for Internal AI Assistant with RAG Core and Multilingual AI

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import sqlite3
from jose import jwt
from datetime import datetime, timedelta
import os
import shutil
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Internal AI Assistant Backend",
    description="Backend for a Retrieval-Augmented Generation (RAG) AI assistant for organizations.",
    version="0.1.0"
)

# --- JWT Configuration (for authentication) ---
# Define this immediately after app initialization to ensure global scope for reloader
SECRET_KEY = "your-super-secret-jwt-key" # CHANGE THIS IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Defined here

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "file://",
    "null",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Embedding Model Initialization ---
embedding_model = None
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure you have an active internet connection to download the model.")


# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. AI responses will not work.")
    print("Please set it like: export GEMINI_API_KEY='YOUR_API_KEY' (Linux/macOS) or $env:GEMINI_API_KEY='YOUR_API_KEY' (Windows PowerShell)")


# --- Database Setup (SQLite for initial development) ---
# Corrected path to avoid SyntaxWarning
DATABASE_URL = "./database.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    print(f"DEBUG: Attempting to connect to database at: {DATABASE_URL}")
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    print("DEBUG: Database connection established.")
    return conn

def create_tables():
    """Creates necessary database tables if they don't exist."""
    print("DEBUG: create_tables() function called.")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        """)
        print("DEBUG: 'users' table checked/created.")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                department TEXT,
                uploaded_by INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (uploaded_by) REFERENCES users(id)
            )
        """)
        print("DEBUG: 'documents' table checked/created.")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        print("DEBUG: 'document_chunks' table checked/created.")
        conn.commit()
        print("DEBUG: Database tables created/verified and committed.")
    except Exception as e:
        print(f"ERROR: Failed to create database tables: {e}")
    finally:
        if conn:
            conn.close()
            print("DEBUG: Database connection closed after table creation.")


@app.on_event("startup")
async def startup_event():
    print("DEBUG: Application startup event triggered.")
    print(f"DEBUG: Current working directory is: {os.getcwd()}")
    create_tables()
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('admin', 'adminpass', 'general_manager'))
            conn.commit()
            print("Default admin user created: admin/adminpass")
        cursor.execute("SELECT * FROM users WHERE username = 'dept_head'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('dept_head', 'deptpass', 'department_head'))
            conn.commit()
            print("Default department head user created: dept_head/deptpass")
    except Exception as e:
        print(f"ERROR: Failed to add default users: {e}")
    finally:
        if conn:
            conn.close()
            print("DEBUG: Database connection closed after user creation.")


# --- RAG Helper Functions ---

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        text = ""
    return text

def extract_text_from_txt(txt_path: str) -> str:
    text = ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error extracting text from TXT {txt_path}: {e}")
        text = ""
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    chunks = []
    if not text:
        return chunks

    words = text.split()
    if not words:
        return chunks

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - chunk_overlap)
        if i >= len(words) and (len(words) - i < chunk_overlap):
            break
    return chunks


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if embedding_model is None:
        print("Embedding model not loaded. Cannot generate embeddings.")
        return []
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()

def get_similar_chunks(query_embedding: List[float], user_role: str, top_k: int = 3) -> List[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT dc.chunk_text, dc.embedding, d.department FROM document_chunks dc JOIN documents d ON dc.document_id = d.id")
    all_chunks_data = cursor.fetchall()
    conn.close()

    if not all_chunks_data:
        print("DEBUG: No document chunks found in database for similarity search.")
        return []

    query_embedding_np = np.array(query_embedding, dtype=np.float32)
    
    similarities = []
    for chunk_data in all_chunks_data:
        chunk_text = chunk_data['chunk_text']
        stored_embedding_blob = chunk_data['embedding']
        document_department = chunk_data['department']

        stored_embedding_np = np.frombuffer(stored_embedding_blob, dtype=np.float32)

        norm_query = np.linalg.norm(query_embedding_np)
        norm_stored = np.linalg.norm(stored_embedding_np)

        if norm_query == 0 or norm_stored == 0:
            similarity = 0.0
        else:
            similarity = np.dot(query_embedding_np, stored_embedding_np) / (norm_query * norm_stored)
        
        similarities.append((similarity, chunk_text, document_department))

    similarities.sort(key=lambda x: x[0], reverse=True)

    retrieved_chunks = []
    print(f"DEBUG: User Role in get_similar_chunks: {user_role}")

    for sim, chunk_text, department in similarities:
        if user_role == 'general_manager':
            retrieved_chunks.append(chunk_text)
        elif not department:
            retrieved_chunks.append(chunk_text)
        
        if len(retrieved_chunks) >= top_k:
            break
            
    return retrieved_chunks


def call_gemini_api(prompt: str) -> str:
    """Calls the Gemini API with the given prompt."""
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Cannot call Gemini API.")
        return "عذراً، مساعد الذكاء الاصطناعي غير متاح حالياً بسبب نقص التهيئة (مفتاح API)."

    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "text/plain"
        }
    }
    
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print("Gemini API response did not contain expected text content:", result)
            return "عذراً، لم أتمكن من توليد إجابة في الوقت الحالي. يرجى المحاولة لاحقاً."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "عذراً، حدث خطأ أثناء الاتصال بالمساعد الذكي. يرجى المحاولة لاحقاً."
    except json.JSONDecodeError as e:
        print(f"Error decoding Gemini API response JSON: {e}")
        return "عذراً، حدث خطأ في معالجة استجابة المساعد الذكي."


# --- Pydantic Models for Request/Response Data ---
class User(BaseModel):
    username: str
    role: str
    id: Optional[int] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None

class Message(BaseModel):
    message: str

class AIMessage(BaseModel):
    response: str

class UploadStatus(BaseModel):
    filename: str
    status: str
    message: str


# --- JWT Helper Functions ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        role: str = payload.get("role")
        if username is None or user_id is None or role is None:
            raise credentials_exception
        token_data = TokenData(username=username, user_id=user_id, role=role)
    except jwt.PyJWTError:
        raise credentials_exception
    return token_data

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_access_token(token, credentials_exception)
    conn = get_db_connection()
    cursor = conn.cursor()
    user_row = cursor.execute("SELECT * FROM users WHERE id = ?", (token_data.user_id,)).fetchone()
    conn.close()
    if user_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found in database",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(id=user_row['id'], username=user_row['username'], role=user_row['role'])


# --- API Endpoints ---

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    cursor = conn.cursor()
    user_row = cursor.execute("SELECT * FROM users WHERE username = ?", (form_data.username,)).fetchone()
    conn.close()

    if not user_row or user_row['password'] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_row['username'], "user_id": user_row['id'], "role": user_row['role']},
        expires_delta=access_token_expires
    )
    
    print(f"DEBUG: Generated access token: {access_token[:10]}... (first 10 chars)")
    print(f"DEBUG: Returning payload: {{'access_token': '...', 'token_type': 'bearer'}}")

    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/chat", response_model=AIMessage)
async def chat_with_ai(message: Message, current_user: User = Depends(get_current_user)):
    print(f"User '{current_user.username}' (Role: {current_user.role}) asked: {message.message}")

    if embedding_model is None:
        return AIMessage(response="عذراً، نظام الذكاء الاصطناعي غير جاهز (نموذج التضمين غير محمل).")
    if not GEMINI_API_KEY:
        return AIMessage(response="عذراً، نظام الذكاء الاصطناعي غير جاهز (مفتاح API غير موجود).")
    
    # Generate embedding for the user's query
    query_embedding = generate_embeddings([message.message])[0]

    # Retrieve relevant document chunks based on query and user's role/permissions
    retrieved_chunks = get_similar_chunks(query_embedding, current_user.role, top_k=3)
    
    context = "\n".join(retrieved_chunks)

    if not context:
        # Prompt without specific language instruction
        prompt = (
            f"You are an internal AI assistant for the organization. Answer the following question in a professional and institutional tone. "
            f"If the question is not related to internal organizational data, state that you do not have information about it. "
            f"Respond in the same language as the user's input, if possible." # Added instruction for language
            f"\n\nQuestion: {message.message}"
        )
        print("DEBUG: No relevant context found. Sending query to LLM without specific context.")
    else:
        # Prompt with context, without specific language instruction
        prompt = (
            f"You are an internal AI assistant for the organization. Use the following contextual information to answer the question. "
            f"Respond in a professional and institutional tone. If the information is insufficient to answer, clearly state that. "
            f"Do not invent information not present in the context. Respond in the same language as the user's input." # Added instruction for language
            f"\n\nContextual Information:\n{context}\n\nQuestion: {message.message}"
        )
        print(f"DEBUG: Relevant context found. Context length: {len(context)} chars. Chunks used: {len(retrieved_chunks)}")

    # Call Gemini API with the enriched prompt
    ai_response_text = call_gemini_api(prompt)

    return AIMessage(response=ai_response_text)

@app.post("/upload-document/", response_model=UploadStatus)
async def upload_document(
    file: UploadFile = File(...),
    department: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["department_head", "general_manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only department heads or general managers can upload documents."
        )

    upload_dir = "uploaded_documents"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)

    document_id = None
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (name, file_path, department, uploaded_by) VALUES (?, ?, ?, ?)",
            (file.filename, file_location, department, current_user.id)
        )
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()

        extracted_text = ""
        if file.filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_location)
        elif file.filename.lower().endswith('.txt'):
            extracted_text = extract_text_from_txt(file_location)
        else:
            print(f"Warning: Unsupported file type for text extraction: {file.filename}. Skipping text processing.")
            return UploadStatus(
                filename=file.filename,
                status="warning",
                message=f"Document '{file.filename}' uploaded, but text extraction not supported for this file type. It will not be used by AI for RAG."
            )

        if not extracted_text:
            print(f"Warning: No text could be extracted from {file.filename}. It will not be used by AI.")
            return UploadStatus(
                filename=file.filename,
                status="warning",
                message=f"Document '{file.filename}' uploaded, but no text could be extracted. It will not be used by AI for RAG."
            )

        chunks = chunk_text(extracted_text)
        print(f"DEBUG: Document '{file.filename}' chunked into {len(chunks)} pieces.")

        if embedding_model is None:
            raise Exception("Embedding model not loaded, cannot process document chunks.")
        
        chunk_embeddings = generate_embeddings(chunks)
        print(f"DEBUG: Generated embeddings for {len(chunk_embeddings)} chunks.")

        conn = get_db_connection()
        cursor = conn.cursor()
        for i, chunk in enumerate(chunks):
            embedding_bytes = np.array(chunk_embeddings[i], dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO document_chunks (document_id, chunk_text, embedding, chunk_index) VALUES (?, ?, ?, ?)",
                (document_id, chunk, embedding_bytes, i)
            )
        conn.commit()
        conn.close()

        print(f"DEBUG: Document '{file.filename}' chunks and embeddings stored successfully.")

        return UploadStatus(
            filename=file.filename,
            status="success",
            message=f"Document '{file.filename}' uploaded, processed, and ready for AI retrieval."
        )
    except Exception as e:
        print(f"Error during file upload or processing: {e}")
        if os.path.exists(file_location):
            os.remove(file_location)
        if document_id:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
                conn.commit()
                conn.close()
                print(f"DEBUG: Cleaned up document metadata and chunks for document_id {document_id}")
            except Exception as cleanup_e:
                print(f"ERROR: Failed to clean up database entries for failed upload: {cleanup_e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not process document: {e}"
        )


# --- Serve Static Files (Frontend) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(current_dir, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# --- Running the FastAPI Application ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
