import os
import io
import json
import base64
import requests
import numpy as np
import pandas as pd
import duckdb
from typing import List, Tuple, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import time
import mimetypes
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

UPLOAD_DIR = "./uploaded_files"
PAGE_DIR = "./pages"
EMBED_DIR = "./embeddings"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PAGE_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

app = FastAPI(title="Dynamic URL + Questions + LLM + DuckDB + Image Support")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"]

DUCKDB_PATH = os.path.join(EMBED_DIR, "embeddings.duckdb")
con = duckdb.connect(DUCKDB_PATH)
con.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER,
    text STRING,
    vector BLOB
)
""")

def read_questions_file(file_bytes: bytes) -> Tuple[str, List[str]]:
    try:
        content = file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file encoding. Must be UTF-8 text.")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="File is empty or invalid format.")
    url = lines[0]
    questions = lines[1:]
    return url, questions

def fetch_url_text_and_tables(url: str) -> Tuple[str, List[pd.DataFrame]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; QuestionsBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "svg", "img"]):
        t.decompose()
    text = "\n".join([line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()])
    tables: List[pd.DataFrame] = []
    try:
        for df in pd.read_html(resp.text):
            tables.append(df)
    except Exception:
        pass
    return text, tables

def save_page_text(url: str, text: str) -> str:
    safe_name = url.replace("://", "_").replace("/", "_")[:150]
    path = os.path.join(PAGE_DIR, f"{safe_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def chunk_text(text: str, max_chunk_tokens: int = 250) -> List[str]:
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks = []
    for block in blocks:
        words = block.split()
        start = 0
        while start < len(words):
            end = min(start + max_chunk_tokens, len(words))
            chunks.append(" ".join(words[start:end]))
            start = end
    return chunks

def build_duckdb_index(chunks: List[str]):
    if not chunks:
        return None
    embs = embedding_model.encode(chunks, convert_to_numpy=True)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    con.execute("DELETE FROM embeddings")  
    for i, vec in enumerate(embs):
        con.execute("INSERT INTO embeddings VALUES (?, ?, ?)", (i, chunks[i], vec.astype(np.float32).tobytes()))
    return chunks

def search_similar_chunks(query: str, index, chunks: List[str], top_k: int = 5):
    q_emb = embedding_model.encode([query], convert_to_numpy=True)[0]
    q_emb /= np.linalg.norm(q_emb) + 1e-12
    res = con.execute("SELECT chunk_id, text, vector FROM embeddings").fetchall()
    sims = []
    for chunk_id, text, vec_bytes in res:
        vec = np.frombuffer(vec_bytes, dtype=np.float32)
        score = float(np.dot(q_emb, vec))
        sims.append((text, score))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

def truncate_context(context: str, max_chars: int = 160000):
    return context if len(context) <= max_chars else context[:max_chars] + "\n...[truncated]..."

def detect_plot_request(question: str) -> bool:
    kws = ["plot", "scatter", "scatterplot", "chart", "draw scatter"]
    qlow = question.lower()
    return any(k in qlow for k in kws)

def select_xy_from_tables(question: str, tables: List[pd.DataFrame]):
    qlow = question.lower()
    for df in tables:
        cols = list(df.columns)
        best_pair = None
        for c1 in cols:
            for c2 in cols:
                if c1 == c2:
                    continue
                c1_nums = pd.to_numeric(df[c1], errors="coerce")
                c2_nums = pd.to_numeric(df[c2], errors="coerce")
                if c1_nums.notna().sum() >= max(1, len(df)//4) and c2_nums.notna().sum() >= max(1, len(df)//4):
                    name_match = str(c1).lower() in qlow or str(c2).lower() in qlow
                    if name_match:
                        return c1_nums.dropna(), c2_nums.dropna(), f"columns '{c1}' and '{c2}' matched"
                    if not best_pair:
                        best_pair = (c1_nums.dropna(), c2_nums.dropna(), f"columns '{c1}' and '{c2}' (first numeric pair)")
        if best_pair:
            return best_pair
    return None, None, "no suitable numeric columns"

def create_scatter_base64(x: List[float], y: List[float], title: str = "Scatter"):
    x = np.array(x)
    y = np.array(y)
    slope = intercept = corr = None
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        corr = float(np.corrcoef(x, y)[0,1])
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)
    ax.scatter(x, y, alpha=0.8, s=40, edgecolor="k")
    if slope is not None:
        xs = np.linspace(min(x), max(x), 200)
        ax.plot(xs, slope*xs + intercept, "--", color="red", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.4)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", slope, intercept, corr

def is_image_question(q: str) -> bool:
    ql = q.lower()
    keywords = ["image", "photo", "picture", "identify", "what is", "what's in", "who is", "read", "ocr", "color", "how many", "is this", "does this", "look", "see"]
    return any(k in ql for k in keywords)

def call_openrouter_chat(api_key: str, model: str, question: str, context_text: str, tables_json: List[dict]):
    context_text = truncate_context(context_text)
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    system_prompt = (
        "You are an advanced data analyst assistant. "
        "Analyze CONTEXT, TABLES, QUESTION and give a precise one-line answer. "
        "Strict JSON format: {\"answers\": [\"...\"], \"computed\": {}, \"image\": null, \"sources\": [], \"warnings\": [], \"requires_external_data\": false}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nTABLES:\n{json.dumps(tables_json)}\n\nQUESTION:\n{question}"}
        ],
        "max_tokens": 800
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenRouter error: {resp.status_code} {resp.text}")
    try:
        content = resp.json()["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return json.loads(content)
        return content
    except Exception:
        return {"raw_response": resp.json()}

def call_openrouter_image(api_key: str, image_b64: str, questions: List[str], mime_type: Optional[str] = None):
    """
    Sends the image to the model as an inline data URL using image_url format which Gemini/OpenRouter expects.
    Retries up to 4 times. Each attempt has a 5-minute timeout.
    mime_type: e.g. "image/png", "image/jpeg", "image/webp" - if None defaults to image/png.
    """
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }

    if not mime_type:
        mime_type = "image/png"
    # sanitize: if content-type contains charset or other extra data, take only the main type
    if ";" in mime_type:
        mime_type = mime_type.split(";")[0].strip()

    system_prompt = (
        "You are an expert AI vision assistant. "
        "Carefully analyze ONLY the provided image and directly answer the user's questions. "
        "Do not guess. If the detail is not visible in the image, respond with 'Not visible in the image'. "
        "Return answers strictly in JSON format like: "
        "{\"answers\": [\"...\"], \"computed\": {}, \"image\": null, \"sources\": [], \"warnings\": [], \"requires_external_data\": false}"
    )

    questions_str = "\n".join([f"- {q}" for q in questions])

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Please analyze this uploaded image and answer:\n{questions_str}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    last_exc = None
    for attempt in range(1, 5):  
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300 
            )
            if resp.status_code == 200:
                data = resp.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    if isinstance(content, str):
                        content = content.replace("```json", "").replace("```", "").strip()
                        return json.loads(content)
                    return content
                except Exception:
                    return {"raw_response": data}
            else:
                last_exc = Exception(f"OpenRouter error: {resp.status_code} {resp.text}")
        except Exception as e:
            last_exc = e

        if attempt < 4:
            time.sleep(2)

    raise HTTPException(status_code=500, detail=f"Image API failed after 4 attempts: {last_exc}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    
@app.post("/api")
async def upload_questions(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form(...),
    top_k: int = Form(5),
    csv_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    start_time = time.time()

    for folder in [UPLOAD_DIR, PAGE_DIR]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files allowed.")
    
    content = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    content_str = content.decode("utf-8", errors="ignore").strip()
    lines = [line.strip() for line in content_str.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="questions.txt is empty")

    url = None
    for line in lines:
        if line.startswith("http://") or line.startswith("https://"):
            url = line
            break

    questions = [l for l in lines if not (l.startswith("http://") or l.startswith("https://"))]

    image_b64 = None
    image_mime = None
    if image_file:
        if image_file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported image type")
        image_bytes = await image_file.read()
        image_b64 = image_to_base64(image_bytes)
        image_mime = image_file.content_type or mimetypes.guess_type(image_file.filename)[0] or "image/png"

    tables: List[pd.DataFrame] = []
    page_text = ""
    page_file = None
    if url:
        page_text, url_tables = fetch_url_text_and_tables(url)
        page_file = save_page_text(url, page_text)
        tables.extend(url_tables)
        if csv_file:
            csv_bytes = await csv_file.read()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            tables.append(df)
        chunks = chunk_text(page_text) if page_text else []
    else:
        if csv_file:
            csv_bytes = await csv_file.read()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            tables = [df]
        chunks = []
        page_text = ""



    index = build_duckdb_index(chunks) if chunks else None

    tables_json = [{"columns": df.columns.tolist(), "rows": df.fillna("").to_dict(orient="records")} for df in tables]

    results = []
    for q in questions:
        try:
            if detect_plot_request(q):
                x_ser, y_ser, reason = select_xy_from_tables(q, tables)
                if x_ser is not None and y_ser is not None and len(x_ser) >= 2:
                    n = min(len(x_ser), len(y_ser))
                    x_vals = pd.to_numeric(x_ser.iloc[:n], errors="coerce").dropna().tolist()
                    y_vals = pd.to_numeric(y_ser.iloc[:n], errors="coerce").dropna().tolist()
                    img_uri, slope, intercept, corr = create_scatter_base64(x_vals, y_vals, title=q)
                    results.append({
                        "question": q,
                        "answer": None,
                        "image": img_uri,
                        "computed": {"slope": slope, "intercept": intercept, "correlation": corr},
                        "context_used": [reason],
                        "warnings": []
                    })
                    continue

            context_for_llm = ""
            similar = []
            if index:
                similar = search_similar_chunks(q, index, chunks, top_k)
                context_for_llm = "\n\n".join([c for c, s in similar])

            if image_b64 and is_image_question(q):
                img_response = call_openrouter_image(api_key, image_b64, [q], image_mime)
                results.append({
                    "question": q,
                    "context_used": [{"chunk": c, "score": s} for c, s in similar],
                    "answer": img_response
                })
                continue

            llm_out = call_openrouter_chat(api_key, model, q, context_for_llm, tables_json)
            results.append({
                "question": q,
                "context_used": [{"chunk": c, "score": s} for c, s in similar],
                "answer": llm_out
            })
        except Exception as e:
            results.append({"question": q, "error": str(e)})

    return JSONResponse(content={
        "mode": "web_or_csv_image_analysis",
        "elapsed_time_seconds": time.time() - start_time,
        "saved_as": file_path,
        "url": url,
        "page_saved_as": page_file if url else None,
        "csv_used": (csv_file.filename if csv_file else None),
        "questions_count": len(questions),
        "results": results
    })
    