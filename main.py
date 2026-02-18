from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import json
import random
import fitz  # PyMuPDF
from utils import calculate_reading_score, analyze_handwriting

app = FastAPI(title="Oliveboard RBI LPT Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
os.makedirs("temp", exist_ok=True)

try:
    with open("passages.json", "r", encoding="utf-8") as f:
        DB          = json.load(f)
        PASSAGES_DB = DB.get("reading_passages", DB)
        TOPICS_DB   = DB.get("writing_topics", {})
except FileNotFoundError:
    PASSAGES_DB = {}
    TOPICS_DB   = {}
    print("WARNING: passages.json not found!")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/get-passage/{language_code}")
def get_passage(language_code: str):
    if language_code not in PASSAGES_DB:
        raise HTTPException(status_code=404, detail="Language not found")
    return {"text": random.choice(PASSAGES_DB[language_code])}

@app.get("/api/get-writing-topics/{language_code}")
def get_writing_topics(language_code: str):
    if language_code not in TOPICS_DB:
        return {"topics": ["My Country", "My Favorite Festival", "Environmental Pollution"]}
    topics = random.sample(TOPICS_DB[language_code], min(3, len(TOPICS_DB[language_code])))
    return {"topics": topics}

@app.post("/api/evaluate-reading")
async def evaluate_reading(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...),
    language: str = Form(...)
):
    try:
        file_location = f"temp/{audio_file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        result = calculate_reading_score(reference_text, file_location, language)
        os.remove(file_location)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/evaluate-writing")
async def evaluate_writing(
    file: UploadFile = File(...),
    language: str = Form(...)
):
    try:
        file_bytes  = await file.read()
        image_bytes = None
        filename    = (file.filename or "").lower()

        # ── PDF → PNG ────────────────────────────────────────────────
        if filename.endswith('.pdf'):
            doc    = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) < 1:
                return JSONResponse(content={"error": "Empty PDF"}, status_code=400)
            page   = doc.load_page(0)
            zoom   = 150 / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix    = page.get_pixmap(matrix=matrix, alpha=False)
            tmp    = "temp/_writing_page.png"
            pix.save(tmp)
            with open(tmp, "rb") as f:
                image_bytes = f.read()
            os.remove(tmp)
            doc.close()
        else:
            image_bytes = file_bytes

        if not image_bytes:
            return JSONResponse(
                content={"error": "Could not extract image from file."},
                status_code=400
            )

        expected_topics = TOPICS_DB.get(language, [])
        ai_response     = analyze_handwriting(image_bytes, language, expected_topics)
        return JSONResponse(content=ai_response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)