from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import json
import random
import fitz  # PyMuPDF
from utils import calculate_reading_score, analyze_handwriting

app = FastAPI(title="Oliveboard RBI LPT Evaluator")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
os.makedirs("temp", exist_ok=True)

# Load Database
# We structure DB to hold both passages and writing topics
try:
    with open("passages.json", "r", encoding="utf-8") as f:
        DB = json.load(f)
        # Handle different JSON structures safely
        PASSAGES_DB = DB.get("reading_passages", DB) 
        TOPICS_DB = DB.get("writing_topics", {})
except FileNotFoundError:
    PASSAGES_DB = {}
    TOPICS_DB = {}
    print("WARNING: passages.json not found!")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- API: GET RANDOM READING PASSAGE ---
@app.get("/api/get-passage/{language_code}")
def get_passage(language_code: str):
    if language_code not in PASSAGES_DB:
        raise HTTPException(status_code=404, detail="Language not found")
    
    selected_text = random.choice(PASSAGES_DB[language_code])
    return {"text": selected_text}

# --- NEW API: GET WRITING TOPICS ---
@app.get("/api/get-writing-topics/{language_code}")
def get_writing_topics(language_code: str):
    if language_code not in TOPICS_DB:
        # Fallback topics if language specific ones aren't found
        return {"topics": ["My Country", "My Favorite Festival", "Environmental Pollution"]}
    
    # Return up to 3 random topics
    topics = random.sample(TOPICS_DB[language_code], min(3, len(TOPICS_DB[language_code])))
    return {"topics": topics}

# --- API: READING TEST ---
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

# --- API: WRITING TEST (PDF + IMAGE SUPPORT) ---
@app.post("/api/evaluate-writing")
async def evaluate_writing(
    file: UploadFile = File(...), # Matches 'file' in Frontend FormData
    language: str = Form(...)
):
    try:
        file_bytes = await file.read()
        
        # PDF Handling
        if file.filename.lower().endswith('.pdf'):
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if len(doc) < 1:
                     return JSONResponse(content={"error": "Empty PDF"}, status_code=400)
                
                # Convert First Page to Image (PNG)
                page = doc.load_page(0) 
                pix = page.get_pixmap(dpi=150)
                image_bytes = pix.tobytes("png")
            except Exception as pdf_err:
                 return JSONResponse(content={"error": f"PDF Error: {str(pdf_err)}"}, status_code=500)
        else:
            # Assume it is already an image (jpg/png)
            image_bytes = file_bytes

        # Fetch Topics for Validity Check
        expected_topics = TOPICS_DB.get(language, [])

        # Send to AI
        ai_response = analyze_handwriting(image_bytes, language, expected_topics)
        
        return JSONResponse(content=ai_response)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)