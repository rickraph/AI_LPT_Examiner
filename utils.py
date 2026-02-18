import os
import json
import re
import string
import time
import difflib
import base64
import subprocess
import Levenshtein
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

LANG_NAMES = {
    'ml': 'Malayalam', 'te': 'Telugu', 'kn': 'Kannada',
    'hi': 'Hindi',     'ta': 'Tamil',  'bn': 'Bengali',
    'en': 'English',
}

WPM_THRESHOLD         = {'hi': 40, 'ml': 20, 'te': 20, 'ta': 20, 'kn': 20}
DEFAULT_WPM_THRESHOLD = 22
GEMINI_FLASH          = "gemini-2.0-flash"

# Retry settings for 429 / 503 transient errors
MAX_RETRIES  = 4
RETRY_DELAYS = [2, 5, 10, 20]   # seconds between retries (exponential-ish)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\([A-Za-z0-9\s]+\)', '', text)
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation + "।॥")
    return " ".join(text.translate(translator).split())


def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True, timeout=10
        )
        val = result.stdout.strip()
        if val:
            return float(val)
    except Exception:
        pass
    try:
        return max(os.path.getsize(audio_path) / 1024 / 16, 1.0)
    except Exception:
        return 30.0


def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return raw.strip()


def detect_image_mime(image_bytes: bytes) -> str:
    if image_bytes[:4] == b'\x89PNG':
        return 'image/png'
    if image_bytes[:2] == b'\xff\xd8':
        return 'image/jpeg'
    return 'image/png'


def gemini_generate_with_retry(model, contents, generation_config):
    """
    Call model.generate_content() with automatic retry on 429/503.
    Waits RETRY_DELAYS[attempt] seconds before each retry.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return model.generate_content(
                contents=contents,
                generation_config=generation_config,
            )
        except (ResourceExhausted, ServiceUnavailable) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAYS[attempt]
                print(f"[Gemini] Rate limit hit (attempt {attempt+1}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise   # non-retryable errors bubble up immediately
    raise last_exc


# ─────────────────────────────────────────────────────────────────────
# READING — Gemini 2.0 Flash Audio
# ─────────────────────────────────────────────────────────────────────

def evaluate_reading_gemini(audio_path: str, reference_text: str,
                             language_code: str) -> dict:
    lang_name      = LANG_NAMES.get(language_code, language_code.upper())
    ref_word_count = len(reference_text.split())
    duration       = get_audio_duration(audio_path)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    ext = os.path.splitext(audio_path)[1].lower()
    mime_map = {
        '.webm': 'audio/webm', '.mp4': 'audio/mp4',
        '.mp3':  'audio/mpeg', '.wav': 'audio/wav',
        '.ogg':  'audio/ogg',  '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
    }
    mime_type = mime_map.get(ext, 'audio/webm')

    prompt = f"""You are a strict {lang_name} language examiner for the RBI Language Proficiency Test.

The candidate must read this COMPLETE passage aloud:

REFERENCE PASSAGE ({ref_word_count} words):
{reference_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — TRANSCRIBE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write exactly what you hear in {lang_name} script. Nothing more, nothing less.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — COUNT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- words_spoken   = how many words from the reference were actually spoken
- missed_words   = reference words NOT spoken (stopping early = ALL remaining words missed)
- extra_words    = words spoken NOT in the reference (filler, repetitions, wrong words)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — CALCULATE SCORE (strictly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
base  = (words_spoken / {ref_word_count}) * 100
deduct 2 points for each extra/wrong word spoken
deduct 1 point for each clear stumble or re-read
score = max(0, min(100, base - extra_penalty - fluency_penalty))

EXAMPLES (so you understand the strictness required):
- Read 1 sentence of 3 → words_spoken ≈ 33% → score ≈ 30-35
- Read 2 sentences of 3 → words_spoken ≈ 66% → score ≈ 60-65
- Read all 3 sentences, no errors → score = 95-100
- Read all 3 sentences, 3 wrong words → score ≈ 88-92
- Read all 3 sentences, 3 wrong + 3 stumbles → score ≈ 83-87

DO NOT be generous. DO NOT assume unread words were read.
If the transcription ends mid-passage, those words are MISSED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON, no markdown:
{{
  "transcription": "<exactly what you heard in {lang_name} script>",
  "score": <integer 0-100, calculated strictly per formula above>,
  "words_spoken": <integer — count of reference words actually spoken>,
  "missed_words": <list of reference words NOT spoken>,
  "extra_words": <list of wrong/extra words spoken that are NOT in reference>,
  "feedback": "<one honest sentence: exactly what was completed and what was missed>"
}}"""

    model    = genai.GenerativeModel(GEMINI_FLASH)
    response = gemini_generate_with_retry(
        model,
        contents=[{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(audio_bytes).decode()
                }},
            ]
        }],
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    result      = json.loads(clean_json(response.text))
    score       = float(result.get("score", 0))
    user_text   = result.get("transcription", "")
    words_spoken = int(result.get("words_spoken", len(user_text.split())))
    wpm         = round((words_spoken / duration) * 60, 1) if duration > 0 else 0
    wpm_min     = WPM_THRESHOLD.get(language_code, DEFAULT_WPM_THRESHOLD)

    return {
        "score": round(score, 2),
        "wpm": wpm,
        "status": "Pass" if score >= 75 and wpm >= wpm_min else "Fail",
        "user_transcription": user_text,
        "feedback": result.get("feedback", ""),
        "metrics": {
            "missed_words_list": result.get("missed_words", []),
            "extra_words_list":  result.get("extra_words", []),
        },
    }


def calculate_reading_score(reference_text: str, audio_path: str,
                             language_code: str) -> dict:
    try:
        return evaluate_reading_gemini(audio_path, reference_text, language_code)
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# WRITING — Gemini 2.0 Flash Vision (topic-aware)
# ─────────────────────────────────────────────────────────────────────

def analyze_handwriting(image_bytes: bytes, language_code: str,
                        expected_topics_list=None) -> dict:
    try:
        lang_name = LANG_NAMES.get(language_code, language_code.upper())

        if expected_topics_list:
            topics_block = "\n".join(
                f"  {i+1}. {t}" for i, t in enumerate(expected_topics_list)
            )
            topic_section = f"""
AVAILABLE TOPICS (candidate chose ONE to write about):
{topics_block}

Identify which topic was written, then evaluate content relevance to THAT specific topic."""
        else:
            topic_section = "\nIdentify the topic the candidate wrote about."

        prompt = f"""You are a strict but fair {lang_name} language examiner for the RBI Language Proficiency Test (LPT).

Carefully examine this handwritten answer sheet image.
{topic_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. TRANSCRIPTION
   Read every word and transcribe the COMPLETE handwritten text
   exactly as written in {lang_name} script. Do NOT correct anything.

B. SPELLING ERRORS
   Check every word against standard {lang_name} spelling.
   List each as: "misspelled → correct"
   If no errors, return empty list.

C. GRAMMAR ERRORS
   List each grammatical mistake in English (wrong verb form,
   wrong case, word order, missing words, etc.)
   If no errors, return empty list.

D. SCORES (each out of 10):

   legibility_score: How clearly is the handwriting written?
     10 = perfectly clear, 1 = barely readable

   content_score: Is the content relevant and substantial?
     10 = fully on-topic, rich ideas, 5+ sentences
     5  = somewhat related, thin content
     1  = off-topic or only 1-2 sentences

   language_score: Vocabulary, sentence variety, fluency
     10 = varied vocab, complex sentences, natural flow
     5  = basic vocab, simple sentences only
     1  = very limited language

   final_score: Weighted average
     = (content_score × 0.35) + (language_score × 0.30)
     + ((10 - spelling_errors_count × 0.5) × 0.20)
     + ((10 - grammar_errors_count × 0.5) × 0.15)
     Minimum 1, Maximum 10. Round to 1 decimal.

E. FEEDBACK
   Write 3-4 sentences of SPECIFIC, ACTIONABLE feedback in English.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON, no markdown:
{{
  "topic_identified": "<exact topic name written about>",
  "transcription": "<complete text exactly as written>",
  "spelling_errors": ["misspelled → correct", ...],
  "grammar_errors": ["description of error", ...],
  "legibility_score": <integer 0-10>,
  "content_score": <number 0-10>,
  "language_score": <number 0-10>,
  "final_score": <number 0-10>,
  "feedback": "<3-4 sentences of specific feedback in English>"
}}"""

        model     = genai.GenerativeModel(GEMINI_FLASH)
        mime_type = detect_image_mime(image_bytes)

        response = gemini_generate_with_retry(
            model,
            contents=[{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode()
                    }},
                ]
            }],
            generation_config=genai.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        return json.loads(clean_json(response.text))

    except Exception as e:
        return {"error": str(e)}