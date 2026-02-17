import os
import json
import re
import string
import difflib
import base64
import subprocess
import Levenshtein
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WHISPER_LANGUAGES   = {'hi', 'en', 'ta', 'bn', 'gu', 'pa', 'ur'}
GPT_AUDIO_LANGUAGES = {'ml', 'te', 'kn', 'mr'}

LANG_NAMES = {
    'ml': 'Malayalam', 'te': 'Telugu', 'kn': 'Kannada',
    'hi': 'Hindi',     'ta': 'Tamil',  'bn': 'Bengali',
}

WPM_THRESHOLD         = {'hi': 40, 'ml': 20, 'te': 20}
DEFAULT_WPM_THRESHOLD = 22


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


def convert_to_mp3(audio_path: str) -> str:
    """
    Convert any browser audio (webm/ogg/mp4) to mp3 via ffmpeg.
    GPT-4o Audio only accepts 'wav' or 'mp3'.
    Returns path to the new mp3 file.
    """
    mp3_path = audio_path.rsplit('.', 1)[0] + '_conv.mp3'
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', audio_path,
         '-ar', '16000', '-ac', '1', '-b:a', '64k',
         mp3_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[-300:]}")
    return mp3_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration in seconds via ffprobe, fall back to size estimate."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        pass
    try:
        size_kb = os.path.getsize(audio_path) / 1024
        return max(size_kb / 16, 1.0)
    except Exception:
        return 30.0


# ─────────────────────────────────────────────────────────────────────
# PATH A: Hindi / English — Whisper + string diff
# ─────────────────────────────────────────────────────────────────────

def transcribe_whisper(audio_path: str, reference_text: str,
                       language_code: str):
    with open(audio_path, "rb") as f:
        args = {
            "model": "whisper-1",
            "file": f,
            "response_format": "verbose_json",
            "prompt": reference_text.strip(),
        }
        if language_code in WHISPER_LANGUAGES:
            args["language"] = language_code
        resp = client.audio.transcriptions.create(**args)
    return resp.text, resp.duration


def evaluate_with_diff(reference_text: str, user_text: str,
                       language_code: str, wpm: float) -> dict:
    ref_clean  = normalize_text(reference_text)
    user_clean = normalize_text(user_text)
    ref_words  = ref_clean.split()
    user_words = user_clean.split()

    matcher       = difflib.SequenceMatcher(None, ref_words, user_words)
    missing_words = []
    extra_words   = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            missing_words.extend(ref_words[i1:i2])
        elif tag == 'insert':
            extra_words.extend(user_words[j1:j2])
        elif tag == 'replace':
            lev = Levenshtein.ratio(
                "".join(ref_words[i1:i2]),
                "".join(user_words[j1:j2])
            )
            if lev < 0.70:
                missing_words.extend(ref_words[i1:i2])
                extra_words.extend(user_words[j1:j2])

    total_ref   = len(ref_words)
    missed      = len(missing_words)
    extra       = len(extra_words)
    coverage    = ((total_ref - missed) / total_ref * 100) if total_ref > 0 else 0
    noise_pen   = min(extra * 3, 30)
    pron_score  = Levenshtein.ratio(ref_clean, user_clean) * 100
    final_score = round(max((coverage * 0.6) + (pron_score * 0.4) - noise_pen, 0), 2)

    wpm_min = WPM_THRESHOLD.get(language_code, DEFAULT_WPM_THRESHOLD)
    return {
        "score": final_score,
        "wpm": wpm,
        "status": "Pass" if final_score >= 75 and wpm >= wpm_min else "Fail",
        "user_transcription": user_text,
        "feedback": "",
        "metrics": {
            "missed_words_list": missing_words,
            "extra_words_list":  extra_words,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# PATH B: Malayalam / Telugu — GPT-4o Audio (single call)
# ─────────────────────────────────────────────────────────────────────

def evaluate_with_gpt_audio(audio_path: str, reference_text: str,
                             language_code: str) -> dict:
    """
    Convert audio to mp3 → send to GPT-4o Audio with reference text.
    GPT-4o Audio natively understands ML/TE speech, transcribes accurately,
    and scores holistically — no Whisper phonetic drift, no hallucinations.
    """
    lang_name      = LANG_NAMES.get(language_code, language_code.upper())
    ref_word_count = len(reference_text.split())

    # Get duration BEFORE conversion (original file)
    duration = get_audio_duration(audio_path)

    # Convert webm/ogg → mp3  (GPT-4o Audio only accepts wav or mp3)
    mp3_path = None
    try:
        mp3_path    = convert_to_mp3(audio_path)
        read_path   = mp3_path
        audio_fmt   = "mp3"
    except Exception as conv_err:
        # If ffmpeg fails, try sending original (works if already mp3/wav)
        read_path = audio_path
        ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
        audio_fmt = ext if ext in ('wav', 'mp3') else 'mp3'

    with open(read_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Clean up temp mp3 after reading
    if mp3_path and os.path.exists(mp3_path):
        try:
            os.remove(mp3_path)
        except Exception:
            pass

    prompt = f"""You are a senior {lang_name} language examiner for the RBI Language Proficiency Test.

Listen carefully to the audio. The candidate is reading this passage aloud:

REFERENCE PASSAGE ({ref_word_count} words):
{reference_text}

TASKS:
1. TRANSCRIBE exactly what you hear in {lang_name} script.
2. SCORE the reading 0-100 based on word coverage and fluency.

SCORING GUIDE:
- 95-100: All words read, excellent fluency
- 85-94 : Nearly complete, very minor issues
- 75-84 : Most words read, 1-2 skips
- 60-74 : Several skips or stumbles
- <60   : Large portions missed

Return ONLY this JSON (no markdown):
{{
  "transcription": "<what you heard in {lang_name} script>",
  "score": <integer 0-100>,
  "word_count_heard": <integer>,
  "feedback": "<one constructive sentence in English>",
  "missed_words": []
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": audio_fmt,
                    }
                },
            ],
        }],
        max_tokens=600,
        temperature=0,
    )

    raw = response.choices[0].message.content
    raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
    raw = re.sub(r'\n?```$', '', raw.strip())
    result = json.loads(raw)

    score       = float(result.get("score", 0))
    user_text   = result.get("transcription", "")
    words_heard = int(result.get("word_count_heard", len(user_text.split())))
    wpm         = round((words_heard / duration) * 60, 1) if duration > 0 else 0

    wpm_min = WPM_THRESHOLD.get(language_code, DEFAULT_WPM_THRESHOLD)
    status  = "Pass" if score >= 75 and wpm >= wpm_min else "Fail"

    return {
        "score": round(score, 2),
        "wpm": wpm,
        "status": status,
        "user_transcription": user_text,
        "feedback": result.get("feedback", ""),
        "metrics": {
            "missed_words_list": result.get("missed_words", []),
            "extra_words_list":  [],
        },
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def calculate_reading_score(reference_text: str, audio_path: str,
                             language_code: str) -> dict:
    try:
        if language_code in GPT_AUDIO_LANGUAGES:
            return evaluate_with_gpt_audio(audio_path, reference_text, language_code)
        else:
            user_text, duration = transcribe_whisper(audio_path, reference_text, language_code)
            wpm = round((len(user_text.split()) / duration) * 60, 1) if duration > 0 else 0
            return evaluate_with_diff(reference_text, user_text, language_code, wpm)
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# HANDWRITING ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def analyze_handwriting(image_bytes: bytes, language_code: str,
                        expected_topics_list=None) -> dict:
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        lang_name    = LANG_NAMES.get(language_code, language_code.upper())

        prompt = f"""You are an expert {lang_name} language examiner for the RBI LPT.
Analyze this handwritten answer sheet image.

1. Transcribe the text exactly as written.
2. Identify spelling and grammatical errors with corrections.
3. Rate handwriting legibility (0-10).
4. Give an overall score out of 10.
5. Write brief constructive feedback.

Return ONLY a valid JSON object with keys:
'transcription', 'errors' (list of strings), 'legibility_score', 'final_score', 'feedback'."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}