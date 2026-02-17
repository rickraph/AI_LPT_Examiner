import os
import string
import json
import base64
import difflib
import Levenshtein
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- HELPER FUNCTIONS ---

def normalize_text(text: str) -> str:
    """
    Cleans text to make comparison fair.
    1. Lowercases text.
    2. Removes standard punctuation and Hindi/Indic danda (|).
    3. Removes extra whitespace.
    """
    if not text:
        return ""
        
    text = text.lower()
    # Punctuation + Hindi 'Danda' (।) and double danda (॥)
    punctuation_to_remove = string.punctuation + "।॥"
    translator = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(translator)
    
    # Normalize Whitespace
    text = " ".join(text.split())
    return text

# --- CORE LOGIC: READING EVALUATOR ---

def calculate_reading_score(reference_text, audio_path, language_code):
    """
    Evaluates reading with:
    1. Word Coverage (Did they say the required words?)
    2. Phonetic Forgiveness (Forgives minor spelling diffs)
    3. Noise Penalty (Penalizes extra words)
    """
    
    # 1. Transcribe Audio using Whisper API
    try:
        audio_file = open(audio_path, "rb")
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language=language_code, 
            response_format="verbose_json"
        )
        audio_file.close()
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}
    
    user_text = transcript_response.text
    duration = transcript_response.duration
    
    # 2. Normalize and Tokenize
    ref_clean = normalize_text(reference_text)
    user_clean = normalize_text(user_text)
    
    ref_words = ref_clean.split()
    user_words = user_clean.split()
    
    # 3. Advanced Comparison Logic (Difflib)
    matcher = difflib.SequenceMatcher(None, ref_words, user_words)
    
    missing_words = []
    extra_words = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            # Missed words
            missing_words.extend(ref_words[i1:i2])
            
        elif tag == 'insert':
            # Extra words
            extra_words.extend(user_words[j1:j2])
            
        elif tag == 'replace':
            # Smart Check: Did they replace it with a similar sounding word?
            ref_segment_str = "".join(ref_words[i1:i2])
            user_segment_str = "".join(user_words[j1:j2])
            
            similarity = Levenshtein.ratio(ref_segment_str, user_segment_str)
            
            # Threshold: 0.70 allows for minor phonetic differences
            if similarity < 0.70:
                missing_words.extend(ref_words[i1:i2]) # Valid miss
                extra_words.extend(user_words[j1:j2])  # Valid noise

    # 4. Scoring Algorithm
    total_ref_words = len(ref_words)
    missed_count = len(missing_words)
    extra_count = len(extra_words)
    
    # Metric A: Coverage Score (0-100)
    if total_ref_words > 0:
        coverage_score = ((total_ref_words - missed_count) / total_ref_words) * 100
    else:
        coverage_score = 0
        
    # Metric B: Noise Penalty (3 points per extra word, max 30)
    noise_penalty = min(extra_count * 3, 30)
    
    # Metric C: Global Pronunciation
    pronunciation_score = Levenshtein.ratio(ref_clean, user_clean) * 100
    
    # Final Weighted Score
    # 60% Coverage, 40% Pronunciation, minus Noise Penalty
    raw_score = (coverage_score * 0.6) + (pronunciation_score * 0.4)
    final_score = max(raw_score - noise_penalty, 0)
    final_score = round(final_score, 2)
    
    # 5. Fluency (WPM)
    word_count = len(user_words)
    wpm = round((word_count / duration) * 60, 1) if duration > 0 else 0
    
    # 6. Verdict
    status = "Fail"
    if final_score >= 80 and wpm >= 50:
        status = "Pass"

    return {
        "score": final_score,
        "wpm": wpm,
        "status": status,
        "metrics": {
            "missed_word_count": missed_count,
            "missed_words_list": missing_words,
            "extra_word_count": extra_count,
            "extra_words_list": extra_words,
            "noise_penalty_applied": noise_penalty
        },
        "user_transcription": user_text,
        "reference_text": reference_text
    }

# --- CORE LOGIC: WRITING EVALUATOR ---

def analyze_handwriting(image_bytes, language_code, expected_topics_list=None):
    """
    Uses GPT-4o Vision to read and grade handwriting.
    Now includes:
    1. Robustness for messy handwriting.
    2. Topic Relevance Check (Did they write on the assigned topic?).
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    lang_map = {"hi": "Hindi", "ml": "Malayalam", "te": "Telugu"}
    full_lang = lang_map.get(language_code, "Indian Language")
    
    # Format topics for the prompt
    topics_str = ", ".join(expected_topics_list) if expected_topics_list else "General Topics"

    prompt = f"""
    You are a strict Language Proficiency Examiner for the RBI (Reserve Bank of India).
    I have provided an image of a candidate's handwriting in {full_lang.upper()}.
    
    The candidate was asked to write on ONE of these topics: [{topics_str}]
    
    Your Task:
    1. **Decipher the Text:** Try your absolute best to read the handwriting, even if it is messy or cursive. Only return "Unreadable" if it is truly impossible.
    2. **Topic Check:** Does the content align with one of the expected topics? 
       - If the content is completely unrelated (e.g., they wrote about a movie instead of "Digital India"), mark it as "Off-Topic" and Fail them.
    3. **Linguistic Analysis:** Check for spelling and grammar errors.
    4. **Grading:** Rate their writing skills (1-10).
    
    Output strictly in JSON format:
    {{
        "transcription": "The text exactly as written...",
        "detected_topic": "Which topic did they write on? (or 'Unknown/Off-topic')",
        "is_relevant": true,
        "spelling_errors": ["List specific errors"],
        "grammar_errors": ["List specific errors"],
        "legibility_score": 8, 
        "relevance_score": 10,
        "overall_status": "Pass" or "Fail",
        "feedback": "Specific feedback on their writing skills and relevance."
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Vision analysis failed: {str(e)}"}