import json
import os
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- Gemini SDK ---
from google import genai
from google.genai import types

app = Flask(__name__)
# Allow CORS for frontend (GitHub Pages + localhost for testing)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Load environment variables ---
load_dotenv(override=False)
if Path(".env").exists():
    load_dotenv(".env", override=False)

# Read Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY. Add it to Render secrets or .env.")

# Instantiate Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- Paths ---
KB_PATH = Path("knowledge_base.json")


def load_knowledge_base() -> List[Dict[str, Any]]:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {KB_PATH.resolve()}")

    with KB_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("knowledge_base.json must be a list of objects.")

    for i, item in enumerate(data):
        if not isinstance(item, dict) or "Context" not in item or "Response" not in item:
            raise ValueError(f"Invalid KB format at index {i}. Must have 'Context' and 'Response'.")
    return data


def build_system_instruction() -> str:
    return (
        "ROLE: You are a compassionate, non-judgmental, AI-guided psychological first-aid provider for students. "
        "You are NOT a medical professional—do not provide diagnoses, treatments, or medical advice.\n\n"
        "TASK: From the provided 'Knowledge Base JSON', find the single most contextually relevant 'Response' that best matches the student's message.\n\n"
        "IMPORTANT RULES:\n"
        "- You only help and guide for smaller-level emotional or academic stress issues.\n"
        "- If the student's message contains trigger words or phrases indicating suicidal thoughts, self-harm, or life-ending situations, DO NOT attempt to solve the problem. "
        "Instead, respond empathetically and clearly instruct the student to visit the 'Counsellor Booking' section on the website and book an appointment with an available counsellor immediately.\n\n"
        "OUTPUT:\n"
        "- If a close match is found: return ONLY the text of that 'Response'.\n"
        "- If no relevant match is found: return a brief, empathetic message that encourages the student to use the Confidential Booking System.\n\n"
        "CONSTRAINTS & TONE:\n"
        "- Use ONLY the provided knowledge base for specific advice; do not invent facts.\n"
        "- Keep language warm, validating, clear, and culturally sensitive.\n"
        "- If the student message suggests crisis or self-harm: avoid instructions; reply empathetically and encourage immediate contact with a counsellor via the Confidential Booking System or local emergency services.\n"
    )


def build_llm_payload(student_message: str, kb_data: List[Dict[str, Any]]) -> str:
    kb_as_json = json.dumps(kb_data, ensure_ascii=False)
    return (
        "Student Message:\n"
        f"{student_message.strip()}\n\n"
        "Knowledge Base JSON (array of {Context, Response}):\n"
        f"{kb_as_json}\n\n"
        "Return the final answer now, following the OUTPUT rules."
    )


def get_ai_response(student_message: str) -> str:
    kb_data = load_knowledge_base()
    system_instruction = build_system_instruction()
    contents = build_llm_payload(student_message, kb_data)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=500,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    return (response.text or "").strip()


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    student_message = (payload.get("message") or "").strip()
    if not student_message:
        return jsonify({"error": "Missing 'message' in request body."}), 400

    try:
        reply = get_ai_response(student_message)
        return jsonify({"reply": reply}), 200
    except FileNotFoundError as fe:
        return jsonify({"error": str(fe)}), 500
    except Exception as e:
        return jsonify({"error": "Server error.", "detail": str(e)}), 500


if __name__ == "__main__":
    # Render provides $PORT automatically
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
