# app.py
import os
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# --- Google Generative AI SDK (Gemini) ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Serve static files (if you have a static folder)
@app.route("/", methods=["GET"])
def home():
    # Try to serve from static folder first, fallback to current directory
    try:
        return send_from_directory('static', 'index.html')
    except:
        return send_from_directory('.', 'index.html')

# Serve other static files if needed
@app.route('/<path:path>')
def serve_static(path):
    try:
        return send_from_directory('static', path)
    except:
        return send_from_directory('.', path)

# -----------------------------------------------------------------------------
# Environment and Gemini client
# -----------------------------------------------------------------------------
load_dotenv(override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). "
        "Add it to .env, e.g., GEMINI_API_KEY=your_key_here."
    )

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Define Safety Settings (BLOCK_ONLY_HIGH for all categories)
# This allows the assistant to address sensitive topics required for student support,
# while only blocking content with the highest risk of harm.
safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
]

# Initialize model with updated name and safety settings
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    system_instruction=(
        "You are a compassionate student support assistant. "
        "Provide brief, empathetic responses for academic/emotional stress. "
        "For crisis situations (self-harm, unalive), immediately refer to counsellor booking. "
        "Keep responses under 150 words."
    ),
    safety_settings=safety_settings
)
# -----------------------------------------------------------------------------
# AI Response Generation
# -----------------------------------------------------------------------------

def get_ai_response(student_message: str) -> str:
    """
    Generates a response from the Gemini model using only the system prompt.
    """
    # Send only the student message (contents)
    contents = student_message.strip()

    response = model.generate_content(
        contents,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 300,
        }
    )
    
    # Check if the response was blocked due to safety and provide a fallback message
    if not response.candidates:
        return "I am unable to process that specific request due to safety guidelines. If you are experiencing a crisis, please seek immediate help or contact a professional counsellor using the booking link."

    return (response.text or "").strip()

# -----------------------------------------------------------------------------
# API endpoint
# -----------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        print("Incoming payload:", payload, flush=True)  # Debug print
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    student_message = (payload.get("message") or "").strip()
    if not student_message:
        return jsonify({"error": "Missing 'message' in request body."}), 400

    try:
        # Note: No FileNotFoundError possible anymore since knowledge_base.json is removed.
        reply = get_ai_response(student_message)
        return jsonify({"reply": reply}), 200

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug print
        return jsonify({"error": "Server error processing your message."}), 500

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if index.html exists
    if not Path("index.html").exists() and not Path("static/index.html").exists():
        print("WARNING: index.html not found. Please create it in the root or static folder.")
    
    print("Starting server on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)