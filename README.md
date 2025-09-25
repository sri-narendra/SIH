# Student Counselor Backend (Flask + Gemini + Render)

This is a Flask API backend that connects to Google Gemini to provide psychological first-aid style responses to student messages, using a local knowledge base JSON.

## Endpoints

- `POST /api/chat`
  - Body: `{ "message": "student text here" }`
  - Response: `{ "reply": "model output" }`

## Running locally

1. Create `.env` file:
