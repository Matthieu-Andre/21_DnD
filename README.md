# 21_DnD

AI based DnD

## Tasks

- [ ] Voxtral API  - Sloan
- [ ] Game state - Matthieu
- [ ] Mistral large 3 as OS  - Matthieu
- [ ] youtube api  - Sloan

## Speech To Text

Record and transcribe:

```bash
python stt.py --duration 10
```

Live transcription with conversation state and mood tracking from the machine microphone:

```bash
python stt.py --live
```

This writes the current session state to `conversation_state.json`, including:

- finalized utterances
- partial live text
- current mood
- mood history

Record, transcribe, and analyze the transcript:

```bash
python stt.py --duration 10 --analyze
```

Analyze an existing output file:

```bash
python analysis_agent.py --input transcription.txt --output analysis.txt
```

## Browser Frontend

The web app is now built for local development and cloud deployment:

- the browser captures microphone audio
- the frontend slices mono WAV chunks locally
- FastAPI receives chunks and calls Mistral for transcription
- mood tracking updates `conversation_state.json`

### Local Dev

Install the web/backend dependencies:

```bash
pip install -r requirements-web.txt
```

Run the local web app:

```bash
python frontend_server.py --reload
```

Then open `http://127.0.0.1:8080`.

### Railway Backend

Railway can now detect the app as a standard Python service because the repo
includes a top-level `requirements.txt` that points at `requirements-web.txt`.

The backend entrypoint is:

```bash
uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

A matching `Procfile` is included, so you do not need a custom start script.

Required environment variables:

```bash
MISTRAL_API_KEY=your_key_here
```

Optional:

```bash
CORS_ALLOW_ORIGINS=https://your-vercel-app.vercel.app
```

### Vercel Frontend

You can deploy the static frontend in `web/` on Vercel and point it at Railway.

The UI has an `API base URL` field:

- leave it blank for local same-origin dev
- set it to your Railway backend origin when the frontend is hosted on Vercel
