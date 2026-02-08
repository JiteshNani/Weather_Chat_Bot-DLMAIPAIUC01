# Global Weather Forecast Chatbot 

A browser-based chatbot that answers weather questions for locations worldwide.

**Stack**
- Backend: Flask
- NLP: NLTK (tokenization + stemming/lemmatization fallback + optional sentiment)
- Weather data: Open-Meteo (no API key) + Open-Meteo Geocoding
- Deploy: Render (Gunicorn)

## 1) Run locally (Windows PowerShell)

> Prerequisites: Python 3.12.x installed (recommended 3.12.6)

From the project folder:
```powershell
py -3.12 -m venv .venv
# If activation is blocked, see "PowerShell activation fix" below
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
python app.py
```

Open: http://127.0.0.1:5000

### PowerShell activation fix (one-time)
If you see "running scripts is disabled":
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Close & reopen PowerShell, then activate again.

## 2) Train the intent model (optional but recommended)
This project includes an **NLTK Naive Bayes** intent classifier trained from `data/training_intents.json`.

```powershell
python train_intent_model.py
```

This creates: `models/intent_model.pkl`

If you skip training, the app still works using rule-based fallback matching.

## 3) Deploy on Render (via GitHub)
1. Create a GitHub repo and push this folder.
2. In Render: **New → Web Service** → connect repo.
3. Build command:
```bash
pip install -r requirements.txt && python -m nltk.downloader vader_lexicon
```
4. Start command:
```bash
gunicorn app:app
```

Render will read `.python-version` and use Python 3.12.6.

## Notes
- For “Use my location”, the browser asks for permission and sends latitude/longitude to the server.
- Open-Meteo returns timezone-aware forecasts.

