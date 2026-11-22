@echo off
title Document Q&A Assistant - Setup and Run
echo ==========================================
echo Document Q&A Assistant - First-time setup
echo ==========================================
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 goto error
)
call .venv\Scripts\activate
if errorlevel 1 goto error
python -m pip install --upgrade pip
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
if not exist .env copy .env.example .env
echo.
echo Starting Streamlit...
streamlit run app.py
goto end
:error
echo.
echo Setup failed. Copy the full error shown above and send it for debugging.
pause
:end
