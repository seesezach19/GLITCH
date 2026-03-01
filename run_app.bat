@echo off
REM Install deps for the SAME Python that runs the app
echo Installing dependencies...
python -m pip install -r requirements.txt -q
echo.
echo Starting app...
python -m streamlit run app.py
pause
