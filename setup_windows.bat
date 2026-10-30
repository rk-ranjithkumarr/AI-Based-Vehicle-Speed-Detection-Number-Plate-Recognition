@echo off
echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Installation completed.
echo Run: python main.py --source 0
pause
