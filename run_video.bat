@echo off
call .venv\Scripts\activate
python main.py --source data/input/traffic.mp4 --ppm 8 --speed-limit 100
pause
