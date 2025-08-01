@echo off
echo 오디오/동영상 텍스트 변환기를 시작합니다...
cd /d "%~dp0"
.venv\Scripts\python.exe audio_text_converter_gui.py
pause
