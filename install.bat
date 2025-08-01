@echo off
echo ====================================
echo 오디오/동영상 텍스트 변환기 설치
echo ====================================

echo.
echo 1. Python 패키지 설치 중...
pip install -r requirements.txt

echo.
echo 2. PyTorch GPU 지원 확인 중...
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"

echo.
echo 3. Whisper 모델 다운로드 중... (base 모델)
python -c "import whisper; whisper.load_model('base')"

echo.
echo ====================================
echo 설치 완료!
echo ====================================
echo.
echo 실행하려면 다음 명령어를 사용하세요:
echo python audio_text_converter_gui_LMM.py
echo.
pause
