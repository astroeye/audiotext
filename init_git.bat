@echo off
echo ====================================
echo GitHub 저장소 초기화
echo ====================================

echo.
echo Git 저장소 초기화 중...
git init

echo.
echo 파일 스테이징 중...
git add .

echo.
echo 첫 번째 커밋 생성 중...
git commit -m "Initial commit: Audio/Video to Text Converter with AI Summary"

echo.
echo ====================================
echo 초기화 완료!
echo ====================================
echo.
echo 다음 단계:
echo 1. GitHub에서 새 저장소를 만드세요
echo 2. 다음 명령어를 실행하세요:
echo    git remote add origin https://github.com/사용자명/저장소명.git
echo    git branch -M main
echo    git push -u origin main
echo.
pause
