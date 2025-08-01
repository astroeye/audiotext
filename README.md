# 🎵 Audio-Text Converter with AI Summary

## 📖 프로젝트 소개

이 프로젝트는 오디오 및 동영상 파일을 텍스트로 변환하고, LLM을 활용해 요약까지 제공하는 GUI 애플리케이션입니다.
**MVC 패턴으로 구조화**되어 코드의 유지보수와 확장성을 높였습니다.

### ✨ 주요 기능

- 📹 **동영상/오디오 변환**: mkv, mp4, avi, mov, wmv, flv, webm, mp3, wav, flac, aac, ogg, m4a 등 다양한 형식 지원
- 🎤 **화자 분리**: 여러 사람이 대화하는 경우 화자를 구분하여 표시
- 🤖 **AI 요약**: OpenAI GPT, Anthropic Claude, Google Gemini를 활용한 텍스트 요약
- 📋 **회의록 형태**: 일반 요약과 구조화된 회의록 형태 요약 지원
- 🖱️ **드래그 앤 드롭**: 직관적인 파일 선택
- 📄 **자막 생성**: SRT 형식 자막 파일 생성

### 🏗️ 프로젝트 구조 (MVC 패턴)

```
audiotext/
├── main.py                          # 메인 실행 파일
├── models/                          # 데이터 모델 및 비즈니스 로직
│   ├── __init__.py
│   └── audio_processor.py           # 오디오 처리 클래스
├── services/                        # 외부 서비스 연동
│   ├── __init__.py
│   └── llm_service.py              # LLM API 서비스
├── views/                          # GUI 뷰 컴포넌트
│   ├── __init__.py
│   └── audio_converter_view.py     # 메인 GUI 클래스
├── controllers/                    # MVC 컨트롤러
│   ├── __init__.py
│   └── audio_converter_controller.py
├── ffmpeg-7.1.1-full_build/       # FFmpeg 바이너리
└── api_keys.json                   # API 키 설정 파일
```

### 🛠️ 기술 스택

- **GUI**: tkinter, tkinterdnd2
- **음성 인식**: OpenAI Whisper
- **오디오 처리**: librosa, soundfile, ffmpeg
- **AI 요약**: OpenAI GPT, Anthropic Claude, Google Gemini
- **화자 분리**: 오디오 특성 기반 분석
- **아키텍처**: MVC (Model-View-Controller) 패턴

## 🚀 설치 및 실행

### 1. 필수 요구사항
- Python 3.8+
- FFmpeg (프로젝트에 포함됨)

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. FFmpeg 설치
- Windows: [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
- 또는 프로젝트에 포함된 ffmpeg-7.1.1-full_build 사용

### 4. 실행
```bash
python main.py
```

## 🔧 사용법

### 1. 기본 변환
1. 오디오/동영상 파일을 드래그 앤 드롭하거나 "파일 선택" 버튼 클릭
2. 저장 위치 선택 (선택사항)
3. Whisper 모델 크기 선택 (tiny, base, small, medium, large)
4. 화자 분리 옵션 선택 (선택사항)
5. "변환 시작" 버튼 클릭

### 2. AI 요약
1. 텍스트 변환 완료 후 "텍스트 요약" 버튼 클릭
2. 또는 기존 .txt 파일을 직접 선택하여 요약
3. LLM 선택 (OpenAI, Anthropic, Gemini)
4. 요약 형태 선택 (일반 요약 / 회의록 형태)

### 3. API 설정
1. "API 설정" 버튼 클릭
2. 사용할 LLM의 API 키 입력
3. API 키 획득 방법:
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/
   - Google Gemini: https://aistudio.google.com/app/apikey

## 📊 Whisper 모델별 특성

| 모델 | 크기 | 속도 | 정확도 | 권장 용도 |
|------|------|------|--------|-----------|
| tiny | ~39MB | 매우 빠름 | 낮음 | 빠른 테스트 |
| base | ~74MB | 빠름 | 보통 | 일반적 사용 (권장) |
| small | ~244MB | 보통 | 높음 | 품질 중시 |
| medium | ~769MB | 느림 | 매우 높음 | 고품질 변환 |
| large | ~1550MB | 매우 느림 | 최고 | 최고 품질 |

## 💡 LLM API 특징

| API | 비용 | 특징 | 할당량 |
|-----|------|------|--------|
| Google Gemini | 무료 | 일 50회 무료 | 제한적 |
| OpenAI GPT | 유료 | 안정적, 빠름 | 사용량 기반 |
| Anthropic Claude | 유료 | 고품질, 긴 텍스트 | 사용량 기반 |

## 📁 출력 파일

- `파일명.txt`: 변환된 텍스트 (화자 분리 포함)
- `파일명.srt`: 자막 파일
- `파일명_speakers.txt`: 화자별 분리된 텍스트 (화자 분리 활성화 시)
- `파일명_summary.txt`: AI 일반 요약
- `파일명_meeting_summary.txt`: AI 회의록 형태 요약

## ⚠️ 주의사항

1. **GPU 지원**: CUDA 지원 GPU가 있으면 자동으로 GPU 가속 사용
2. **파일 크기**: 큰 파일의 경우 변환 시간이 오래 걸릴 수 있음
3. **API 비용**: OpenAI, Anthropic는 사용량에 따라 요금 부과
4. **개인정보**: API 키는 로컬에 저장되며 외부로 전송되지 않음

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 📄 라이선스

MIT License

## 🔗 관련 링크

- [OpenAI Whisper](https://github.com/openai/whisper)
- [FFmpeg](https://ffmpeg.org/)
- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://console.anthropic.com/)
- [Google Gemini API](https://aistudio.google.com/)
