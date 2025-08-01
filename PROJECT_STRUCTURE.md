# 프로젝트 구조

```
audiotext/
├── 📁 Core Files (핵심 파일)
│   ├── audio_text_converter_gui_LMM.py    # 메인 GUI 애플리케이션 (LLM 요약 포함)
│   ├── audio_text_converter_gui.py        # 기본 GUI 애플리케이션
│   ├── whisper_wav_to_text.py             # Whisper 기반 음성 인식
│   └── convert_mkv_to_text_gpu.py         # GPU 가속 변환
│
├── 📁 Utilities (유틸리티)
│   ├── check_ffmpeg_and_file.py           # FFmpeg 및 파일 검증
│   ├── convert_text_to_srt.py             # 텍스트를 SRT 자막으로 변환
│   └── test_*.py                          # 테스트 파일들
│
├── 📁 Configuration (설정)
│   ├── requirements.txt                   # Python 패키지 의존성
│   ├── api_keys_template.json             # API 키 템플릿
│   ├── .gitignore                         # Git 제외 파일 목록
│   └── 실행.bat                           # Windows 실행 스크립트
│
├── 📁 Installation (설치)
│   ├── install.bat                        # 자동 설치 스크립트
│   └── init_git.bat                       # Git 저장소 초기화
│
├── 📁 Documentation (문서)
│   ├── README.md                          # 프로젝트 설명서
│   ├── 사용법.md                          # 한국어 사용법
│   └── PROJECT_STRUCTURE.md               # 이 파일
│
└── 📁 Excluded (제외 파일들 - .gitignore에 의해)
    ├── *.mkv, *.mp4, *.avi, *.mov, ...   # 동영상 파일
    ├── *.mp3, *.wav, *.flac, *.m4a, ...  # 오디오 파일
    ├── *.txt, *.srt                      # 생성된 텍스트/자막 파일
    ├── api_keys.json                      # 실제 API 키 (보안)
    ├── ffmpeg-7.1.1-full_build/          # FFmpeg 바이너리
    └── .venv/                             # 가상환경
```

## 파일별 설명

### 🎯 메인 애플리케이션
- **audio_text_converter_gui_LMM.py**: LLM 요약 기능이 포함된 완전한 GUI 애플리케이션
- **audio_text_converter_gui.py**: 기본 변환 기능만 있는 간단한 버전

### 🔧 핵심 모듈
- **whisper_wav_to_text.py**: OpenAI Whisper를 사용한 음성 인식 엔진
- **convert_mkv_to_text_gpu.py**: GPU 가속을 활용한 고속 변환

### 🛠️ 유틸리티
- **check_ffmpeg_and_file.py**: 시스템 요구사항 및 파일 유효성 검사
- **convert_text_to_srt.py**: 변환된 텍스트를 SRT 자막 형식으로 변환

### 📋 설정 파일
- **requirements.txt**: 필요한 Python 패키지 목록
- **api_keys_template.json**: API 키 설정 템플릿
- **.gitignore**: Git에서 제외할 파일 패턴

### 🚀 실행 스크립트
- **실행.bat**: Windows에서 애플리케이션 빠른 실행
- **install.bat**: 의존성 자동 설치
- **init_git.bat**: Git 저장소 초기화

## 사용 순서

1. **설치**: `install.bat` 실행
2. **API 설정**: `api_keys_template.json`을 `api_keys.json`으로 복사 후 키 입력
3. **실행**: `python audio_text_converter_gui_LMM.py` 또는 `실행.bat`

## 개발 환경 설정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd audiotext

# 2. 가상환경 생성
python -m venv .venv
.venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. API 키 설정
copy api_keys_template.json api_keys.json
# api_keys.json 파일을 편집하여 실제 API 키 입력

# 5. 실행
python audio_text_converter_gui_LMM.py
```
