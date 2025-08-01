import whisper
import os

# ffmpeg 경로 설정 (여러 방법으로 시도)
ffmpeg_bin_path = r"C:\workspace\audiotext\ffmpeg-7.1.1-full_build\bin"
os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")

# whisper가 사용하는 ffmpeg 경로도 설정
try:
    import whisper.audio
    whisper.audio.FFMPEG_PATH = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
except:
    pass

# wav 파일 경로 지정 (절대 경로 사용)
audio_file = os.path.abspath('test_audio.wav')

# 파일 존재 확인
if not os.path.exists(audio_file):
    print(f"오디오 파일을 찾을 수 없습니다: {audio_file}")
    exit(1)

print(f"오디오 파일 경로: {audio_file}")
print(f"파일 크기: {os.path.getsize(audio_file)} 바이트")

# ffmpeg 경로 확인
ffmpeg_path = r"C:\workspace\audiotext\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
print(f"ffmpeg 경로: {ffmpeg_path}")
print(f"ffmpeg 존재: {os.path.exists(ffmpeg_path)}")

# Whisper 모델 로드 (base 모델, 자동으로 GPU 사용)
print("Whisper 모델을 로드하는 중...")
model = whisper.load_model("base")
print("모델 로드 완료!")

# 오디오 파일을 문자로 변환
print("오디오 파일을 텍스트로 변환하는 중...")
result = model.transcribe(audio_file)
print("변환 완료!")

# 결과 출력
print("결과:")
print(result["text"])