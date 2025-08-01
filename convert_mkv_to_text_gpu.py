import ffmpeg
import whisper
import torch
import os
import soundfile as sf
from tkinter import filedialog, messagebox
import tkinter as tk

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

# GUI를 숨기고 파일 선택 대화상자 표시
root = tk.Tk()
root.withdraw()  # GUI 창 숨기기

# 1. 입력 파일 선택 (동영상 또는 오디오 파일)
print("입력 파일을 선택해주세요...")
input_file = filedialog.askopenfilename(
    title="변환할 파일을 선택하세요",
    filetypes=[
        ("모든 지원 파일", "*.mkv;*.mp4;*.avi;*.mov;*.wmv;*.flv;*.webm;*.mp3;*.wav;*.flac;*.aac;*.ogg"),
        ("동영상 파일", "*.mkv;*.mp4;*.avi;*.mov;*.wmv;*.flv;*.webm"),
        ("오디오 파일", "*.mp3;*.wav;*.flac;*.aac;*.ogg"),
        ("모든 파일", "*.*")
    ]
)

if not input_file:
    print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
    exit(0)

# 2. 출력 텍스트 파일 저장 위치 선택
print("텍스트 파일을 저장할 위치를 선택해주세요...")
output_text = filedialog.asksaveasfilename(
    title="텍스트 파일 저장 위치",
    defaultextension=".txt",
    filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
)

if not output_text:
    print("저장 위치가 선택되지 않았습니다. 프로그램을 종료합니다.")
    exit(0)

# 자막 파일 경로 설정 (같은 경로에 .srt 확장자로)
output_subtitle = os.path.splitext(output_text)[0] + ".srt"

# 3. 파일 정보 확인
print(f"\n선택된 입력 파일: {input_file}")
print(f"파일 크기: {os.path.getsize(input_file):,} 바이트")
print(f"텍스트 저장 위치: {output_text}")
print(f"자막 파일 저장 위치: {output_subtitle}")

# 4. 임시 오디오 파일 경로 설정 (WAV 형식)
import tempfile
temp_dir = tempfile.mkdtemp()
temp_audio_file = os.path.join(temp_dir, "temp_audio.wav")

# 5. 입력 파일 확장자 확인
file_ext = os.path.splitext(input_file)[1].lower()
audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']

if file_ext in audio_extensions:
    # 이미 오디오 파일인 경우
    print("입력 파일이 오디오 파일입니다. 직접 변환을 진행합니다.")
    audio_file_for_whisper = input_file
else:
    # 동영상 파일인 경우 오디오 추출
    print("동영상에서 오디오를 추출합니다...")
    audio_file_for_whisper = temp_audio_file

# ffmpeg 경로 확인
ffmpeg_path = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
print(f"ffmpeg 경로: {ffmpeg_path}")
print(f"ffmpeg 존재: {os.path.exists(ffmpeg_path)}")

# 6. 오디오 추출 (동영상 파일인 경우만)
if file_ext not in audio_extensions:
    print("동영상에서 오디오 추출을 시작합니다...")
    try:
        ffmpeg.input(input_file).output(
            temp_audio_file, format='wav', acodec='pcm_s16le', ac=1, ar=16000
        ).run(
            cmd=ffmpeg_path,
            capture_stdout=True,
            capture_stderr=True,
            overwrite_output=True
        )
    except ffmpeg.Error as e:
        print('ffmpeg error:', e.stderr.decode())
        raise
    
    print("오디오 추출이 완료되었습니다.")
    print(f"추출된 오디오 파일: {temp_audio_file}")
    print(f"파일 존재 여부: {os.path.exists(temp_audio_file)}")

# 7. 오디오 파일 검증 (soundfile로 읽어보기)
try:
    data, samplerate = sf.read(audio_file_for_whisper)
    print(f"오디오 파일 정보: 샘플레이트 {samplerate}Hz, 길이 {len(data)/samplerate:.2f}초")
except Exception as e:
    print(f"오디오 파일 읽기 오류: {e}")
    exit(1)

# 2. GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 3. Whisper로 음성 → 텍스트 변환 (segment 정보 포함)
print("Whisper 모델을 로드하는 중...")
model = whisper.load_model("base", device=device)
print("모델 로드 완료!")

if not os.path.exists(audio_file_for_whisper):
    raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_file_for_whisper}")

print("오디오 파일을 텍스트로 변환하는 중...")
result = model.transcribe(audio_file_for_whisper, task="transcribe")
print("변환 완료!")

# 4. 각 문장의 시작 시간(ms 포함)과 텍스트를 파일에 저장
def format_timestamp_ms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def format_srt_timestamp(seconds):
    """SRT 자막 형식의 타임스탬프 생성 (HH:MM:SS,mmm)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# 텍스트 파일 저장 (기존 형식)
with open(output_text, 'w', encoding='utf-8') as f:
    for segment in result['segments']:
        timestamp = format_timestamp_ms(segment['start'])
        f.write(f"{timestamp} {segment['text'].strip()}\n")

print(f"문장별 시작 시간이 ms 단위로 포함된 텍스트가 {output_text}에 저장되었습니다.")

# SRT 자막 파일 저장
with open(output_subtitle, 'w', encoding='utf-8') as f:
    for i, segment in enumerate(result['segments'], 1):
        start_time = format_srt_timestamp(segment['start'])
        end_time = format_srt_timestamp(segment['end'])
        text = segment['text'].strip()
        
        f.write(f"{i}\n")
        f.write(f"{start_time} --> {end_time}\n")
        f.write(f"{text}\n\n")

print(f"VLC용 SRT 자막 파일이 {output_subtitle}에 저장되었습니다.")

# 5. 임시 파일 정리
if file_ext not in audio_extensions and os.path.exists(temp_audio_file):
    try:
        os.remove(temp_audio_file)
        os.rmdir(temp_dir)
        print("임시 파일이 정리되었습니다.")
    except:
        print(f"임시 파일 정리 중 오류가 발생했습니다: {temp_audio_file}")

print("\n변환이 완료되었습니다!")
messagebox.showinfo("완료", f"텍스트 변환이 완료되었습니다!\n\n텍스트 파일: {output_text}\nVLC 자막 파일: {output_subtitle}\n\n자막 사용법:\n1. VLC에서 동영상 열기\n2. 자막 → 자막 파일 추가\n3. {os.path.basename(output_subtitle)} 선택")
root.destroy()