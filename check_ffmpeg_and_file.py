import os
import ffmpeg

filename = "2025-07-29 13-01-22"
input_video = f"{filename}.mkv"
output_audio = f"{filename}_audio.mp3"
# ffmpeg.exe 경로 (알맞게 수정)
ffmpeg_path = r"C:/workspace/audiotext/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe"

if not os.path.exists(input_video):
    print(f"입력 파일이 존재하지 않습니다: {input_video}")
    exit()

try:
    ffmpeg.input(input_video).output(
        output_audio, format='mp3', acodec='libmp3lame', ac=1
    ).run(cmd=ffmpeg_path)
    print("오디오 추출 성공!")
except Exception as e:
    print("ffmpeg 실행 오류:", e)