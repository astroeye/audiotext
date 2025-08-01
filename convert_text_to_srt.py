import re
import os
from tkinter import filedialog, messagebox
import tkinter as tk

def convert_text_to_srt():
    """기존 텍스트 파일을 SRT 자막 파일로 변환"""
    
    # GUI 설정
    root = tk.Tk()
    root.withdraw()
    
    # 텍스트 파일 선택
    print("변환할 텍스트 파일을 선택해주세요...")
    input_file = filedialog.askopenfilename(
        title="텍스트 파일 선택",
        filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
    )
    
    if not input_file:
        print("파일이 선택되지 않았습니다.")
        return
    
    # SRT 파일 저장 위치 선택
    output_file = filedialog.asksaveasfilename(
        title="SRT 자막 파일 저장 위치",
        defaultextension=".srt",
        filetypes=[("SRT 자막 파일", "*.srt"), ("모든 파일", "*.*")]
    )
    
    if not output_file:
        print("저장 위치가 선택되지 않았습니다.")
        return
    
    # 텍스트 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # SRT 파일 생성
    with open(output_file, 'w', encoding='utf-8') as f:
        subtitle_number = 1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 타임스탬프와 텍스트 분리 (00:00:00.000 형식)
            match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+(.*)', line)
            if match:
                start_time = match.group(1)
                text = match.group(2)
                
                # 다음 라인의 시작 시간을 종료 시간으로 사용
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    next_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})', next_line)
                    if next_match:
                        end_time = next_match.group(1)
                    else:
                        # 마지막 라인이거나 다음 타임스탬프가 없으면 3초 추가
                        end_time = add_seconds_to_timestamp(start_time, 3)
                else:
                    # 마지막 라인인 경우 3초 추가
                    end_time = add_seconds_to_timestamp(start_time, 3)
                
                # SRT 형식으로 변환 (콤마 사용)
                start_srt = start_time.replace('.', ',')
                end_srt = end_time.replace('.', ',')
                
                # SRT 형식으로 쓰기
                f.write(f"{subtitle_number}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
                
                subtitle_number += 1
    
    print(f"SRT 자막 파일이 생성되었습니다: {output_file}")
    messagebox.showinfo("완료", f"SRT 자막 파일이 생성되었습니다!\n\n저장 위치: {output_file}\n\nVLC 사용법:\n1. VLC에서 동영상 열기\n2. 자막 → 자막 파일 추가\n3. {os.path.basename(output_file)} 선택")
    root.destroy()

def add_seconds_to_timestamp(timestamp, seconds):
    """타임스탬프에 초를 추가하는 함수"""
    # 00:00:00.000 형식 파싱
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_ms = parts[2].split('.')
    secs = int(sec_ms[0])
    ms = int(sec_ms[1])
    
    # 총 밀리초로 변환
    total_ms = (hours * 3600 + minutes * 60 + secs) * 1000 + ms
    
    # 지정된 초를 추가
    total_ms += seconds * 1000
    
    # 다시 시:분:초.밀리초 형식으로 변환
    total_seconds = total_ms // 1000
    ms = total_ms % 1000
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"

if __name__ == "__main__":
    convert_text_to_srt()
