import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import os
import tempfile
import ffmpeg
import whisper
import torch
import soundfile as sf
import json
import requests
from typing import Optional

class AudioTextConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("오디오/동영상 → 텍스트 변환기 + AI 요약")
        self.root.geometry("600x800")
        self.root.resizable(True, True)
        
        # ffmpeg 경로 설정
        self.setup_ffmpeg()
        
        # 변수 초기화
        self.input_file = ""
        self.output_dir = ""
        self.is_converting = False
        self.stop_simulation = False
        self.audio_duration = 0
        self.last_output_text = ""  # 마지막으로 생성된 텍스트 파일 경로
        self.api_keys = {}  # API 키 저장
        
        # API 키 설정 파일 로드
        self.load_api_keys()
        
        # GUI 생성
        self.create_widgets()
        
        # 드래그 앤 드롭 설정
        self.setup_drag_drop()
        
    def setup_ffmpeg(self):
        """ffmpeg 경로 설정"""
        ffmpeg_bin_path = r"C:\workspace\audiotext\ffmpeg-7.1.1-full_build\bin"
        os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
        
        try:
            import whisper.audio
            whisper.audio.FFMPEG_PATH = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
        except:
            pass
    
    def load_api_keys(self):
        """API 키 설정 파일 로드"""
        config_file = os.path.join(os.path.dirname(__file__), "api_keys.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.api_keys = json.load(f)
        except Exception as e:
            self.api_keys = {}
    
    def save_api_keys(self):
        """API 키 설정 파일 저장"""
        config_file = os.path.join(os.path.dirname(__file__), "api_keys.json")
        try:
            # 빈 키는 제거하고 저장
            filtered_keys = {k: v for k, v in self.api_keys.items() if v and v.strip()}
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_keys, f, indent=2)
            
            print(f"API 키 저장됨: {config_file}")  # 디버깅용
            print(f"저장된 키: {list(filtered_keys.keys())}")  # 디버깅용
            
        except Exception as e:
            error_msg = f"API 키 저장 중 오류가 발생했습니다: {str(e)}\n파일 경로: {config_file}"
            print(f"저장 오류: {error_msg}")  # 디버깅용
            messagebox.showerror("오류", error_msg)
    
    def summarize_with_openai(self, text: str, api_key: str) -> Optional[str]:
        """OpenAI GPT를 사용한 텍스트 요약"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 텍스트 요약 전문가입니다. 주어진 텍스트를 명확하고 간결하게 요약해주세요. 한국어로 답변해주세요."},
                    {"role": "user", "content": f"다음 텍스트를 요약해주세요:\n\n{text}"}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API 오류: {str(e)}")
    
    def summarize_with_anthropic(self, text: str, api_key: str) -> Optional[str]:
        """Anthropic Claude를 사용한 텍스트 요약"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": f"다음 텍스트를 한국어로 명확하고 간결하게 요약해주세요:\n\n{text}"}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API 오류: {str(e)}")
    
    def summarize_with_gemini(self, text: str, api_key: str) -> Optional[str]:
        """Google Gemini를 사용한 텍스트 요약"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"다음 텍스트를 한국어로 명확하고 간결하게 요약해주세요:\n\n{text}"
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API 오류: {str(e)}")
    
    def show_api_settings(self):
        """API 설정 대화상자"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("LLM API 설정")
        settings_window.geometry("500x400")
        settings_window.resizable(False, False)
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # 창을 부모 창 중앙에 배치
        settings_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        main_frame = ttk.Frame(settings_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        title_label = ttk.Label(main_frame, text="LLM API 설정", font=("맑은 고딕", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 설명
        desc_label = ttk.Label(main_frame, 
                              text="텍스트 요약을 위해 사용할 LLM API의 키를 설정하세요.\n적어도 하나의 API 키는 설정해야 합니다.",
                              justify=tk.CENTER)
        desc_label.pack(pady=(0, 20))
        
        # API 키 입력 필드들
        api_frame = ttk.Frame(main_frame)
        api_frame.pack(fill=tk.X, pady=(0, 20))
        
        # OpenAI
        ttk.Label(api_frame, text="OpenAI API Key:", font=("맑은 고딕", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        openai_entry = ttk.Entry(api_frame, width=50, show="*")
        openai_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        openai_entry.insert(0, self.api_keys.get("openai", ""))
        
        # Anthropic
        ttk.Label(api_frame, text="Anthropic API Key:", font=("맑은 고딕", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        anthropic_entry = ttk.Entry(api_frame, width=50, show="*")
        anthropic_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        anthropic_entry.insert(0, self.api_keys.get("anthropic", ""))
        
        # Google Gemini
        ttk.Label(api_frame, text="Google Gemini API Key:", font=("맑은 고딕", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        gemini_entry = ttk.Entry(api_frame, width=50, show="*")
        gemini_entry.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        gemini_entry.insert(0, self.api_keys.get("gemini", ""))
        
        api_frame.columnconfigure(0, weight=1)
        
        # 안내 텍스트
        help_text = tk.Text(main_frame, height=6, wrap=tk.WORD, font=("맑은 고딕", 9))
        help_text.pack(fill=tk.X, pady=(0, 20))
        help_text.insert(tk.END, 
            "API 키 획득 방법:\n"
            "• OpenAI: https://platform.openai.com/api-keys\n"
            "• Anthropic: https://console.anthropic.com/\n"
            "• Google Gemini: https://aistudio.google.com/app/apikey\n\n"
            "주의: API 키는 로컬에 저장되며, 사용량에 따라 요금이 부과될 수 있습니다.")
        help_text.config(state=tk.DISABLED)
        
        # 버튼
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def save_and_close():
            self.api_keys["openai"] = openai_entry.get().strip()
            self.api_keys["anthropic"] = anthropic_entry.get().strip()
            self.api_keys["gemini"] = gemini_entry.get().strip()
            self.save_api_keys()
            self.update_api_status()  # API 상태 업데이트
            settings_window.destroy()
            messagebox.showinfo("저장 완료", "API 키가 저장되었습니다.")
        
        ttk.Button(button_frame, text="저장", command=save_and_close).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="취소", command=settings_window.destroy).pack(side=tk.RIGHT)
    
    def summarize_text(self):
        """텍스트 요약 실행"""
        if not self.last_output_text or not os.path.exists(self.last_output_text):
            messagebox.showerror("오류", "요약할 텍스트 파일이 없습니다.\n먼저 오디오/동영상을 텍스트로 변환해주세요.")
            return
        
        # API 키 확인
        available_apis = []
        if self.api_keys.get("openai"):
            available_apis.append(("OpenAI GPT", "openai"))
        if self.api_keys.get("anthropic"):
            available_apis.append(("Anthropic Claude", "anthropic"))
        if self.api_keys.get("gemini"):
            available_apis.append(("Google Gemini", "gemini"))
        
        if not available_apis:
            result = messagebox.askyesno("API 키 없음", 
                "설정된 API 키가 없습니다.\nAPI 키를 설정하시겠습니까?")
            if result:
                self.show_api_settings()
            return
        
        # API 선택 대화상자
        api_window = tk.Toplevel(self.root)
        api_window.title("LLM 선택")
        api_window.geometry("400x300")
        api_window.resizable(False, False)
        api_window.transient(self.root)
        api_window.grab_set()
        
        # 창을 부모 창 중앙에 배치
        api_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 100,
            self.root.winfo_rooty() + 100
        ))
        
        main_frame = ttk.Frame(api_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="요약에 사용할 LLM을 선택하세요:", 
                 font=("맑은 고딕", 12, "bold")).pack(pady=(0, 20))
        
        selected_api = tk.StringVar(value=available_apis[0][1])
        
        for name, value in available_apis:
            ttk.Radiobutton(main_frame, text=name, variable=selected_api, 
                           value=value).pack(anchor=tk.W, pady=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_summary():
            api_window.destroy()
            # 별도 스레드에서 요약 실행
            thread = threading.Thread(target=self.run_summarization, args=(selected_api.get(),))
            thread.daemon = True
            thread.start()
        
        ttk.Button(button_frame, text="요약 시작", command=start_summary).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="취소", command=api_window.destroy).pack(side=tk.RIGHT)
    
    def run_summarization(self, api_type: str):
        """요약 실행 (별도 스레드)"""
        try:
            self.log_message("=" * 50)
            self.log_message("텍스트 요약을 시작합니다...")
            
            # 텍스트 파일 읽기
            with open(self.last_output_text, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if len(text_content.strip()) == 0:
                self.root.after(0, lambda: messagebox.showerror("오류", "텍스트 파일이 비어있습니다."))
                return
            
            self.log_message(f"텍스트 길이: {len(text_content)} 문자")
            self.log_message(f"사용 LLM: {api_type}")
            
            # API 호출
            api_key = self.api_keys.get(api_type)
            if not api_key:
                self.root.after(0, lambda: messagebox.showerror("오류", f"{api_type} API 키가 설정되지 않았습니다."))
                return
            
            self.log_message("LLM 요약 중...")
            
            if api_type == "openai":
                summary = self.summarize_with_openai(text_content, api_key)
            elif api_type == "anthropic":
                summary = self.summarize_with_anthropic(text_content, api_key)
            elif api_type == "gemini":
                summary = self.summarize_with_gemini(text_content, api_key)
            else:
                raise Exception("지원하지 않는 API 타입입니다.")
            
            if not summary:
                raise Exception("요약 결과가 비어있습니다.")
            
            # 요약 파일 저장
            base_name = os.path.splitext(self.last_output_text)[0]
            summary_file = f"{base_name}_summary.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 텍스트 요약 ===\n")
                f.write(f"원본 파일: {os.path.basename(self.last_output_text)}\n")
                f.write(f"요약 모델: {api_type}\n")
                f.write(f"생성 시간: {self.get_current_time()}\n\n")
                f.write("=== 요약 내용 ===\n")
                f.write(summary)
            
            self.log_message("요약 완료!")
            self.log_message(f"요약 파일: {summary_file}")
            self.log_message("=" * 50)
            
            # 완료 메시지와 요약 내용 표시
            self.root.after(0, lambda: self.show_summary_result(summary, summary_file))
            
        except Exception as e:
            error_msg = str(e)
            self.log_message(f"요약 오류: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("요약 오류", f"요약 중 오류가 발생했습니다:\n\n{error_msg}"))
    
    def show_summary_result(self, summary: str, summary_file: str):
        """요약 결과 표시 대화상자"""
        result_window = tk.Toplevel(self.root)
        result_window.title("요약 결과")
        result_window.geometry("700x500")
        result_window.transient(self.root)
        
        # 창을 부모 창 중앙에 배치
        result_window.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        main_frame = ttk.Frame(result_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        ttk.Label(main_frame, text="텍스트 요약 결과", 
                 font=("맑은 고딕", 14, "bold")).pack(pady=(0, 10))
        
        # 파일 정보
        ttk.Label(main_frame, text=f"저장 위치: {summary_file}", 
                 foreground="gray").pack(pady=(0, 20))
        
        # 요약 내용 표시
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("맑은 고딕", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, summary)
        text_widget.config(state=tk.DISABLED)
        
        # 버튼
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def copy_to_clipboard():
            result_window.clipboard_clear()
            result_window.clipboard_append(summary)
            messagebox.showinfo("복사 완료", "요약 내용이 클립보드에 복사되었습니다.")
        
        def open_file_location():
            os.startfile(os.path.dirname(summary_file))
        
        ttk.Button(button_frame, text="파일 위치 열기", command=open_file_location).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="클립보드 복사", command=copy_to_clipboard).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="닫기", command=result_window.destroy).pack(side=tk.RIGHT)
    
    def get_current_time(self):
        """현재 시간을 문자열로 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def create_widgets(self):
        """GUI 위젯 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="오디오/동영상 → 텍스트 변환기", 
                               font=("맑은 고딕", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 파일 선택 영역
        file_frame = ttk.LabelFrame(main_frame, text="1. 파일 선택", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 드래그 앤 드롭 영역
        self.drop_label = ttk.Label(file_frame, 
                                   text="파일을 여기로 드래그하거나\n아래 버튼을 클릭하세요\n\n지원 형식:\n동영상: mkv, mp4, avi, mov, wmv, flv, webm\n오디오: mp3, wav, flac, aac, ogg",
                                   justify=tk.CENTER,
                                   relief="groove",
                                   padding="20")
        self.drop_label.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 파일 선택 버튼
        self.select_button = ttk.Button(file_frame, text="파일 선택", command=self.select_file)
        self.select_button.grid(row=1, column=0, pady=(0, 10))
        
        # 선택된 파일 표시
        self.file_label = ttk.Label(file_frame, text="선택된 파일: 없음", foreground="gray")
        self.file_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 출력 폴더 선택 영역
        output_frame = ttk.LabelFrame(main_frame, text="2. 저장 위치 선택", padding="10")
        output_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.output_button = ttk.Button(output_frame, text="저장 폴더 선택", command=self.select_output_dir)
        self.output_button.grid(row=0, column=0, pady=(0, 10))
        
        self.output_label = ttk.Label(output_frame, text="저장 위치: 기본값 (입력 파일과 같은 폴더)", foreground="gray")
        self.output_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 옵션 영역
        options_frame = ttk.LabelFrame(main_frame, text="3. 변환 옵션", padding="10")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # GPU 사용 여부 표시
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        device_label = ttk.Label(options_frame, text=f"사용 장치: {device}")
        device_label.grid(row=0, column=0, sticky=tk.W)
        
        # Whisper 모델 선택
        ttk.Label(options_frame, text="모델 크기:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large"],
                                  state="readonly", width=15)
        model_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0), padx=(10, 0))
        
        # 모델 설명
        model_info = ttk.Label(options_frame, 
                              text="tiny: 가장 빠름, 정확도 낮음\nbase: 균형잡힌 선택 (권장)\nsmall: 더 정확, 조금 느림\nmedium/large: 가장 정확, 매우 느림",
                              font=("맑은 고딕", 8), foreground="gray")
        model_info.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # 변환 버튼
        self.convert_button = ttk.Button(main_frame, text="변환 시작", 
                                        command=self.start_conversion, 
                                        style="Accent.TButton")
        self.convert_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        # 요약 기능 영역
        summary_frame = ttk.LabelFrame(main_frame, text="4. LLM 텍스트 요약", padding="10")
        summary_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        summary_desc = ttk.Label(summary_frame, 
                                text="변환된 텍스트를 AI로 요약할 수 있습니다.",
                                font=("맑은 고딕", 9), foreground="gray")
        summary_desc.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        button_frame = ttk.Frame(summary_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.summary_button = ttk.Button(button_frame, text="텍스트 요약", 
                                        command=self.summarize_text)
        self.summary_button.grid(row=0, column=0, sticky=tk.W)
        
        self.api_settings_button = ttk.Button(button_frame, text="API 설정", 
                                             command=self.show_api_settings)
        self.api_settings_button.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # API 상태 표시
        self.api_status_label = ttk.Label(summary_frame, text="API 키: 미설정", 
                                         foreground="orange", font=("맑은 고딕", 8))
        self.api_status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # API 상태 업데이트
        self.update_api_status()
        
        # 진행률 표시
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # 진행률 퍼센트 표시
        self.progress_label = ttk.Label(progress_frame, text="0%", font=("맑은 고딕", 9))
        self.progress_label.grid(row=0, column=1, sticky=tk.E)
        
        self.status_label = ttk.Label(progress_frame, text="대기 중...")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # 로그 영역
        log_frame = ttk.LabelFrame(main_frame, text="변환 로그", padding="10")
        log_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 스크롤바가 있는 텍스트 영역
        text_frame = ttk.Frame(log_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(text_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        file_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(1, weight=0)  # 퍼센트 표시는 고정 크기
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def setup_drag_drop(self):
        """드래그 앤 드롭 설정"""
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
    
    def update_api_status(self):
        """API 상태 표시 업데이트"""
        configured_apis = []
        if self.api_keys.get("openai"):
            configured_apis.append("OpenAI")
        if self.api_keys.get("anthropic"):
            configured_apis.append("Anthropic")
        if self.api_keys.get("gemini"):
            configured_apis.append("Gemini")
        
        if configured_apis:
            status_text = f"API 키: {', '.join(configured_apis)} 설정됨"
            color = "green"
        else:
            status_text = "API 키: 미설정"
            color = "orange"
        
        self.api_status_label.config(text=status_text, foreground=color)
    
    def on_drop(self, event):
        """파일 드롭 이벤트 처리"""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.input_file = files[0]
            self.update_file_display()
    
    def select_file(self):
        """파일 선택 대화상자"""
        filetypes = [
            ("모든 지원 파일", "*.mkv;*.mp4;*.avi;*.mov;*.wmv;*.flv;*.webm;*.mp3;*.wav;*.flac;*.aac;*.ogg"),
            ("동영상 파일", "*.mkv;*.mp4;*.avi;*.mov;*.wmv;*.flv;*.webm"),
            ("오디오 파일", "*.mp3;*.wav;*.flac;*.aac;*.ogg"),
            ("모든 파일", "*.*")
        ]
        
        file = filedialog.askopenfilename(title="변환할 파일을 선택하세요", filetypes=filetypes)
        if file:
            self.input_file = file
            self.update_file_display()
    
    def update_file_display(self):
        """선택된 파일 정보 업데이트"""
        if self.input_file:
            filename = os.path.basename(self.input_file)
            file_size = os.path.getsize(self.input_file)
            size_mb = file_size / (1024 * 1024)
            self.file_label.config(text=f"선택된 파일: {filename} ({size_mb:.1f} MB)", foreground="black")
            self.drop_label.config(text=f"✓ {filename}\n({size_mb:.1f} MB)")
    
    def select_output_dir(self):
        """출력 폴더 선택"""
        folder = filedialog.askdirectory(title="저장 폴더를 선택하세요")
        if folder:
            self.output_dir = folder
            self.output_label.config(text=f"저장 위치: {folder}", foreground="black")
    
    def log_message(self, message):
        """로그 메시지 추가"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, status, progress=None):
        """상태 업데이트"""
        self.status_label.config(text=status)
        if progress is not None:
            self.progress['value'] = progress
            self.progress_label.config(text=f"{progress:.0f}%")
        self.root.update_idletasks()
    
    def update_progress(self, value):
        """진행률 업데이트"""
        self.progress['value'] = value
        self.progress_label.config(text=f"{value:.0f}%")
        self.root.update_idletasks()
    
    def simulate_conversion_progress(self, audio_duration, model_size):
        """텍스트 변환 중 진행률 시뮬레이션 - 파일 크기와 모델에 따라 조정"""
        import time
        self.stop_simulation = False
        
        # 모델별 변환 속도 계수 (초당 처리 시간)
        model_speed = {
            "tiny": 0.1,    # 가장 빠름
            "base": 0.2,    # 기본
            "small": 0.4,   # 조금 느림
            "medium": 0.8,  # 느림
            "large": 1.2    # 가장 느림
        }
        
        # 예상 변환 시간 계산 (초)
        speed_factor = model_speed.get(model_size, 0.2)
        estimated_time = audio_duration * speed_factor
        
        # 최소 5초, 최대 300초로 제한
        estimated_time = max(5, min(estimated_time, 300))
        
        # 진행률 업데이트 간격 계산
        total_steps = int(estimated_time / 0.5)  # 0.5초마다 업데이트
        progress_per_step = 30 / total_steps  # 50%에서 80%까지 30% 증가
        
        current_progress = 50
        step_count = 0
        
        self.log_message(f"예상 변환 시간: {estimated_time:.1f}초 (모델: {model_size}, 오디오: {audio_duration:.1f}초)")
        
        while current_progress < 79 and not self.stop_simulation and step_count < total_steps:
            time.sleep(0.5)
            if not self.stop_simulation:
                step_count += 1
                # 비선형 진행률 증가 (초반에는 빠르게, 후반에는 느리게)
                progress_increment = progress_per_step * (1.2 - (current_progress - 50) / 30 * 0.8)
                current_progress += progress_increment
                current_progress = min(current_progress, 79)  # 79% 이상 안 가게
                
                self.root.after(0, lambda p=current_progress: self.update_progress(p))
                
                # 진행 상황 로그 (10% 단위로)
                if step_count % max(1, total_steps // 3) == 0:
                    remaining_time = estimated_time - (step_count * 0.5)
                    self.root.after(0, lambda: self.log_message(f"변환 진행 중... 예상 남은 시간: {max(0, remaining_time):.0f}초"))
    
    def start_conversion(self):
        """변환 시작"""
        if not self.input_file:
            messagebox.showerror("오류", "변환할 파일을 선택해주세요.")
            return
        
        if not os.path.exists(self.input_file):
            messagebox.showerror("오류", "선택한 파일이 존재하지 않습니다.")
            return
        
        if self.is_converting:
            messagebox.showwarning("경고", "이미 변환이 진행 중입니다.")
            return
        
        # 출력 폴더 설정
        if not self.output_dir:
            self.output_dir = os.path.dirname(self.input_file)
        
        # 변환 스레드 시작
        self.is_converting = True
        self.stop_simulation = False
        self.convert_button.config(state="disabled")
        self.progress['value'] = 0
        self.progress_label.config(text="0%")
        
        thread = threading.Thread(target=self.convert_file)
        thread.daemon = True
        thread.start()
    
    def convert_file(self):
        """파일 변환 (별도 스레드에서 실행)"""
        try:
            self.log_text.delete(1.0, tk.END)
            
            # 단계별 진행률 정의
            # 1. 파일 준비: 5%
            # 2. 오디오 추출: 20% (동영상인 경우) 또는 건너뛰기
            # 3. 오디오 검증: 25%
            # 4. 모델 로드: 40%
            # 5. 텍스트 변환: 80%
            # 6. 파일 저장: 95%
            # 7. 완료: 100%
            
            # 1. 파일 준비 (5%)
            self.update_status("파일 준비 중...", 5)
            
            # 출력 파일 경로 설정
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            output_text = os.path.join(self.output_dir, f"{base_name}.txt")
            output_subtitle = os.path.join(self.output_dir, f"{base_name}.srt")
            
            self.log_message(f"입력 파일: {self.input_file}")
            self.log_message(f"출력 폴더: {self.output_dir}")
            self.log_message("")
            
            # 파일 형식 확인
            file_ext = os.path.splitext(self.input_file)[1].lower()
            audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
            
            if file_ext in audio_extensions:
                self.log_message("오디오 파일을 직접 처리합니다.")
                audio_file = self.input_file
                # 오디오 파일인 경우 추출 단계를 건너뛰고 25%로 점프
                self.update_status("오디오 파일 확인 완료", 25)
            else:
                # 2. 동영상에서 오디오 추출 (5% -> 20%)
                self.update_status("동영상에서 오디오 추출 중...", 10)
                self.log_message("동영상에서 오디오를 추출합니다...")
                
                temp_dir = tempfile.mkdtemp()
                audio_file = os.path.join(temp_dir, "temp_audio.wav")
                
                ffmpeg_path = r"C:\workspace\audiotext\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
                
                try:
                    ffmpeg.input(self.input_file).output(
                        audio_file, format='wav', acodec='pcm_s16le', ac=1, ar=16000
                    ).run(
                        cmd=ffmpeg_path,
                        capture_stdout=True,
                        capture_stderr=True,
                        overwrite_output=True
                    )
                    self.update_status("오디오 추출 완료!", 20)
                    self.log_message("오디오 추출 완료!")
                except ffmpeg.Error as e:
                    self.log_message(f"ffmpeg 오류: {e.stderr.decode()}")
                    raise
            
            # 3. 오디오 파일 검증 (25%)
            self.update_status("오디오 파일 검증 중...", 25)
            try:
                data, samplerate = sf.read(audio_file)
                duration = len(data) / samplerate
                self.audio_duration = duration  # 변환 시간 계산을 위해 저장
                self.log_message(f"오디오 정보: {samplerate}Hz, {duration:.1f}초")
                self.update_status("오디오 파일 검증 완료", 30)
            except Exception as e:
                self.log_message(f"오디오 파일 읽기 오류: {e}")
                raise
            
            # 4. Whisper 모델 로드 (30% -> 40%)
            self.update_status("Whisper 모델 로딩 중...", 35)
            self.log_message("Whisper 모델을 로드하는 중...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_size = self.model_var.get()
            model = whisper.load_model(model_size, device=device)
            
            self.update_status("모델 로드 완료!", 40)
            self.log_message(f"모델 로드 완료! (모델: {model_size}, 장치: {device})")
            
            # 5. 텍스트 변환 (40% -> 80%)
            self.update_status("음성을 텍스트로 변환 중...", 50)
            self.log_message("음성을 텍스트로 변환하는 중...")
            
            # 진행률 시뮬레이션을 위한 스레드 시작 (파일 길이와 모델 크기 반영)
            conversion_thread = threading.Thread(
                target=self.simulate_conversion_progress, 
                args=(self.audio_duration, model_size)
            )
            conversion_thread.daemon = True
            conversion_thread.start()
            
            # Whisper transcribe 실행 시작 시간 기록
            import time
            start_time = time.time()
            
            # Whisper transcribe 실행
            result = model.transcribe(audio_file, task="transcribe")
            
            # 실제 변환 시간 기록
            actual_time = time.time() - start_time
            
            # 진행률 시뮬레이션 중지
            self.stop_simulation = True
            
            self.update_status("텍스트 변환 완료!", 80)
            self.log_message(f"변환 완료! (실제 소요 시간: {actual_time:.1f}초)")
            
            # 6. 결과 저장 (80% -> 95%)
            self.update_status("결과 파일 저장 중...", 85)
            self.save_results(result, output_text, output_subtitle)
            self.last_output_text = output_text  # 요약 기능을 위해 저장
            self.update_status("파일 저장 완료!", 95)
            
            # 7. 임시 파일 정리 및 완료 (95% -> 100%)
            if file_ext not in audio_extensions:
                try:
                    os.remove(audio_file)
                    os.rmdir(temp_dir)
                    self.log_message("임시 파일 정리 완료")
                except:
                    pass
            
            self.update_status("변환 완료!", 100)
            self.log_message("")
            self.log_message("=== 변환 완료 ===")
            self.log_message(f"텍스트 파일: {output_text}")
            self.log_message(f"자막 파일: {output_subtitle}")
            
            # 성능 통계 추가
            processing_speed = self.audio_duration / actual_time if actual_time > 0 else 0
            self.log_message(f"처리 속도: {processing_speed:.2f}x (실시간 대비)")
            
            # 완료 메시지
            self.root.after(0, lambda: messagebox.showinfo("완료", 
                f"변환이 완료되었습니다!\n\n"
                f"텍스트 파일: {os.path.basename(output_text)}\n"
                f"자막 파일: {os.path.basename(output_subtitle)}\n\n"
                f"처리 통계:\n"
                f"• 오디오 길이: {self.audio_duration:.1f}초\n"
                f"• 변환 시간: {actual_time:.1f}초\n"
                f"• 처리 속도: {processing_speed:.2f}x\n\n"
                f"저장 위치: {self.output_dir}\n\n"
                f"VLC 자막 사용법:\n"
                f"1. VLC에서 동영상 열기\n"
                f"2. 자막 → 자막 파일 추가\n"
                f"3. {os.path.basename(output_subtitle)} 선택"))
            
        except Exception as e:
            self.log_message(f"오류 발생: {str(e)}")
            self.stop_simulation = True  # 오류 시에도 시뮬레이션 중지
            self.update_status("오류 발생", 0)
            self.root.after(0, lambda: messagebox.showerror("오류", f"변환 중 오류가 발생했습니다:\n{str(e)}"))
        
        finally:
            # UI 상태 복원
            self.stop_simulation = True  # 시뮬레이션 중지
            self.root.after(0, self.conversion_finished)
    
    def save_results(self, result, output_text, output_subtitle):
        """결과를 파일로 저장"""
        def format_timestamp_ms(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02}.{ms:03}"
        
        def format_srt_timestamp(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        
        # 텍스트 파일 저장
        with open(output_text, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                timestamp = format_timestamp_ms(segment['start'])
                f.write(f"{timestamp} {segment['text'].strip()}\n")
        
        # SRT 자막 파일 저장
        with open(output_subtitle, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start_time = format_srt_timestamp(segment['start'])
                end_time = format_srt_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        self.log_message("텍스트 파일 저장 완료")
        self.log_message("SRT 자막 파일 저장 완료")
    
    def conversion_finished(self):
        """변환 완료 후 UI 상태 복원"""
        self.is_converting = False
        self.convert_button.config(state="normal")
        # 진행률 바를 100%로 유지하거나 0으로 리셋
        if self.progress['value'] != 100:
            self.progress['value'] = 0
            self.progress_label.config(text="0%")
            self.update_status("대기 중...")

def main():
    # tkinterdnd2가 없는 경우 일반 Tk 사용
    try:
        root = TkinterDnD.Tk()
    except:
        root = tk.Tk()
        messagebox.showwarning("알림", "드래그 앤 드롭 기능을 사용하려면 'tkinterdnd2' 패키지를 설치해주세요.\npip install tkinterdnd2")
    
    app = AudioTextConverterGUI(root)
    
    # 창 닫기 시 확인
    def on_closing():
        if app.is_converting:
            if messagebox.askokcancel("종료", "변환이 진행 중입니다. 정말 종료하시겠습니까?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
