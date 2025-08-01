"""
GUI 뷰 컴포넌트
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
from typing import Callable, Optional


class AudioConverterView:
    def __init__(self, root):
        self.root = root
        self.callbacks = {}
        
        # UI 상태 변수들
        self.input_file = ""
        self.output_dir = ""
        self.is_converting = False
        self.enable_speaker_diarization = tk.BooleanVar(value=False)
        
        self.setup_window()
        self.create_widgets()
        self.setup_drag_drop()
    
    def setup_window(self):
        """윈도우 기본 설정"""
        self.root.title("오디오/동영상 → 텍스트 변환기 + AI 요약")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
    
    def create_widgets(self):
        """GUI 위젯 생성"""
        # 메인 프레임 (2열 레이아웃)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 그리드 설정 (왼쪽:오른쪽 = 2:3 비율로 변경)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)  # 메인 영역이 확장되도록
        
        # === 왼쪽 컬럼 ===
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(3, weight=1)  # 마지막 섹션이 확장되도록
        
        # 파일 선택 섹션
        file_frame = ttk.LabelFrame(left_frame, text="파일 선택", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        self.select_file_btn = ttk.Button(file_frame, text="파일 선택", command=self.select_input_file)
        self.select_file_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.select_output_btn = ttk.Button(file_frame, text="출력 폴더", command=self.select_output_folder)
        self.select_output_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 변환 옵션 섹션
        options_frame = ttk.LabelFrame(left_frame, text="변환 옵션", padding="5")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(0, weight=1)
        
        # Whisper 모델 선택
        ttk.Label(options_frame, text="Whisper 모델:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large"], 
                                  state="readonly", width=12)
        model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 화자 분리 옵션
        self.speaker_check = ttk.Checkbutton(options_frame, text="화자 분리", 
                                           variable=self.enable_speaker_diarization)
        self.speaker_check.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # 변환 실행 버튼
        self.convert_btn = ttk.Button(options_frame, text="변환 시작", command=self.start_conversion)
        self.convert_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 요약 섹션
        summary_frame = ttk.LabelFrame(left_frame, text="AI 요약", padding="5")
        summary_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        summary_frame.columnconfigure(0, weight=1)
        
        # API 설정 버튼
        self.api_settings_btn = ttk.Button(summary_frame, text="API 설정", command=self.show_api_settings)
        self.api_settings_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # API 상태 표시
        self.api_status_frame = ttk.Frame(summary_frame)
        self.api_status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        self.api_status_frame.columnconfigure(1, weight=1)
        
        # 요약 타입 선택
        ttk.Label(summary_frame, text="요약 형식:").grid(row=2, column=0, sticky=tk.W, pady=(5, 2))
        self.summary_type_var = tk.StringVar(value="general")
        summary_type_frame = ttk.Frame(summary_frame)
        summary_type_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Radiobutton(summary_type_frame, text="일반", variable=self.summary_type_var, 
                       value="general").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(summary_type_frame, text="회의록", variable=self.summary_type_var, 
                       value="meeting").grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # LLM 선택 및 요약 버튼들
        llm_frame = ttk.Frame(summary_frame)
        llm_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        llm_frame.columnconfigure(0, weight=1)
        
        self.openai_btn = ttk.Button(llm_frame, text="OpenAI 요약", 
                                   command=lambda: self.summarize_text("openai"))
        self.openai_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=1)
        
        self.anthropic_btn = ttk.Button(llm_frame, text="Claude 요약", 
                                      command=lambda: self.summarize_text("anthropic"))
        self.anthropic_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=1)
        
        self.gemini_btn = ttk.Button(llm_frame, text="Gemini 요약", 
                                   command=lambda: self.summarize_text("gemini"))
        self.gemini_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=1)
        
        # === 오른쪽 컬럼 ===
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)  # 로그 영역 확장
        
        # 선택된 파일 정보
        info_frame = ttk.LabelFrame(right_frame, text="선택된 파일", padding="5")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="입력 파일:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_label = ttk.Label(info_frame, text="선택된 파일 없음", foreground="gray", 
                                   wraplength=400)
        self.input_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        ttk.Label(info_frame, text="출력 폴더:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_label = ttk.Label(info_frame, text="선택된 폴더 없음", foreground="gray", 
                                    wraplength=400)
        self.output_label.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # 진행률 표시
        self.progress_var = tk.StringVar(value="")
        self.progress_label = ttk.Label(info_frame, textvariable=self.progress_var, 
                                      foreground="blue")
        self.progress_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.progress_bar = ttk.Progressbar(info_frame, mode='indeterminate')
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # 로그 영역
        log_frame = ttk.LabelFrame(right_frame, text="진행 상황", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 스크롤바와 텍스트 위젯
        text_scroll_frame = ttk.Frame(log_frame)
        text_scroll_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scroll_frame.columnconfigure(0, weight=1)
        text_scroll_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(text_scroll_frame, height=12, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_scroll_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 창 크기 조정 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def setup_drag_drop(self):
        """드래그 앤 드롭 설정"""
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop(self, event):
        """파일 드롭 이벤트 처리"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            self.set_input_file(file_path)
            if 'on_file_drop' in self.callbacks:
                self.callbacks['on_file_drop'](file_path)
    
    def select_input_file(self):
        """입력 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="오디오/동영상 파일 선택",
            filetypes=[
                ("모든 지원 파일", "*.wav *.mp3 *.mp4 *.avi *.mov *.mkv *.m4a *.flac *.ogg *.webm"),
                ("오디오 파일", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("동영상 파일", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("모든 파일", "*.*")
            ]
        )
        if file_path:
            self.set_input_file(file_path)
            if 'on_file_select' in self.callbacks:
                self.callbacks['on_file_select'](file_path)
    
    def select_output_folder(self):
        """출력 폴더 선택"""
        folder_path = filedialog.askdirectory(title="출력 폴더 선택")
        if folder_path:
            self.set_output_folder(folder_path)
            if 'on_folder_select' in self.callbacks:
                self.callbacks['on_folder_select'](folder_path)
    
    def set_input_file(self, file_path: str):
        """입력 파일 설정"""
        self.input_file = file_path
        filename = os.path.basename(file_path) if file_path else "선택된 파일 없음"
        self.input_label.config(text=filename, foreground="black" if file_path else "gray")
    
    def set_output_folder(self, folder_path: str):
        """출력 폴더 설정"""
        self.output_dir = folder_path
        folder_name = folder_path if folder_path else "선택된 폴더 없음"
        self.output_label.config(text=folder_name, foreground="black" if folder_path else "gray")
    
    def start_conversion(self):
        """변환 시작"""
        if 'on_convert_start' in self.callbacks:
            model = self.model_var.get()
            speaker_diarization = self.enable_speaker_diarization.get()
            self.callbacks['on_convert_start'](self.input_file, self.output_dir, model, speaker_diarization)
    
    def summarize_text(self, api_type: str):
        """텍스트 요약"""
        if 'on_summarize' in self.callbacks:
            summary_type = self.summary_type_var.get()
            self.callbacks['on_summarize'](api_type, summary_type)
    
    def show_api_settings(self):
        """API 설정 창 표시"""
        if 'on_api_settings' in self.callbacks:
            self.callbacks['on_api_settings']()
    
    def log_message(self, message: str):
        """로그 메시지 추가"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """로그 지우기"""
        self.log_text.delete(1.0, tk.END)
    
    def set_progress(self, message: str, show_bar: bool = False):
        """진행률 설정"""
        self.progress_var.set(message)
        if show_bar:
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
    
    def set_conversion_state(self, converting: bool):
        """변환 상태 설정"""
        self.is_converting = converting
        state = "disabled" if converting else "normal"
        self.convert_btn.config(state=state)
        self.select_file_btn.config(state=state)
        self.select_output_btn.config(state=state)
    
    def update_api_status(self, status: dict):
        """API 상태 업데이트"""
        # 기존 상태 위젯들 제거
        for widget in self.api_status_frame.winfo_children():
            widget.destroy()
        
        api_names = {"openai": "OpenAI", "anthropic": "Claude", "gemini": "Gemini"}
        
        for i, (api_type, is_available) in enumerate(status.items()):
            color = "green" if is_available else "red"
            symbol = "●" if is_available else "○"
            text = f"{symbol} {api_names.get(api_type, api_type)}"
            
            label = ttk.Label(self.api_status_frame, text=text, foreground=color)
            label.grid(row=i//2, column=i%2, sticky=tk.W, padx=(0, 10))
    
    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """메시지 다이얼로그 표시"""
        if msg_type == "error":
            messagebox.showerror(title, message)
        elif msg_type == "warning":
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)
    
    def ask_question(self, title: str, question: str) -> bool:
        """질문 다이얼로그 표시"""
        return messagebox.askyesno(title, question)
    
    def select_text_file(self) -> Optional[str]:
        """텍스트 파일 선택 다이얼로그"""
        return filedialog.askopenfilename(
            title="요약할 텍스트 파일 선택",
            filetypes=[
                ("텍스트 파일", "*.txt"),
                ("SRT 파일", "*.srt"), 
                ("모든 파일", "*.*")
            ]
        )
    
    def register_callback(self, event_name: str, callback: Callable):
        """콜백 함수 등록"""
        self.callbacks[event_name] = callback
    
    def get_selected_model(self) -> str:
        """선택된 Whisper 모델 반환"""
        return self.model_var.get()
    
    def get_speaker_diarization_enabled(self) -> bool:
        """화자 분리 옵션 상태 반환"""
        return self.enable_speaker_diarization.get()
    
    def get_summary_type(self) -> str:
        """선택된 요약 타입 반환"""
        return self.summary_type_var.get()


class APISettingsDialog:
    def __init__(self, parent, api_keys: dict, test_callback: Callable):
        self.parent = parent
        self.api_keys = api_keys.copy()
        self.test_callback = test_callback
        self.result = None
        
        self.create_dialog()
    
    def create_dialog(self):
        """API 설정 다이얼로그 생성"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("API 키 설정")
        self.dialog.geometry("500x400")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # 다이얼로그를 부모 창 중앙에 배치
        self.dialog.geometry("+%d+%d" % (
            self.parent.winfo_rootx() + 50,
            self.parent.winfo_rooty() + 50
        ))
        
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        title_label = ttk.Label(main_frame, text="AI 요약 서비스 API 키 설정", 
                               font=("", 12, "bold"))
        title_label.pack(pady=(0, 20))
        
        # API 키 입력 필드들
        self.entries = {}
        self.show_vars = {}
        
        api_info = {
            "openai": {"name": "OpenAI (GPT)", "url": "https://platform.openai.com/api-keys"},
            "anthropic": {"name": "Anthropic (Claude)", "url": "https://console.anthropic.com/"},
            "gemini": {"name": "Google (Gemini)", "url": "https://makersuite.google.com/app/apikey"}
        }
        
        for api_type, info in api_info.items():
            # API 이름과 링크
            api_frame = ttk.LabelFrame(main_frame, text=info["name"], padding="10")
            api_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 키 입력 프레임
            key_frame = ttk.Frame(api_frame)
            key_frame.pack(fill=tk.X)
            key_frame.columnconfigure(0, weight=1)
            
            # 키 입력 필드
            self.show_vars[api_type] = tk.BooleanVar()
            entry = ttk.Entry(key_frame, show="*", width=50)
            entry.insert(0, self.api_keys.get(api_type, ""))
            entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
            self.entries[api_type] = entry
            
            # 보기/숨기기 버튼
            show_btn = ttk.Checkbutton(key_frame, text="보기", 
                                     variable=self.show_vars[api_type],
                                     command=lambda e=entry, v=self.show_vars[api_type]: 
                                     e.config(show="" if v.get() else "*"))
            show_btn.grid(row=0, column=1)
            
            # API 키 발급 링크
            link_label = ttk.Label(api_frame, text=f"키 발급: {info['url']}", 
                                 foreground="blue", cursor="hand2")
            link_label.pack(anchor=tk.W, pady=(5, 0))
            # 실제 브라우저 열기는 구현하지 않음 (보안상 이유)
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # 테스트 버튼
        test_btn = ttk.Button(button_frame, text="연결 테스트", command=self.test_keys)
        test_btn.pack(side=tk.LEFT)
        
        # 저장/취소 버튼
        ttk.Button(button_frame, text="취소", command=self.cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="저장", command=self.save_and_close).pack(side=tk.RIGHT)
        
        # 엔터/ESC 키 바인딩
        self.dialog.bind('<Return>', lambda e: self.save_and_close())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
        
        # 첫 번째 입력 필드에 포커스
        list(self.entries.values())[0].focus()
    
    def test_keys(self):
        """API 키 테스트"""
        results = {}
        for api_type, entry in self.entries.items():
            key = entry.get().strip()
            if key:
                try:
                    results[api_type] = self.test_callback(api_type, key)
                except:
                    results[api_type] = False
            else:
                results[api_type] = False
        
        # 결과 표시
        message = "API 키 테스트 결과:\n\n"
        for api_type, success in results.items():
            api_names = {"openai": "OpenAI", "anthropic": "Claude", "gemini": "Gemini"}
            status = "✓ 연결 성공" if success else "✗ 연결 실패"
            message += f"{api_names[api_type]}: {status}\n"
        
        messagebox.showinfo("테스트 결과", message)
    
    def save_and_close(self):
        """저장하고 닫기"""
        for api_type, entry in self.entries.items():
            self.api_keys[api_type] = entry.get().strip()
        
        self.result = self.api_keys
        self.dialog.destroy()
    
    def cancel(self):
        """취소"""
        self.result = None
        self.dialog.destroy()
    
    def show(self) -> Optional[dict]:
        """다이얼로그 표시 및 결과 반환"""
        self.dialog.wait_window()
        return self.result
