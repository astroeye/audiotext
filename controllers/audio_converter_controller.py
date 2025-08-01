"""
메인 컨트롤러 - MVC 패턴의 Controller
"""
import threading
import os
import tempfile
from typing import Optional

from models.audio_processor import AudioProcessor
from services.llm_service import LLMService
from views.audio_converter_view import AudioConverterView, APISettingsDialog


class AudioConverterController:
    def __init__(self, view: AudioConverterView):
        self.view = view
        self.audio_processor = AudioProcessor()
        self.llm_service = LLMService()
        
        # 상태 변수
        self.last_output_text = ""
        self.temp_files = []
        
        # 이벤트 콜백 등록
        self.setup_callbacks()
        
        # 초기 API 상태 업데이트
        self.view.root.after(100, self.update_api_status)
    
    def setup_callbacks(self):
        """뷰 이벤트 콜백 설정"""
        self.view.register_callback('on_file_select', self.on_file_select)
        self.view.register_callback('on_file_drop', self.on_file_drop)
        self.view.register_callback('on_folder_select', self.on_folder_select)
        self.view.register_callback('on_convert_start', self.on_convert_start)
        self.view.register_callback('on_summarize', self.on_summarize)
        self.view.register_callback('on_api_settings', self.on_api_settings)
    
    def on_file_select(self, file_path: str):
        """파일 선택 이벤트 처리"""
        self.view.log_message(f"파일 선택됨: {os.path.basename(file_path)}")
        
        # 기본 출력 폴더 설정 (파일과 같은 폴더)
        if not self.view.output_dir:
            output_dir = os.path.dirname(file_path)
            self.view.set_output_folder(output_dir)
    
    def on_file_drop(self, file_path: str):
        """파일 드롭 이벤트 처리"""
        self.on_file_select(file_path)
    
    def on_folder_select(self, folder_path: str):
        """폴더 선택 이벤트 처리"""
        self.view.log_message(f"출력 폴더 선택됨: {folder_path}")
    
    def on_convert_start(self, input_file: str, output_dir: str, model: str, speaker_diarization: bool):
        """변환 시작 이벤트 처리"""
        if not input_file:
            self.view.show_message("오류", "입력 파일을 선택해주세요.", "error")
            return
        
        if not output_dir:
            self.view.show_message("오류", "출력 폴더를 선택해주세요.", "error")
            return
        
        # 백그라운드에서 변환 실행
        thread = threading.Thread(
            target=self.convert_audio_to_text,
            args=(input_file, output_dir, model, speaker_diarization),
            daemon=True
        )
        thread.start()
    
    def on_summarize(self, api_type: str, summary_type: str):
        """요약 시작 이벤트 처리"""
        # 변환된 텍스트 파일이 있는지 확인
        if not self.last_output_text or not os.path.exists(self.last_output_text):
            # 텍스트 파일을 직접 선택하도록 함
            text_file = self.view.select_text_file()
            if not text_file:
                return
            self.last_output_text = text_file
        else:
            # 기존 파일 사용할지 새 파일 선택할지 묻기
            use_existing = self.view.ask_question(
                "텍스트 파일 선택", 
                f"기존 변환된 파일을 사용하시겠습니까?\n\n{os.path.basename(self.last_output_text)}\n\n"
                "'아니오'를 선택하면 새 파일을 선택할 수 있습니다."
            )
            
            if not use_existing:
                text_file = self.view.select_text_file()
                if text_file:
                    self.last_output_text = text_file
                else:
                    return
        
        # 백그라운드에서 요약 실행
        thread = threading.Thread(
            target=self.run_summarization,
            args=(api_type, summary_type, self.last_output_text),
            daemon=True
        )
        thread.start()
    
    def on_api_settings(self):
        """API 설정 창 표시"""
        dialog = APISettingsDialog(
            self.view.root, 
            self.llm_service.api_keys,
            self.llm_service.test_api_key
        )
        
        result = dialog.show()
        if result is not None:
            self.llm_service.update_api_keys(result)
            self.update_api_status()
            self.view.log_message("API 키가 업데이트되었습니다.")
    
    def convert_audio_to_text(self, input_file: str, output_dir: str, model: str, speaker_diarization: bool):
        """오디오를 텍스트로 변환"""
        self.view.set_conversion_state(True)
        self.view.set_progress("변환 준비 중...", True)
        self.view.clear_log()
        
        try:
            # 1. 파일 검증 및 변환
            self.view.log_message("📁 파일 검증 중...")
            audio_file, duration = self.audio_processor.validate_and_convert_audio(input_file)
            
            # 임시 파일 추적
            if audio_file != input_file:
                self.temp_files.append(audio_file)
            
            self.view.log_message(f"✅ 파일 검증 완료 (길이: {duration:.1f}초)")
            
            # 2. 음성 인식 실행
            self.view.set_progress("음성 인식 중...")
            result = self.audio_processor.transcribe_audio(
                audio_file, 
                model, 
                speaker_diarization,
                progress_callback=self.view.log_message
            )
            
            # 3. 결과 파일 저장
            self.view.set_progress("결과 저장 중...")
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # 텍스트 파일 저장
            txt_file = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                if speaker_diarization and 'segments' in result:
                    # 화자 분리된 결과
                    for segment in result['segments']:
                        speaker = segment.get('speaker', '화자1')
                        text = segment['text'].strip()
                        start_time = segment['start']
                        end_time = segment['end']
                        f.write(f"[{start_time:.1f}s-{end_time:.1f}s] {speaker}: {text}\n")
                else:
                    # 일반 결과
                    f.write(result['text'])
            
            # SRT 파일 저장 (자막 형식)
            if 'segments' in result and result['segments']:
                srt_file = os.path.join(output_dir, f"{base_name}.srt")
                with open(srt_file, 'w', encoding='utf-8') as f:
                    for i, segment in enumerate(result['segments'], 1):
                        start_time = self.seconds_to_srt_time(segment['start'])
                        end_time = self.seconds_to_srt_time(segment['end'])
                        text = segment['text'].strip()
                        
                        if speaker_diarization:
                            speaker = segment.get('speaker', '화자1')
                            text = f"{speaker}: {text}"
                        
                        f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
                
                self.view.log_message(f"📄 SRT 파일 저장: {os.path.basename(srt_file)}")
            
            self.last_output_text = txt_file
            
            # 완료 메시지
            self.view.set_progress("변환 완료!")
            self.view.log_message(f"✅ 변환 완료!")
            self.view.log_message(f"📄 텍스트 파일: {os.path.basename(txt_file)}")
            self.view.log_message(f"📁 저장 위치: {output_dir}")
            
            if speaker_diarization:
                speaker_count = len(set(segment.get('speaker', '화자1') for segment in result.get('segments', [])))
                self.view.log_message(f"👥 감지된 화자 수: {speaker_count}명")
            
            self.view.show_message("완료", f"변환이 완료되었습니다!\n\n저장 위치: {output_dir}")
            
        except Exception as e:
            self.view.set_progress("변환 실패")
            self.view.log_message(f"❌ 오류 발생: {str(e)}")
            self.view.show_message("오류", f"변환 중 오류가 발생했습니다:\n\n{str(e)}", "error")
        
        finally:
            self.view.set_conversion_state(False)
            self.view.set_progress("", False)
            
            # 임시 파일 정리
            self.audio_processor.cleanup_temp_files(self.temp_files)
            self.temp_files.clear()
    
    def run_summarization(self, api_type: str, summary_type: str, text_file_path: str):
        """텍스트 요약 실행"""
        try:
            self.view.log_message(f"📖 요약 시작 ({api_type.upper()})...")
            self.view.set_progress("텍스트 요약 중...", True)
            
            # 텍스트 파일 읽기
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                raise ValueError("텍스트 파일이 비어있습니다.")
            
            # 요약 실행
            summary = self.llm_service.summarize_text(text, api_type, summary_type)
            
            if not summary:
                raise ValueError("요약 결과가 비어있습니다.")
            
            # 요약 결과 저장
            base_name = os.path.splitext(text_file_path)[0]
            summary_type_name = "회의록" if summary_type == "meeting" else "요약"
            summary_file = f"{base_name}_{api_type}_{summary_type_name}.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {api_type.upper()} {summary_type_name} ===\n\n")
                f.write(summary)
                f.write(f"\n\n=== 원본 파일 ===\n{os.path.basename(text_file_path)}")
            
            self.view.log_message(f"✅ 요약 완료!")
            self.view.log_message(f"📄 요약 파일: {os.path.basename(summary_file)}")
            
            # 완료 메시지
            self.view.show_message(
                "요약 완료", 
                f"{api_type.upper()} 요약이 완료되었습니다!\n\n"
                f"파일: {os.path.basename(summary_file)}\n"
                f"위치: {os.path.dirname(summary_file)}"
            )
            
        except Exception as e:
            self.view.log_message(f"❌ 요약 실패: {str(e)}")
            self.view.show_message("요약 오류", f"요약 중 오류가 발생했습니다:\n\n{str(e)}", "error")
        
        finally:
            self.view.set_progress("", False)
    
    def seconds_to_srt_time(self, seconds: float) -> str:
        """초를 SRT 시간 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def update_api_status(self):
        """API 상태 업데이트"""
        status = self.llm_service.get_api_status()
        self.view.update_api_status(status)
