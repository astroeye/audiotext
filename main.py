"""
오디오/동영상 → 텍스트 변환기 + AI 요약
MVC 패턴으로 구조화된 메인 애플리케이션
"""
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tkinterdnd2 import TkinterDnD
    import tkinter as tk
except ImportError as e:
    print("필수 패키지가 설치되지 않았습니다.")
    print("다음 명령어로 설치해주세요:")
    print("pip install tkinterdnd2")
    sys.exit(1)

from views.audio_converter_view import AudioConverterView
from controllers.audio_converter_controller import AudioConverterController


def main():
    """메인 애플리케이션 실행"""
    # tkinterdnd2를 사용한 루트 윈도우 생성
    root = TkinterDnD.Tk()
    
    try:
        # MVC 패턴으로 애플리케이션 구성
        view = AudioConverterView(root)
        controller = AudioConverterController(view)
        
        # 애플리케이션 시작
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()
