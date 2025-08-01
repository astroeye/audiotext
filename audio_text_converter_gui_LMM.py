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
import numpy as np
import librosa

class AudioTextConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("오디오/동영상 → 텍스트 변환기 + AI 요약")
        self.root.geometry("650x800")
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
        self.enable_speaker_diarization = tk.BooleanVar(value=False)  # 화자 분리 옵션
        
        # API 키 설정 파일 로드
        self.load_api_keys()
        
        # GUI 생성
        self.create_widgets()
        
        # 드래그 앤 드롭 설정
        self.setup_drag_drop()
        
        # 시작할 때 API 상태 표시 (GUI 생성 후)
        self.root.after(100, self.update_api_status)
        
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
    
    def simple_speaker_diarization(self, audio_file, segments):
        """간단한 화자 분리 - 오디오 특성 기반 (개선된 버전)"""
        try:
            self.log_message("화자 분리 분석 중...")
            
            # 오디오 로드
            y, sr = librosa.load(audio_file, sr=16000)
            
            speakers = []
            speaker_features = []  # 각 화자의 특성을 저장
            
            # 1단계: 모든 세그먼트의 특성 추출
            segment_features = []
            valid_segments = []
            
            for i, segment in enumerate(segments):
                start_time = segment['start']
                end_time = segment['end']
                
                # 해당 시간 구간의 오디오 추출
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = y[start_sample:end_sample]
                
                # 최소 길이 확인 (0.5초 이상)
                if len(audio_segment) < sr * 0.5:
                    continue
                
                try:
                    # 피치 추출 (더 안정적인 방법)
                    pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr, threshold=0.1)
                    valid_pitches = pitches[pitches > 80]  # 80Hz 이상만 (인간 음성 범위)
                    pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
                    pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0
                    
                    # 스펙트럴 중심 (음성의 밝기)
                    spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
                    spectral_centroid = np.mean(spectral_centroids)
                    
                    # RMS 에너지
                    rms = librosa.feature.rms(y=audio_segment)
                    energy = np.mean(rms)
                    
                    # ZCR (Zero Crossing Rate) - 음성/비음성 구분
                    zcr = librosa.feature.zero_crossing_rate(audio_segment)
                    zcr_mean = np.mean(zcr)
                    
                    # 유효한 피치가 있는 경우만 포함
                    if pitch_mean > 80:
                        segment_features.append({
                            'pitch_mean': pitch_mean,
                            'pitch_std': pitch_std,
                            'spectral_centroid': spectral_centroid,
                            'energy': energy,
                            'zcr': zcr_mean,
                            'segment_idx': i
                        })
                        valid_segments.append(i)
                
                except Exception as e:
                    continue
            
            if len(segment_features) < 2:
                # 특성 추출이 불가능한 경우 모두 화자 1로 처리
                return ["화자 1"] * len(segments)
            
            # 2단계: 첫 번째 화자 초기화 (첫 10개 세그먼트의 평균)
            init_count = min(10, len(segment_features))
            first_speaker_features = segment_features[:init_count]
            
            speaker_profiles = [{
                'pitch_mean': np.mean([f['pitch_mean'] for f in first_speaker_features]),
                'pitch_std': np.mean([f['pitch_std'] for f in first_speaker_features]),
                'spectral_centroid': np.mean([f['spectral_centroid'] for f in first_speaker_features]),
                'energy': np.mean([f['energy'] for f in first_speaker_features]),
                'count': init_count
            }]
            
            # 3단계: 각 세그먼트를 화자에 할당
            speaker_assignments = []
            
            for i, features in enumerate(segment_features):
                best_speaker = 0
                min_distance = float('inf')
                
                # 기존 화자들과 비교
                for speaker_idx, profile in enumerate(speaker_profiles):
                    # 정규화된 거리 계산
                    pitch_diff = abs(features['pitch_mean'] - profile['pitch_mean']) / max(profile['pitch_mean'], 1)
                    spectral_diff = abs(features['spectral_centroid'] - profile['spectral_centroid']) / max(profile['spectral_centroid'], 1)
                    energy_diff = abs(features['energy'] - profile['energy']) / max(profile['energy'], 1e-6)
                    
                    # 가중 거리 (피치가 가장 중요)
                    distance = pitch_diff * 3.0 + spectral_diff * 1.5 + energy_diff * 1.0
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = speaker_idx
                
                # 새 화자 생성 조건 (매우 엄격)
                if (min_distance > 0.8 and  # 거리 임계값 증가
                    len(speaker_profiles) < 3 and  # 최대 3명으로 제한
                    i > 20 and  # 최소 20개 세그먼트 후에만
                    features['pitch_mean'] > 0):
                    
                    # 새 화자 생성
                    speaker_profiles.append({
                        'pitch_mean': features['pitch_mean'],
                        'pitch_std': features['pitch_std'],
                        'spectral_centroid': features['spectral_centroid'],
                        'energy': features['energy'],
                        'count': 1
                    })
                    best_speaker = len(speaker_profiles) - 1
                    speaker_assignments.append(best_speaker)
                else:
                    # 기존 화자에 할당 및 프로필 업데이트
                    profile = speaker_profiles[best_speaker]
                    alpha = 0.1  # 학습률
                    
                    profile['pitch_mean'] = profile['pitch_mean'] * (1 - alpha) + features['pitch_mean'] * alpha
                    profile['spectral_centroid'] = profile['spectral_centroid'] * (1 - alpha) + features['spectral_centroid'] * alpha
                    profile['energy'] = profile['energy'] * (1 - alpha) + features['energy'] * alpha
                    profile['count'] += 1
                    
                    speaker_assignments.append(best_speaker)
            
            # 4단계: 결과를 전체 세그먼트에 매핑
            speakers = ["화자 1"] * len(segments)
            assignment_idx = 0
            
            for i in range(len(segments)):
                if i in valid_segments:
                    speaker_id = speaker_assignments[assignment_idx] + 1
                    speakers[i] = f"화자 {speaker_id}"
                    assignment_idx += 1
                else:
                    # 이전 세그먼트의 화자를 유지
                    if i > 0:
                        speakers[i] = speakers[i-1]
            
            # 5단계: 후처리 - 연속성 개선
            self.smooth_speaker_transitions(speakers, segments)
            
            # 6단계: 소수 화자 병합
            self.merge_minor_speakers(speakers, segments)
            
            final_speaker_count = len(set(speakers))
            self.log_message(f"화자 분리 완료: 총 {final_speaker_count}명의 화자 감지")
            
            return speakers
            
        except Exception as e:
            self.log_message(f"화자 분리 오류: {e}")
            return ["화자 1"] * len(segments)
    
    def smooth_speaker_transitions(self, speakers, segments):
        """화자 전환의 연속성을 개선"""
        if len(speakers) < 3:
            return
        
        # 짧은 화자 변경을 평활화 (3개 이하 연속 세그먼트)
        for i in range(1, len(speakers) - 1):
            current = speakers[i]
            prev = speakers[i-1]
            next_speaker = speakers[i+1] if i+1 < len(speakers) else current
            
            # 단일 세그먼트 이상치 제거
            if current != prev and current != next_speaker and prev == next_speaker:
                speakers[i] = prev
            
        # 짧은 화자 구간 병합 (3개 미만 연속 세그먼트)
        i = 0
        while i < len(speakers):
            current_speaker = speakers[i]
            consecutive_count = 1
            
            # 연속된 같은 화자 세그먼트 개수 세기
            j = i + 1
            while j < len(speakers) and speakers[j] == current_speaker:
                consecutive_count += 1
                j += 1
            
            # 3개 미만의 짧은 구간은 인접 화자에 병합
            if consecutive_count < 3 and i > 0:
                for k in range(i, min(i + consecutive_count, len(speakers))):
                    speakers[k] = speakers[i-1]
            
            i = j
    
    def merge_minor_speakers(self, speakers, segments):
        """너무 적은 발화를 가진 화자들을 주요 화자에 병합 (개선된 버전)"""
        from collections import Counter
        
        # 각 화자의 발화 횟수 계산
        speaker_counts = Counter(speakers)
        total_segments = len(segments)
        
        # 더 엄격한 기준: 전체 발화의 10% 미만을 가진 화자들을 소수 화자로 분류
        minor_threshold = max(5, total_segments * 0.1)  # 최소 5개 또는 전체의 10%
        
        minor_speakers = []
        major_speakers = []
        
        for speaker, count in speaker_counts.items():
            if count < minor_threshold:
                minor_speakers.append(speaker)
            else:
                major_speakers.append((speaker, count))
        
        # 주요 화자를 발화 수로 정렬 (가장 많이 말한 순서)
        major_speakers.sort(key=lambda x: x[1], reverse=True)
        major_speaker_names = [speaker for speaker, _ in major_speakers]
        
        # 소수 화자가 있고, 주요 화자가 2-3명일 때만 병합 수행
        if minor_speakers and 2 <= len(major_speakers) <= 3:
            self.log_message(f"소수 화자 {len(minor_speakers)}명을 주요 화자에 병합합니다.")
            
            # 소수 화자들을 주요 화자에 분산 병합
            for i, speaker in enumerate(speakers):
                if speaker in minor_speakers:
                    # 위치에 따라 적절한 주요 화자에 병합
                    if len(major_speaker_names) >= 2:
                        # 짝수 인덱스는 첫 번째 주요 화자, 홀수는 두 번째 주요 화자에
                        target_speaker = major_speaker_names[i % 2]
                    else:
                        target_speaker = major_speaker_names[0]
                    speakers[i] = target_speaker
        
        # 화자가 너무 많은 경우 (3명 초과) 상위 3명만 유지
        if len(set(speakers)) > 3:
            # 상위 3명의 화자만 유지
            top_speakers = [speaker for speaker, _ in speaker_counts.most_common(3)]
            
            for i, speaker in enumerate(speakers):
                if speaker not in top_speakers:
                    # 가장 가까운 주요 화자로 병합
                    speakers[i] = top_speakers[0]  # 가장 많이 말한 화자로
        
        # 화자 번호 재정렬 (발화 순서대로)
        unique_speakers = []
        for speaker, _ in speaker_counts.most_common():
            if speaker in set(speakers):  # 병합 후에도 남아있는 화자만
                unique_speakers.append(speaker)
        
        # 최대 3명까지만
        unique_speakers = unique_speakers[:3]
        
        speaker_mapping = {}
        for i, speaker in enumerate(unique_speakers, 1):
            speaker_mapping[speaker] = f"화자 {i}"
        
        for i, speaker in enumerate(speakers):
            if speaker in speaker_mapping:
                speakers[i] = speaker_mapping[speaker]
            else:
                speakers[i] = "화자 1"  # 예외 처리
    
    def advanced_speaker_diarization(self, audio_file, segments):
        """고급 화자 분리 - 음성 임베딩 기반 (개선된 버전)"""
        try:
            self.log_message("고급 화자 분리 분석 중...")
            
            # SpeechBrain을 사용한 화자 임베딩
            from speechbrain.pretrained import SpeakerRecognition
            
            # 사전 훈련된 모델 로드
            verification = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # 각 세그먼트의 임베딩 추출 (더 엄격한 필터링)
            embeddings = []
            valid_indices = []
            y, sr = librosa.load(audio_file, sr=16000)
            
            for i, segment in enumerate(segments):
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                audio_segment = y[start_sample:end_sample]
                
                # 최소 1초 이상, 최대 10초 이하의 세그먼트만 사용
                if sr * 1.0 <= len(audio_segment) <= sr * 10.0:
                    try:
                        # 임베딩 추출
                        embedding = verification.encode_batch(torch.tensor(audio_segment).unsqueeze(0))
                        embeddings.append(embedding.squeeze().cpu().numpy())
                        valid_indices.append(i)
                    except:
                        continue
            
            if len(embeddings) < 10:  # 최소 10개 이상의 유효한 임베딩이 필요
                self.log_message("유효한 임베딩이 부족하여 간단한 방법으로 대체합니다.")
                return self.simple_speaker_diarization(audio_file, segments)
            
            # 클러스터링으로 화자 그룹핑 (더 보수적)
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            
            # 2명과 3명만 시도 (더 제한적)
            best_clustering = None
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in [2, 3]:
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage='ward'
                    )
                    cluster_labels = clustering.fit_predict(embeddings)
                    
                    # 클러스터 품질 평가
                    score = silhouette_score(embeddings, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_clustering = cluster_labels
                        best_n_clusters = n_clusters
                        
                except:
                    continue
            
            # 품질이 너무 낮으면 간단한 방법으로 대체
            if best_score < 0.3:  # 실루엣 스코어가 0.3 미만이면
                self.log_message("클러스터링 품질이 낮아 간단한 방법으로 대체합니다.")
                return self.simple_speaker_diarization(audio_file, segments)
            
            # 화자 라벨 생성
            speakers = ["화자 1"] * len(segments)
            
            if best_clustering is not None:
                embedding_idx = 0
                for i in range(len(segments)):
                    if i in valid_indices:
                        speaker_id = best_clustering[embedding_idx] + 1
                        speakers[i] = f"화자 {speaker_id}"
                        embedding_idx += 1
                    else:
                        # 이전 세그먼트의 화자 유지
                        if i > 0:
                            speakers[i] = speakers[i-1]
                
                # 후처리
                self.smooth_speaker_transitions(speakers, segments)
                self.merge_minor_speakers(speakers, segments)
                
                unique_speakers = len(set(speakers))
                self.log_message(f"고급 화자 분리 완료: 총 {unique_speakers}명의 화자 감지 (품질: {best_score:.2f})")
            else:
                speakers = ["화자 1"] * len(segments)
                
            return speakers
            
        except Exception as e:
            self.log_message(f"고급 화자 분리 실패, 간단한 방법으로 대체: {e}")
            return self.simple_speaker_diarization(audio_file, segments)
    
    def load_api_keys(self):
        """API 키 설정 파일 로드"""
        config_file = os.path.join(os.path.dirname(__file__), "api_keys.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_keys = json.load(f)
                    # 빈 문자열이나 None 값 제거
                    self.api_keys = {k: v for k, v in loaded_keys.items() if v and v.strip()}
                    print(f"API 키 로드됨: {list(self.api_keys.keys())}")  # 디버깅용
            else:
                self.api_keys = {}
                print("API 키 파일이 존재하지 않음")  # 디버깅용
        except Exception as e:
            print(f"API 키 로드 오류: {e}")  # 디버깅용
            self.api_keys = {}
    
    def save_api_keys(self):
        """API 키 설정 파일 저장"""
        config_file = os.path.join(os.path.dirname(__file__), "api_keys.json")
        try:
            # 디렉토리 권한 확인
            directory = os.path.dirname(config_file)
            if not os.access(directory, os.W_OK):
                raise Exception(f"디렉토리에 쓰기 권한이 없습니다: {directory}")
            
            # 빈 키는 제거하고 저장
            filtered_keys = {k: v for k, v in self.api_keys.items() if v and v.strip()}
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_keys, f, indent=2, ensure_ascii=False)
            
            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(config_file):
                file_size = os.path.getsize(config_file)
                print(f"API 키 저장 성공: {config_file} (크기: {file_size} bytes)")
                print(f"저장된 키: {list(filtered_keys.keys())}")
                
                # 저장된 내용 재확인
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_content = f.read()
                    print(f"저장된 내용 확인: {saved_content[:100]}...")
            else:
                raise Exception("파일이 생성되지 않았습니다")
            
        except Exception as e:
            error_msg = f"API 키 저장 중 오류가 발생했습니다: {str(e)}\n파일 경로: {config_file}"
            print(f"저장 오류: {error_msg}")
            messagebox.showerror("오류", error_msg)
            raise  # 예외를 다시 발생시켜 호출하는 곳에서 처리할 수 있게 함
    
    def summarize_with_openai(self, text: str, api_key: str, summary_type: str = "general") -> Optional[str]:
        """OpenAI GPT를 사용한 텍스트 요약"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            if summary_type == "meeting":
                system_prompt = """당신은 전문적인 회의록 작성자입니다. 주어진 텍스트를 회의록 형태로 구조화하여 요약해주세요.
다음 형식으로 한국어로 작성해주세요:

## 회의 요약

### 📋 주요 논의사항
- [논의된 주요 주제들을 정리]

### ✅ 결정사항
- [회의에서 내려진 결정들을 정리]

### 📝 액션 아이템
- [누가 무엇을 언제까지 해야 하는지 정리]

### 💡 주요 의견
- [참석자들의 중요한 의견이나 제안사항]

### 📌 다음 단계
- [향후 계획이나 다음 회의 일정 등]"""
                
                user_prompt = f"다음 회의 내용을 위 형식에 맞춰 회의록으로 작성해주세요:\n\n{text}"
            else:
                system_prompt = "당신은 텍스트 요약 전문가입니다. 주어진 텍스트를 명확하고 간결하게 요약해주세요. 한국어로 답변해주세요."
                user_prompt = f"다음 텍스트를 요약해주세요:\n\n{text}"
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,  # 회의록은 더 길 수 있으므로 토큰 수 증가
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            
            # 할당량/크레딧 관련 오류 처리
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                raise Exception("OpenAI API 사용량 한도에 도달했습니다.\n"
                              "• 잠시 후 다시 시도하거나\n"
                              "• OpenAI 계정의 사용량 및 결제 정보를 확인해주세요.\n"
                              "• 또는 다른 API를 사용해주세요.")
            
            # 인증 오류 처리
            elif "401" in error_str or "unauthorized" in error_str.lower() or "invalid api key" in error_str.lower():
                raise Exception("OpenAI API 키가 유효하지 않습니다.\n"
                              "API 설정에서 올바른 키를 입력해주세요.")
            
            # 크레딧 부족 오류
            elif "insufficient" in error_str.lower() or "credit" in error_str.lower():
                raise Exception("OpenAI 계정의 크레딧이 부족합니다.\n"
                              "OpenAI 계정에 결제 수단을 추가하거나\n"
                              "다른 API를 사용해주세요.")
            
            # 기타 오류
            else:
                raise Exception(f"OpenAI API 오류: {error_str}")
    
    def summarize_with_anthropic(self, text: str, api_key: str, summary_type: str = "general") -> Optional[str]:
        """Anthropic Claude를 사용한 텍스트 요약"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            if summary_type == "meeting":
                prompt = f"""다음 회의 내용을 회의록 형태로 구조화하여 한국어로 작성해주세요:

## 회의 요약

### 📋 주요 논의사항
- [논의된 주요 주제들을 정리]

### ✅ 결정사항
- [회의에서 내려진 결정들을 정리]

### 📝 액션 아이템
- [누가 무엇을 언제까지 해야 하는지 정리]

### 💡 주요 의견
- [참석자들의 중요한 의견이나 제안사항]

### 📌 다음 단계
- [향후 계획이나 다음 회의 일정 등]

회의 내용:
{text}"""
            else:
                prompt = f"다음 텍스트를 한국어로 명확하고 간결하게 요약해주세요:\n\n{text}"
            
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,  # 회의록은 더 길 수 있으므로 토큰 수 증가
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            error_str = str(e)
            
            # 할당량/크레딧 관련 오류 처리
            if "429" in error_str or "rate_limit" in error_str.lower():
                raise Exception("Anthropic API 사용량 한도에 도달했습니다.\n"
                              "• 잠시 후 다시 시도하거나\n"
                              "• Anthropic 계정의 사용량을 확인해주세요.\n"
                              "• 또는 다른 API를 사용해주세요.")
            
            # 인증 오류 처리
            elif "401" in error_str or "unauthorized" in error_str.lower():
                raise Exception("Anthropic API 키가 유효하지 않습니다.\n"
                              "API 설정에서 올바른 키를 입력해주세요.")
            
            # 크레딧 부족 오류
            elif "insufficient" in error_str.lower() or "credit" in error_str.lower():
                raise Exception("Anthropic 계정의 크레딧이 부족합니다.\n"
                              "Anthropic 계정에 결제 수단을 추가하거나\n"
                              "다른 API를 사용해주세요.")
            
            # 기타 오류
            else:
                raise Exception(f"Anthropic API 오류: {error_str}")
    
    def summarize_with_gemini(self, text: str, api_key: str, summary_type: str = "general") -> Optional[str]:
        """Google Gemini를 사용한 텍스트 요약"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            if summary_type == "meeting":
                prompt = f"""다음 회의 내용을 회의록 형태로 구조화하여 한국어로 작성해주세요:

## 회의 요약

### 📋 주요 논의사항
- [논의된 주요 주제들을 정리]

### ✅ 결정사항
- [회의에서 내려진 결정들을 정리]

### 📝 액션 아이템
- [누가 무엇을 언제까지 해야 하는지 정리]

### 💡 주요 의견
- [참석자들의 중요한 의견이나 제안사항]

### 📌 다음 단계
- [향후 계획이나 다음 회의 일정 등]

회의 내용:
{text}"""
            else:
                prompt = f"다음 텍스트를 한국어로 명확하고 간결하게 요약해주세요:\n\n{text}"
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            
            # 할당량 초과 오류 처리
            if "429" in error_str or "quota" in error_str.lower():
                if "free_tier" in error_str.lower():
                    raise Exception("Gemini API 무료 할당량이 초과되었습니다.\n"
                                  "• 오늘의 무료 사용량 (50회)을 모두 사용했습니다.\n"
                                  "• 내일 다시 시도하거나 다른 API를 사용해주세요.\n"
                                  "• 또는 Google AI Studio에서 유료 요금제로 업그레이드하세요.")
                else:
                    raise Exception("Gemini API 할당량이 초과되었습니다.\n"
                                  "잠시 후 다시 시도하거나 다른 API를 사용해주세요.")
            
            # API 키 오류 처리
            elif "401" in error_str or "unauthorized" in error_str.lower():
                raise Exception("Gemini API 키가 유효하지 않습니다.\n"
                              "API 설정에서 올바른 키를 입력해주세요.")
            
            # 기타 오류
            else:
                raise Exception(f"Gemini API 오류: {error_str}")
    
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
        
        # 키 표시/숨기기 체크박스
        show_keys_var = tk.BooleanVar()
        show_keys_cb = ttk.Checkbutton(api_frame, text="API 키 보기", variable=show_keys_var)
        show_keys_cb.grid(row=6, column=0, sticky=tk.W, pady=(0, 10))
        
        def toggle_key_visibility():
            show_char = "" if show_keys_var.get() else "*"
            openai_entry.config(show=show_char)
            anthropic_entry.config(show=show_char)
            gemini_entry.config(show=show_char)
        
        show_keys_cb.config(command=toggle_key_visibility)
        
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
            # 입력된 값들을 가져오기
            openai_key = openai_entry.get().strip()
            anthropic_key = anthropic_entry.get().strip()
            gemini_key = gemini_entry.get().strip()
            
            # 디버깅용 출력
            print(f"입력된 키들:")
            print(f"OpenAI: {'설정됨' if openai_key else '비어있음'}")
            print(f"Anthropic: {'설정됨' if anthropic_key else '비어있음'}")
            print(f"Gemini: {'설정됨' if gemini_key else '비어있음'}")
            
            # 최소 하나의 키가 입력되었는지 확인
            if not any([openai_key, anthropic_key, gemini_key]):
                result = messagebox.askquestion("확인", 
                    "모든 API 키가 비어있습니다.\n"
                    "이대로 저장하면 요약 기능을 사용할 수 없습니다.\n"
                    "계속하시겠습니까?")
                if result != 'yes':
                    return
            
            # 키 업데이트 (빈 문자열도 일단 저장)
            self.api_keys["openai"] = openai_key
            self.api_keys["anthropic"] = anthropic_key
            self.api_keys["gemini"] = gemini_key
            
            # 저장 시도
            try:
                self.save_api_keys()
                self.update_api_status()  # API 상태 업데이트
                settings_window.destroy()
                
                # 설정된 키 개수 확인
                configured_count = sum(1 for key in [openai_key, anthropic_key, gemini_key] if key)
                if configured_count > 0:
                    messagebox.showinfo("저장 완료", f"API 키가 저장되었습니다.\n설정된 API: {configured_count}개")
                else:
                    messagebox.showinfo("저장 완료", "설정이 저장되었습니다.\n(API 키가 설정되지 않았습니다)")
                
            except Exception as e:
                messagebox.showerror("저장 실패", f"API 키 저장에 실패했습니다:\n{str(e)}")
        
        def test_keys():
            """API 키 유효성 간단 테스트"""
            openai_key = openai_entry.get().strip()
            anthropic_key = anthropic_entry.get().strip()
            gemini_key = gemini_entry.get().strip()
            
            results = []
            if openai_key:
                results.append(f"OpenAI: {'유효한 형식' if len(openai_key) > 20 and openai_key.startswith('sk-') else '형식 오류'}")
            if anthropic_key:
                results.append(f"Anthropic: {'유효한 형식' if len(anthropic_key) > 20 else '형식 오류'}")
            if gemini_key:
                results.append(f"Gemini: {'유효한 형식' if len(gemini_key) > 20 else '형식 오류'}")
            
            if results:
                messagebox.showinfo("키 형식 검사", "\n".join(results))
            else:
                messagebox.showwarning("검사 불가", "입력된 API 키가 없습니다.")
        
        # 엔터 키로 저장
        def on_enter(event):
            save_and_close()
        
        # 모든 엔트리에 엔터 키 바인딩
        openai_entry.bind('<Return>', on_enter)
        anthropic_entry.bind('<Return>', on_enter)
        gemini_entry.bind('<Return>', on_enter)
        
        # ESC 키로 취소
        def on_escape(event):
            settings_window.destroy()
        
        settings_window.bind('<Escape>', on_escape)
        
        ttk.Button(button_frame, text="키 검사", command=test_keys).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="저장 (Enter)", command=save_and_close).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="취소 (ESC)", command=settings_window.destroy).pack(side=tk.RIGHT)
    
    def summarize_text(self):
        """텍스트 요약 실행"""
        # 변환된 파일이 없으면 사용자가 직접 txt 파일 선택
        selected_file = None
        if not self.last_output_text or not os.path.exists(self.last_output_text):
            result = messagebox.askyesno("파일 선택", 
                "변환된 텍스트 파일이 없습니다.\n"
                "요약할 텍스트 파일을 직접 선택하시겠습니까?")
            if not result:
                return
            
            # 텍스트 파일 선택 대화상자
            filetypes = [
                ("텍스트 파일", "*.txt"),
                ("모든 파일", "*.*")
            ]
            
            selected_file = filedialog.askopenfilename(
                title="요약할 텍스트 파일을 선택하세요", 
                filetypes=filetypes
            )
            
            if not selected_file:
                return
        else:
            selected_file = self.last_output_text
        
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
        
        # API별 특징 정보
        api_info = {
            "openai": "OpenAI GPT (유료, 안정적)",
            "anthropic": "Anthropic Claude (유료, 고품질)",
            "gemini": "Google Gemini (일 50회 무료)"
        }
        
        for name, value in available_apis:
            display_name = api_info.get(value, name)
            ttk.Radiobutton(main_frame, text=display_name, variable=selected_api, 
                           value=value).pack(anchor=tk.W, pady=5)
        
        # 요약 형태 선택
        ttk.Label(main_frame, text="요약 형태를 선택하세요:", 
                 font=("맑은 고딕", 12, "bold")).pack(pady=(20, 10))
        
        summary_type = tk.StringVar(value="general")
        ttk.Radiobutton(main_frame, text="일반 요약", variable=summary_type, 
                       value="general").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(main_frame, text="회의록 형태", variable=summary_type, 
                       value="meeting").pack(anchor=tk.W, pady=2)
        
        # 설명 텍스트
        desc_text = ttk.Label(main_frame, 
                             text="• 일반 요약: 전체 내용을 간결하게 요약\n• 회의록 형태: 주요 논의사항, 결정사항, 액션아이템 등으로 구조화",
                             font=("맑은 고딕", 8), foreground="gray")
        desc_text.pack(pady=(5, 0))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_summary():
            api_window.destroy()
            # 별도 스레드에서 요약 실행 (선택된 파일 전달)
            thread = threading.Thread(target=self.run_summarization, args=(selected_api.get(), summary_type.get(), selected_file))
            thread.daemon = True
            thread.start()
        
        ttk.Button(button_frame, text="요약 시작", command=start_summary).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="취소", command=api_window.destroy).pack(side=tk.RIGHT)
    
    def run_summarization(self, api_type: str, summary_type: str = "general", text_file_path: str = None):
        """요약 실행 (별도 스레드)"""
        try:
            self.log_message("=" * 50)
            summary_type_name = "회의록 형태" if summary_type == "meeting" else "일반 요약"
            self.log_message(f"텍스트 요약을 시작합니다... (형태: {summary_type_name})")
            
            # 파일 경로 결정 (매개변수로 받은 경로가 있으면 그것을 사용, 없으면 기존 방식)
            file_to_summarize = text_file_path if text_file_path else self.last_output_text
            
            if not file_to_summarize or not os.path.exists(file_to_summarize):
                self.root.after(0, lambda: messagebox.showerror("오류", "요약할 텍스트 파일을 찾을 수 없습니다."))
                return
            
            self.log_message(f"요약 대상 파일: {file_to_summarize}")
            
            # 텍스트 파일 읽기
            with open(file_to_summarize, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if len(text_content.strip()) == 0:
                self.root.after(0, lambda: messagebox.showerror("오류", "텍스트 파일이 비어있습니다."))
                return
            
            self.log_message(f"텍스트 길이: {len(text_content)} 문자")
            self.log_message(f"사용 LLM: {api_type}")
            self.log_message(f"요약 형태: {summary_type_name}")
            
            # API 호출
            api_key = self.api_keys.get(api_type)
            if not api_key:
                self.root.after(0, lambda: messagebox.showerror("오류", f"{api_type} API 키가 설정되지 않았습니다."))
                return
            
            self.log_message("LLM 요약 중...")
            
            if api_type == "openai":
                summary = self.summarize_with_openai(text_content, api_key, summary_type)
            elif api_type == "anthropic":
                summary = self.summarize_with_anthropic(text_content, api_key, summary_type)
            elif api_type == "gemini":
                summary = self.summarize_with_gemini(text_content, api_key, summary_type)
            else:
                raise Exception("지원하지 않는 API 타입입니다.")
            
            if not summary:
                raise Exception("요약 결과가 비어있습니다.")
            
            # 요약 파일 저장 (선택된 파일을 기준으로)
            base_name = os.path.splitext(file_to_summarize)[0]
            suffix = "_meeting_summary" if summary_type == "meeting" else "_summary"
            summary_file = f"{base_name}{suffix}.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 텍스트 요약 ===\n")
                f.write(f"원본 파일: {os.path.basename(file_to_summarize)}\n")
                f.write(f"요약 모델: {api_type}\n")
                f.write(f"요약 형태: {summary_type_name}\n")
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
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="오디오/동영상 → 텍스트 변환기", 
                               font=("맑은 고딕", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # 왼쪽 프레임 (기본 설정들) - 더 넓게
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        
        # 오른쪽 프레임 (요약 및 로그) - 더 좁게
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 파일 선택 영역
        file_frame = ttk.LabelFrame(left_frame, text="1. 파일 선택", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        # 드래그 앤 드롭 영역
        self.drop_label = ttk.Label(file_frame, 
                                   text="파일을 여기로 드래그하거나\n아래 버튼을 클릭하세요\n\n지원 형식:\n동영상: mkv, mp4, avi, mov,\n wmv, flv, webm\n오디오: mp3, wav, flac, aac,\n ogg, m4a",
                                   justify=tk.LEFT,
                                   relief="groove",
                                   padding="20")
        self.drop_label.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 파일 선택 버튼
        self.select_button = ttk.Button(file_frame, text="파일 선택", command=self.select_file)
        self.select_button.grid(row=1, column=0, pady=(0, 10))
        
        # 선택된 파일 표시
        self.file_label = ttk.Label(file_frame, text="선택된 파일: 없음", foreground="gray", wraplength=400)
        self.file_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 출력 폴더 선택 영역
        output_frame = ttk.LabelFrame(left_frame, text="2. 저장 위치 선택", padding="10")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        self.output_button = ttk.Button(output_frame, text="저장 폴더 선택", command=self.select_output_dir)
        self.output_button.grid(row=0, column=0, pady=(0, 10))
        
        self.output_label = ttk.Label(output_frame, text="저장 위치: 기본값 (입력 파일과 같은 폴더)", foreground="gray", wraplength=400)
        self.output_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 옵션 영역
        options_frame = ttk.LabelFrame(left_frame, text="3. 변환 옵션", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
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
        
        # 화자 분리 옵션
        speaker_frame = ttk.Frame(options_frame)
        speaker_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.speaker_cb = ttk.Checkbutton(speaker_frame, text="화자 분리 (말하는 사람 구분)", 
                                         variable=self.enable_speaker_diarization)
        self.speaker_cb.grid(row=0, column=0, sticky=tk.W)
        
        speaker_info = ttk.Label(speaker_frame, 
                               text="여러 사람이 대화하는 경우 화자를 \n구분하여 표시합니다 (처리 시간 증가)",
                               font=("맑은 고딕", 8), foreground="gray")
        speaker_info.grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        
        # 변환 버튼
        self.convert_button = ttk.Button(left_frame, text="변환 시작", 
                                        command=self.start_conversion, 
                                        style="Accent.TButton")
        self.convert_button.grid(row=3, column=0, pady=15)
        
        # 진행률 표시 (왼쪽에 배치)
        progress_frame = ttk.Frame(left_frame)
        progress_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # 진행률 퍼센트 표시
        self.progress_label = ttk.Label(progress_frame, text="0%", font=("맑은 고딕", 9))
        self.progress_label.grid(row=0, column=1, sticky=tk.E)
        
        self.status_label = ttk.Label(progress_frame, text="대기 중...")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # 요약 기능 영역 (오른쪽에 배치)
        summary_frame = ttk.LabelFrame(right_frame, text="4. LLM 텍스트 요약", padding="10")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        
        summary_desc = ttk.Label(summary_frame, 
                                text="변환된 텍스트를 AI로 요약할 수 있습니다.\n• Gemini: 일 50회 무료 / OpenAI, Anthropic: 유료",
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
        
        # 로그 영역 (오른쪽에 배치)
        log_frame = ttk.LabelFrame(right_frame, text="변환 로그", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 8))
        
        # 스크롤바가 있는 텍스트 영역 (가로/세로 스크롤 모두 지원)
        text_frame = ttk.Frame(log_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(text_frame, height=15, wrap=tk.NONE)  # wrap=NONE으로 가로 스크롤 활성화
        
        # 세로 스크롤바
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=v_scrollbar.set)
        
        # 가로 스크롤바
        h_scrollbar = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.configure(xscrollcommand=h_scrollbar.set)
        
        # 텍스트 위젯과 스크롤바 배치
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # 왼쪽 프레임을 넓게 (이 시스템에서는 작은 값이 넓어짐)
        main_frame.columnconfigure(1, weight=5)  # 오른쪽 프레임을 좁게 (이 시스템에서는 큰 값이 좁아짐)
        main_frame.rowconfigure(1, weight=1)
        
        # 왼쪽/오른쪽 프레임 설정
        left_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)  # 로그 영역이 확장되도록
        
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
        # 파일에서 직접 읽어서 확인
        config_file = os.path.join(os.path.dirname(__file__), "api_keys.json")
        current_keys = {}
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    current_keys = json.load(f)
                print(f"파일에서 읽은 키: {list(current_keys.keys())}")
            else:
                print("API 키 파일 없음")
        except Exception as e:
            print(f"API 키 파일 읽기 오류: {e}")
        
        # 메모리와 파일 동기화
        self.api_keys = current_keys
        
        # 실제로 값이 있는 키만 확인
        configured_apis = []
        if current_keys.get("openai") and current_keys["openai"].strip():
            configured_apis.append("OpenAI")
        if current_keys.get("anthropic") and current_keys["anthropic"].strip():
            configured_apis.append("Anthropic")
        if current_keys.get("gemini") and current_keys["gemini"].strip():
            configured_apis.append("Gemini")
        
        print(f"API 상태 업데이트: {configured_apis}")  # 디버깅용
        
        if configured_apis:
            status_text = f"API 키: {', '.join(configured_apis)} 설정됨"
            color = "green"
        else:
            status_text = "API 키: 미설정"
            color = "orange"
        
        # 파일 상태도 표시
        if os.path.exists(config_file):
            file_size = os.path.getsize(config_file)
            status_text += f" (파일: {file_size}B)"
        
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
            ("모든 지원 파일", "*.mkv;*.mp4;*.avi;*.mov;*.wmv;*.flv;*.webm;*.mp3;*.wav;*.flac;*.aac;*.ogg;*.m4a"),
            ("동영상 파일", "*.mkv;*.mp4;*.avi;*.mov;*.wmv;*.flv;*.webm"),
            ("오디오 파일", "*.mp3;*.wav;*.flac;*.aac;*.ogg;*.m4a"),
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
            
            # 파일명이 긴 경우를 위한 줄바꿈 처리
            self.file_label.config(text=f"선택된 파일: {filename} ({size_mb:.1f} MB)", foreground="black")
            
            # 드롭 라벨도 긴 파일명에 대해 줄바꿈 처리
            # 파일명이 30자 이상이면 줄바꿈 추가
            if len(filename) > 30:
                # 적절한 위치에서 줄바꿈
                mid_point = len(filename) // 2
                # 가장 가까운 공백이나 특수문자에서 줄바꿈
                break_chars = [' ', '_', '-', '.']
                break_point = mid_point
                for i in range(mid_point, min(len(filename), mid_point + 15)):
                    if filename[i] in break_chars:
                        break_point = i
                        break
                
                if break_point < len(filename) - 5:  # 너무 끝에서 자르지 않기
                    display_name = filename[:break_point] + '\n' + filename[break_point:]
                else:
                    display_name = filename
            else:
                display_name = filename
                
            self.drop_label.config(text=f"✓ {display_name}\n({size_mb:.1f} MB)")
    
    def select_output_dir(self):
        """출력 폴더 선택"""
        folder = filedialog.askdirectory(title="저장 폴더를 선택하세요")
        if folder:
            self.output_dir = folder
            # 긴 경로의 경우 wraplength로 자동 줄바꿈 처리됨
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
                # librosa를 사용하여 m4a 등 다양한 형식 지원
                data, samplerate = librosa.load(audio_file, sr=None)
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
            
            # 화자 분리 처리 (선택사항)
            speakers = None
            if self.enable_speaker_diarization.get():
                self.update_status("화자 분리 분석 중...", 82)
                try:
                    speakers = self.simple_speaker_diarization(audio_file, result['segments'])
                    unique_speakers = len(set(speakers)) if speakers else 1
                    self.log_message(f"화자 분리 완료: {unique_speakers}명의 화자 감지됨")
                except Exception as e:
                    self.log_message(f"화자 분리 실패: {e}")
                    speakers = None
            
            # 6. 결과 저장 (80% -> 95%)
            self.update_status("결과 파일 저장 중...", 85)
            self.save_results(result, output_text, output_subtitle, speakers)
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
            speaker_info = ""
            if speakers:
                unique_speakers = len(set(speakers))
                speaker_info = f"\n• 화자 수: {unique_speakers}명"
                
            files_info = f"텍스트 파일: {os.path.basename(output_text)}\n자막 파일: {os.path.basename(output_subtitle)}"
            if speakers:
                base_name = os.path.splitext(os.path.basename(output_text))[0]
                files_info += f"\n화자별 대화: {base_name}_speakers.txt"
            
            self.root.after(0, lambda: messagebox.showinfo("완료", 
                f"변환이 완료되었습니다!\n\n"
                f"{files_info}\n\n"
                f"처리 통계:\n"
                f"• 오디오 길이: {self.audio_duration:.1f}초\n"
                f"• 변환 시간: {actual_time:.1f}초\n"
                f"• 처리 속도: {processing_speed:.2f}x{speaker_info}\n\n"
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
    
    def save_results(self, result, output_text, output_subtitle, speakers=None):
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
        
        # 텍스트 파일 저장 (화자 정보 포함)
        with open(output_text, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments']):
                timestamp = format_timestamp_ms(segment['start'])
                text = segment['text'].strip()
                
                if speakers and i < len(speakers):
                    # 화자 정보 포함
                    f.write(f"{timestamp} [{speakers[i]}] {text}\n")
                else:
                    # 기본 형식
                    f.write(f"{timestamp} {text}\n")
        
        # SRT 자막 파일 저장 (화자 정보 포함)
        with open(output_subtitle, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start_time = format_srt_timestamp(segment['start'])
                end_time = format_srt_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                
                if speakers and (i-1) < len(speakers):
                    # 화자 정보 포함
                    f.write(f"[{speakers[i-1]}] {text}\n\n")
                else:
                    # 기본 형식
                    f.write(f"{text}\n\n")
        
        # 화자별 대화 파일 생성 (화자 분리가 활성화된 경우)
        if speakers:
            base_name = os.path.splitext(output_text)[0]
            speaker_file = f"{base_name}_speakers.txt"
            
            with open(speaker_file, 'w', encoding='utf-8') as f:
                f.write("=== 화자별 대화 내용 ===\n\n")
                
                # 화자별로 그룹화
                speaker_groups = {}
                for i, segment in enumerate(result['segments']):
                    if i < len(speakers):
                        speaker = speakers[i]
                        if speaker not in speaker_groups:
                            speaker_groups[speaker] = []
                        
                        timestamp = format_timestamp_ms(segment['start'])
                        speaker_groups[speaker].append({
                            'time': timestamp,
                            'text': segment['text'].strip()
                        })
                
                # 화자별로 출력
                for speaker, lines in speaker_groups.items():
                    f.write(f"■ {speaker} ({len(lines)}개 발화)\n")
                    f.write("-" * 50 + "\n")
                    for line in lines:
                        f.write(f"{line['time']} {line['text']}\n")
                    f.write("\n")
            
            self.log_message(f"화자별 대화 파일 저장: {speaker_file}")
        
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
