"""
LLM API 서비스 클래스
"""
import json
import requests
import os
from typing import Optional, Dict


class LLMService:
    def __init__(self):
        self.api_keys = {}
        self.load_api_keys()
    
    def load_api_keys(self):
        """API 키 설정 파일 로드"""
        config_file = "api_keys.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.api_keys = json.load(f)
            except Exception as e:
                print(f"API 키 설정 파일 로드 실패: {e}")
                self.api_keys = {}
        else:
            # 기본 설정 파일 생성
            self.api_keys = {
                "openai": "",
                "anthropic": "", 
                "gemini": ""
            }
            self.save_api_keys()
    
    def save_api_keys(self):
        """API 키 설정 파일 저장"""
        try:
            with open("api_keys.json", 'w', encoding='utf-8') as f:
                json.dump(self.api_keys, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"API 키 설정 파일 저장 실패: {e}")
    
    def update_api_keys(self, keys: Dict[str, str]):
        """API 키 업데이트"""
        self.api_keys.update(keys)
        self.save_api_keys()
    
    def get_summary_prompt(self, summary_type: str) -> str:
        """요약 타입에 따른 프롬프트 반환"""
        if summary_type == "meeting":
            return """다음 텍스트를 회의록 형태로 요약해주세요. 다음 형식을 따라주세요:

# 회의록

## 📅 회의 개요
- 일시: [추정 또는 명시되지 않음]
- 참석자: [화자 정보 기반으로 추정]
- 주제: [내용 기반으로 추정]

## 📋 주요 안건 및 논의사항
1. **[주요 주제 1]**
   - 논의 내용: 
   - 주요 의견:

2. **[주요 주제 2]**
   - 논의 내용:
   - 주요 의견:

## ✅ 결정사항
- [결정된 내용들]

## 📝 액션 아이템
- [ ] [담당자]: [해야 할 일]
- [ ] [담당자]: [해야 할 일]

## 💬 기타사항
- [추가 논의된 내용이나 특이사항]

---
*이 회의록은 AI가 음성 인식 결과를 바탕으로 자동 생성되었습니다.*

텍스트:"""
        else:
            return """다음 텍스트를 핵심 내용 위주로 간결하게 요약해주세요. 
주요 키워드와 중요한 정보를 포함하여 이해하기 쉽게 정리해주세요:

텍스트:"""
    
    def summarize_with_openai(self, text: str, api_key: str, summary_type: str = "general") -> Optional[str]:
        """OpenAI GPT를 사용한 텍스트 요약"""
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = self.get_summary_prompt(summary_type)
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system", 
                        "content": "당신은 텍스트 요약 전문가입니다. 한국어로 명확하고 간결하게 요약해주세요."
                    },
                    {
                        "role": "user", 
                        "content": f"{prompt}\n\n{text}"
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result['choices'][0]['message']['content'].strip()
                return summary
            elif response.status_code == 401:
                raise ValueError("OpenAI API 키가 올바르지 않습니다.")
            elif response.status_code == 429:
                raise ValueError("OpenAI API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            elif response.status_code == 402:
                raise ValueError("OpenAI API 크레딧이 부족합니다. 결제 정보를 확인해주세요.")
            else:
                error_msg = f"OpenAI API 오류 (상태 코드: {response.status_code})"
                try:
                    error_detail = response.json().get('error', {}).get('message', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    pass
                raise ValueError(error_msg)
                
        except requests.exceptions.Timeout:
            raise ValueError("OpenAI API 요청 시간이 초과되었습니다.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OpenAI API 연결 오류: {str(e)}")
        except Exception as e:
            raise ValueError(f"OpenAI 요약 중 오류 발생: {str(e)}")
    
    def summarize_with_anthropic(self, text: str, api_key: str, summary_type: str = "general") -> Optional[str]:
        """Anthropic Claude를 사용한 텍스트 요약"""
        if not api_key or not api_key.strip():
            raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
        
        try:
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            prompt = self.get_summary_prompt(summary_type)
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{text}"
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result['content'][0]['text'].strip()
                return summary
            elif response.status_code == 401:
                raise ValueError("Anthropic API 키가 올바르지 않습니다.")
            elif response.status_code == 429:
                raise ValueError("Anthropic API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            elif response.status_code == 402:
                raise ValueError("Anthropic API 크레딧이 부족합니다. 결제 정보를 확인해주세요.")
            else:
                error_msg = f"Anthropic API 오류 (상태 코드: {response.status_code})"
                try:
                    error_detail = response.json().get('error', {}).get('message', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    pass
                raise ValueError(error_msg)
                
        except requests.exceptions.Timeout:
            raise ValueError("Anthropic API 요청 시간이 초과되었습니다.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Anthropic API 연결 오류: {str(e)}")
        except Exception as e:
            raise ValueError(f"Anthropic 요약 중 오류 발생: {str(e)}")
    
    def summarize_with_gemini(self, text: str, api_key: str, summary_type: str = "general") -> Optional[str]:
        """Google Gemini를 사용한 텍스트 요약"""
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        
        try:
            prompt = self.get_summary_prompt(summary_type)
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"{prompt}\n\n{text}"
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 2000
                }
            }
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        summary = candidate['content']['parts'][0]['text'].strip()
                        return summary
                    else:
                        raise ValueError("Gemini API 응답 형식이 올바르지 않습니다.")
                else:
                    raise ValueError("Gemini API에서 응답을 생성하지 못했습니다.")
            elif response.status_code == 401:
                raise ValueError("Gemini API 키가 올바르지 않습니다.")
            elif response.status_code == 429:
                raise ValueError("Gemini API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            elif response.status_code == 402:
                raise ValueError("Gemini API 크레딧이 부족합니다. 결제 정보를 확인해주세요.")
            else:
                error_msg = f"Gemini API 오류 (상태 코드: {response.status_code})"
                try:
                    error_detail = response.json().get('error', {}).get('message', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    pass
                raise ValueError(error_msg)
                
        except requests.exceptions.Timeout:
            raise ValueError("Gemini API 요청 시간이 초과되었습니다.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Gemini API 연결 오류: {str(e)}")
        except Exception as e:
            raise ValueError(f"Gemini 요약 중 오류 발생: {str(e)}")
    
    def summarize_text(self, text: str, api_type: str, summary_type: str = "general") -> str:
        """텍스트 요약 실행"""
        if api_type == "openai":
            return self.summarize_with_openai(text, self.api_keys.get("openai", ""), summary_type)
        elif api_type == "anthropic":
            return self.summarize_with_anthropic(text, self.api_keys.get("anthropic", ""), summary_type)  
        elif api_type == "gemini":
            return self.summarize_with_gemini(text, self.api_keys.get("gemini", ""), summary_type)
        else:
            raise ValueError(f"지원되지 않는 API 타입: {api_type}")
    
    def test_api_key(self, api_type: str, api_key: str) -> bool:
        """API 키 테스트"""
        test_text = "안녕하세요. 이것은 API 키 테스트입니다."
        
        try:
            if api_type == "openai":
                result = self.summarize_with_openai(test_text, api_key, "general")
            elif api_type == "anthropic":  
                result = self.summarize_with_anthropic(test_text, api_key, "general")
            elif api_type == "gemini":
                result = self.summarize_with_gemini(test_text, api_key, "general")
            else:
                return False
            
            return result is not None and len(result.strip()) > 0
            
        except Exception:
            return False
    
    def get_api_status(self) -> Dict[str, bool]:
        """모든 API 키의 상태 확인"""
        status = {}
        for api_type in ["openai", "anthropic", "gemini"]:
            api_key = self.api_keys.get(api_type, "")
            status[api_type] = bool(api_key and api_key.strip())
        return status
