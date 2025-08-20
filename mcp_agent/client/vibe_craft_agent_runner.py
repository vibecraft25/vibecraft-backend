__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio
import subprocess
import logging
import shutil
import os
from typing import Dict, Any, Union, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
from sse_starlette.sse import ServerSentEvent

# Environment loading
from dotenv import load_dotenv

# Custom imports
from mcp_agent.schemas import VisualizationType
from schemas import SSEEventBuilder


class VibeCraftAgentRunner:
    """npm으로 전역 설치된 VibeCraft Agent CLI 실행 클래스"""

    def __init__(
            self,
            agent_command: str = "vibecraft-agent",
            auto_load_env: bool = True
    ):
        """
        초기화

        Args:
            agent_command: 실행할 명령어 (기본값: "vibecraft-agent")
                          npm 전역 설치 시 "vibecraft-agent"
                          로컬 개발 시 "./vibecraft-agent/vibecraft-agent" 등으로 지정 가능
            auto_load_env: .env 파일 자동 로딩 여부 (기본값: True)
        """
        import platform

        self.agent_command = agent_command
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 플랫폼별 설정
        self.is_windows = platform.system() == "Windows"
        self.use_shell = self.is_windows  # Windows에서만 shell=True 사용
        self.encoding = "utf-8"

        if auto_load_env:
            load_dotenv()

    @staticmethod
    def check_gemini_api_key() -> Dict[str, Any]:
        """GEMINI_API_KEY 환경 변수 확인"""
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return {
                "exists": False,
                "message": "GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.",
                "recommendation": ".env 파일에 GEMINI_API_KEY=your_api_key를 추가해주세요."
            }

        if not api_key.strip():
            return {
                "exists": False,
                "message": "GEMINI_API_KEY가 비어있습니다.",
                "recommendation": ".env 파일에서 올바른 API 키를 설정해주세요."
            }

        # API 키 형식 간단 검증 (일반적으로 Gemini API 키는 특정 패턴을 가짐)
        if len(api_key.strip()) < 10:
            return {
                "exists": False,
                "message": "GEMINI_API_KEY가 너무 짧습니다. 올바른 API 키인지 확인해주세요.",
                "recommendation": "Gemini API 콘솔에서 올바른 API 키를 확인해주세요."
            }

        return {
            "exists": True,
            "message": "GEMINI_API_KEY가 올바르게 설정되었습니다.",
            "key_preview": f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        }

    def run_agent(
            self,
            sqlite_path: str,
            visualization_type: Union[str, VisualizationType],
            user_prompt: str,
            output_dir: str = "./output",
            project_name: str = None,
            model: str = "flash",
            debug: bool = False,
            skip_api_key_check: bool = False,
    ) -> Dict[str, Any]:
        """동기 방식으로 vibecraft-agent를 실행합니다."""

        # GEMINI_API_KEY 확인
        if not skip_api_key_check:
            api_key_status = self.check_gemini_api_key()
            if not api_key_status["exists"]:
                return {
                    "success": False,
                    "message": "API 키 확인 실패",
                    "error_details": api_key_status
                }

        viz_type_str = self._get_type_string(visualization_type)

        if not self._is_implemented_type(visualization_type):
            return {
                "success": False,
                "message": f"'{viz_type_str}' 타입은 아직 구현되지 않았습니다."
            }

        # 출력 디렉토리 확인 및 생성
        if not self._ensure_output_directory(output_dir):
            return {
                "success": False,
                "message": "출력 디렉토리 생성 실패"
            }

        command = [
            self.agent_command,
            "--sqlite-path", sqlite_path,
            "--visualization-type", viz_type_str,
            "--user-prompt", user_prompt,
            "--output-dir", output_dir,
            "--model", model
        ]

        if project_name:
            command.extend(["--project-name", project_name])

        if debug:
            command.append("--debug")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding=self.encoding,
                errors="replace",
                shell=self.use_shell
            )
            return {
                "success": True,
                "message": "실행 완료",
                "output_dir": output_dir,
                "visualization_type": viz_type_str
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "message": f"실행 실패 (exit code: {e.returncode})",
                "stderr": e.stderr
            }

    async def run_agent_async(
            self,
            sqlite_path: str,
            visualization_type: Union[str, VisualizationType],
            user_prompt: str,
            output_dir: str = "./output",
            project_name: str = None,
            model: str = "flash",
            debug: bool = False,
            skip_api_key_check: bool = False,
            require_final_complete: bool = False
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """비동기 방식으로 실행하며 SSE 이벤트를 yield합니다."""

        # GEMINI_API_KEY 확인
        if not skip_api_key_check:
            yield SSEEventBuilder.create_info_event("GEMINI_API_KEY 확인 중...")
            api_key_status = self.check_gemini_api_key()

            if not api_key_status["exists"]:
                yield SSEEventBuilder.create_error_event(
                    f"API 키 확인 실패: {api_key_status['message']}"
                )
                return
            else:
                yield SSEEventBuilder.create_info_event(
                    f"API 키 확인 완료: {api_key_status['key_preview']}"
                )

        viz_type_str = self._get_type_string(visualization_type)
        yield SSEEventBuilder.create_info_event(f"시각화 타입 '{viz_type_str}' 검증 중...")

        if not self._is_implemented_type(visualization_type):
            yield SSEEventBuilder.create_error_event(
                f"'{viz_type_str}' 타입은 아직 구현되지 않았습니다."
            )
            return

        if not self._ensure_output_directory(output_dir):
            yield SSEEventBuilder.create_error_event(
                f"'{output_dir}' directory does not exist."
            )
            return

        yield SSEEventBuilder.create_info_event("검증 완료")

        # 출력 디렉토리 확인 및 생성
        yield SSEEventBuilder.create_info_event(f"출력 디렉토리 확인 중: {output_dir}")
        if not self._ensure_output_directory(output_dir):
            yield SSEEventBuilder.create_error_event("출력 디렉토리 생성 실패")
            return
        yield SSEEventBuilder.create_info_event("출력 디렉토리 준비 완료")

        # 명령어 구성
        command_parts = [
            self.agent_command,
            "--sqlite-path", sqlite_path,
            "--visualization-type", viz_type_str,
            "--user-prompt", user_prompt,
            "--output-dir", output_dir,
            "--model", model
        ]

        if project_name:
            command_parts.extend(["--project-name", project_name])

        if debug:
            command_parts.append("--debug")

        # 명령어를 문자열로 결합 (크로스 플랫폼 고려)
        command = self._build_command_string(command_parts)

        yield SSEEventBuilder.create_info_event("프로세스 시작 중...")

        if debug:
            yield SSEEventBuilder.create_info_event(f"실행 명령어: {command}")

        try:
            process = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._create_process,
                command
            )

            last_message = None

            async for output_line in self._read_process_output(process):
                last_message = output_line
                yield SSEEventBuilder.create_ai_message_chunk(output_line)

            # 프로세스 종료 코드 확인
            return_code = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                process.wait
            )

            if return_code == 0:
                if require_final_complete and (
                        not last_message or "complete" not in last_message.lower()
                ):
                    yield SSEEventBuilder.create_info_event(
                        "실행은 성공했지만 마지막 메시지에 'complete'가 없습니다."
                    )
                else:
                    yield SSEEventBuilder.create_data_event("실행 완료")
                    yield SSEEventBuilder.create_complete_event("execution_complete")
            else:
                yield SSEEventBuilder.create_error_event(
                    f"실행 실패 (exit code: {return_code})"
                )

        except Exception as e:
            yield SSEEventBuilder.create_error_event(
                f"프로세스 실행 중 오류 발생: {str(e)}"
            )

    async def run_agent_stream(
            self,
            sqlite_path: str,
            visualization_type: Union[str, VisualizationType],
            user_prompt: str,
            output_dir: str = "./output",
            project_name: str = None,
            model: str = "flash",
            debug: bool = False,
            skip_api_key_check: bool = False,
            require_final_complete: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """호환성을 위한 기존 딕셔너리 형태 출력 메서드"""

        # GEMINI_API_KEY 확인
        if not skip_api_key_check:
            yield {"type": "info", "message": "GEMINI_API_KEY 확인 중..."}
            api_key_status = self.check_gemini_api_key()

            if not api_key_status["exists"]:
                yield {
                    "type": "error",
                    "message": "API 키 확인 실패",
                    "details": api_key_status
                }
                return
            else:
                yield {
                    "type": "success",
                    "message": f"API 키 확인 완료: {api_key_status['key_preview']}"
                }

        viz_type_str = self._get_type_string(visualization_type)

        yield {"type": "info", "message": f"시각화 타입 '{viz_type_str}' 검증 중..."}

        if not self._is_implemented_type(visualization_type):
            yield {
                "type": "error",
                "message": f"'{viz_type_str}' 타입은 아직 구현되지 않았습니다."
            }
            return

        if not self._ensure_output_directory(output_dir):
            yield {
                "type": "error",
                "message": f"'{output_dir}' directory does not exist."
            }
            return

        yield {"type": "success", "message": "검증 완료"}

        # 명령어 구성
        command_parts = [
            self.agent_command,
            "--sqlite-path", sqlite_path,
            "--visualization-type", viz_type_str,
            "--user-prompt", user_prompt,
            "--output-dir", output_dir,
            "--model", model
        ]

        if project_name:
            command_parts.extend(["--project-name", project_name])

        if debug:
            command_parts.append("--debug")

        # 명령어를 문자열로 결합 (크로스 플랫폼 고려)
        command = self._build_command_string(command_parts)

        yield {"type": "info", "message": "프로세스 시작 중..."}

        if debug:
            yield {"type": "debug", "message": f"실행 명령어: {command}"}

        try:
            process = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._create_process,
                command
            )

            last_message = None

            async for output_line in self._read_process_output(process):
                last_message = output_line
                yield {"type": "output", "message": output_line}

            # 프로세스 종료 코드 확인
            return_code = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                process.wait
            )

            if return_code == 0:
                if require_final_complete and (
                        not last_message or "complete" not in last_message.lower()
                ):
                    yield {
                        "type": "warn",
                        "message": "실행은 성공했지만 마지막 메시지에 'complete'가 없습니다."
                    }
                else:
                    yield {"type": "success", "message": "실행 완료"}
            else:
                yield {
                    "type": "error",
                    "message": f"실행 실패 (exit code: {return_code})"
                }

        except Exception as e:
            yield {"type": "error", "message": f"프로세스 실행 중 오류 발생: {str(e)}"}

    def _build_command_string(self, command_parts: list) -> str:
        """크로스 플랫폼 호환 명령어 문자열 생성

        Args:
            command_parts: 명령어 부분들의 리스트

        Returns:
            str: 플랫폼에 맞는 명령어 문자열
        """
        import platform
        import shlex

        is_windows = platform.system() == "Windows"

        if is_windows:
            # Windows: 공백이 포함된 경우 따옴표로 감싸기
            escaped_parts = []
            for part in command_parts:
                if " " in part or '"' in part:
                    # 이미 따옴표가 있는 경우 이스케이프
                    escaped_part = part.replace('"', '\\"')
                    escaped_parts.append(f'"{escaped_part}"')
                else:
                    escaped_parts.append(part)
            return " ".join(escaped_parts)
        else:
            # Unix 계열: shlex.join 사용 (Python 3.8+) 또는 수동 이스케이프
            try:
                return shlex.join(command_parts)
            except AttributeError:
                # Python 3.7 이하 호환성
                return " ".join(shlex.quote(part) for part in command_parts)

    def _ensure_output_directory(self, output_dir: str) -> bool:
        """출력 디렉토리 존재 확인 및 생성

        Args:
            output_dir: 생성할 디렉토리 경로

        Returns:
            bool: 디렉토리 생성 성공 여부
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"출력 디렉토리 준비 완료: {output_dir}")
            return True
        except PermissionError:
            self.logger.error(f"출력 디렉토리 생성 권한 없음: {output_dir}")
            return False
        except Exception as e:
            self.logger.error(f"출력 디렉토리 생성 실패: {output_dir}, 오류: {str(e)}")
            return False

    def _create_process(self, command: str) -> subprocess.Popen:
        """subprocess.Popen을 사용해서 프로세스를 생성합니다."""
        process_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,  # stderr를 stdout으로 리다이렉트
            "text": True,
            "bufsize": 1,  # 라인 버퍼링
            "universal_newlines": True,
            "encoding": self.encoding,
            "errors": "replace",
            "shell": self.use_shell,
        }

        # 플랫폼별 추가 옵션
        if self.is_windows:
            process_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            process_kwargs["preexec_fn"] = os.setsid

        return subprocess.Popen(command, **process_kwargs)

    async def _read_process_output(self, process: subprocess.Popen) -> AsyncGenerator[str, None]:
        """프로세스의 실시간 출력을 비동기적으로 읽습니다."""
        loop = asyncio.get_event_loop()

        while True:
            try:
                # 블로킹 readline을 비동기로 실행
                line = await loop.run_in_executor(
                    self.executor,
                    process.stdout.readline
                )

                if not line:
                    break

                line = line.strip()
                if line:
                    yield line

            except Exception as e:
                self.logger.error(f"출력 읽기 중 오류: {str(e)}")
                break

        # 프로세스가 여전히 실행 중인지 확인
        if process.poll() is None:
            self.logger.info("프로세스 종료 대기 중...")

    def _get_type_string(self, visualization_type: Union[str, VisualizationType]) -> str:
        """VisualizationType을 문자열로 변환"""
        if isinstance(visualization_type, VisualizationType):
            return visualization_type.value
        return visualization_type

    def _is_implemented_type(self, visualization_type: Union[str, VisualizationType]) -> bool:
        """구현된 타입인지 확인"""
        if isinstance(visualization_type, VisualizationType):
            return visualization_type.is_implemented
        try:
            vt = VisualizationType.from_string(visualization_type)
            return vt.is_implemented
        except ValueError:
            return False

    def is_available(self) -> bool:
        """명령어 사용 가능 여부 확인 (크로스 플랫폼 고려)"""
        try:
            command_path = shutil.which(self.agent_command)
            if command_path is None:
                self.logger.warning(f"'{self.agent_command}' 명령어를 PATH에서 찾을 수 없습니다.")
                return False

            # 플랫폼별 도움말 명령어 실행
            import platform
            is_windows = platform.system() == "Windows"

            # 타임아웃 설정 (크로스 플랫폼)
            timeout = 10

            result = subprocess.run(
                [self.agent_command, "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=is_windows,  # Windows에서만 shell=True
                timeout=timeout
            )

            if "vibecraft-agent" in result.stdout.lower() or "usage:" in result.stdout.lower():
                self.logger.info(f"vibecraft-agent 사용 가능 (경로: {command_path})")
                return True
            elif result.returncode == 0:
                self.logger.info(f"vibecraft-agent 사용 가능 (경로: {command_path})")
                return True
            else:
                self.logger.error(f"명령어 실행 실패: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("명령어 실행 시간 초과")
            return False
        except FileNotFoundError:
            self.logger.error(f"명령어를 찾을 수 없습니다: {self.agent_command}")
            return False
        except Exception as e:
            self.logger.error(f"명령어 확인 중 오류 발생: {e}")
            return False

    def get_installation_info(self) -> Dict[str, Any]:
        """설치 정보 및 상태를 반환합니다."""
        command_path = shutil.which(self.agent_command)

        info = {
            "command": self.agent_command,
            "available": self.is_available(),
            "path": command_path,
            "installation_method": "unknown",
            "platform": "Windows" if self.is_windows else "Unix-like",
            "use_shell": self.use_shell,
            "encoding": self.encoding,
            "gemini_api_key": self.check_gemini_api_key()
        }

        if command_path:
            if "npm" in command_path or "node_modules" in command_path:
                info["installation_method"] = "npm_global"
            elif command_path.startswith("./") or command_path.startswith("/"):
                info["installation_method"] = "local_binary"

        return info

    def __del__(self):
        """소멸자: ThreadPoolExecutor 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# 사용 예시
if __name__ == "__main__":
    import asyncio


    async def sse_example():
        """SSE 이벤트 출력 예시"""
        runner = VibeCraftAgentRunner()

        print("=== SSE 이벤트 출력 예시 ===")
        async for sse_event in runner.run_agent_async(
                sqlite_path="./data-store/test.sqlite",
                visualization_type=VisualizationType.TIME_SERIES,
                user_prompt="월별 매출 추이를 보여주는 대시보드",
                output_dir="./output/test",
                project_name="test-dashboard",
                model="flash",
                debug=True,
                require_final_complete=True
        ):
            print(f"Event: {sse_event.event}, Data: {sse_event.data}")


    async def dict_example():
        """기존 딕셔너리 형태 출력 예시"""
        runner = VibeCraftAgentRunner()

        print("=== 딕셔너리 형태 출력 예시 ===")
        async for output in runner.run_agent_stream(
                sqlite_path="./data-store/test.sqlite",
                visualization_type=VisualizationType.TIME_SERIES,
                user_prompt="월별 매출 추이를 보여주는 대시보드",
                output_dir="./output/test",
                project_name="test-dashboard",
                model="flash",
                debug=True,
                require_final_complete=False
        ):
            print(f"[{output['type']}] {output['message']}")


    # SSE 예시 실행
    asyncio.run(sse_example())
    print("\n" + "=" * 50 + "\n")
    # 딕셔너리 예시 실행
    asyncio.run(dict_example())
