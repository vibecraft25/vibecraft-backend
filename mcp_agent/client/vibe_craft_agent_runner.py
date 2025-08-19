__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio
import subprocess
import logging
import shutil
import os
from typing import Dict, Any, Union, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

# Environment loading
from dotenv import load_dotenv

# Custom imports
from mcp_agent.schemas import VisualizationType


class VibeCraftAgentRunner:
    """npm으로 전역 설치된 VibeCraft Agent CLI 실행 클래스"""

    def __init__(self, agent_command: str = "vibecraft-agent", auto_load_env: bool = True):
        """
        초기화

        Args:
            agent_command: 실행할 명령어 (기본값: "vibecraft-agent")
                          npm 전역 설치 시 "vibecraft-agent"
                          로컬 개발 시 "./vibecraft-agent/vibecraft-agent" 등으로 지정 가능
            auto_load_env: .env 파일 자동 로딩 여부 (기본값: True)
        """
        self.agent_command = agent_command
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=2)

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
            skip_api_key_check: bool = False
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
                encoding="utf-8",
                errors="replace",
                shell=True
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
            skip_api_key_check: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """비동기 방식으로 실행하며 실시간 출력을 yield합니다."""

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

        # 명령어를 문자열로 결합 (shell=True 사용을 위해)
        command = " ".join(f'"{part}"' if " " in part else part for part in command_parts)

        yield {"type": "info", "message": "프로세스 시작 중..."}
        yield {"type": "debug", "message": f"실행 명령어: {command}"}

        try:
            # subprocess.Popen을 사용한 프로세스 시작
            process = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._create_process,
                command
            )

            # 실시간 출력 읽기
            async for output in self._read_process_output(process):
                yield output

            # 프로세스 완료 대기
            return_code = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                process.wait
            )

            if return_code == 0:
                yield {
                    "type": "success",
                    "message": "실행 완료",
                    "output_dir": output_dir,
                    "step": "execution_complete"
                }
            else:
                yield {
                    "type": "error",
                    "message": f"실행 실패 (exit code: {return_code})"
                }

        except Exception as e:
            yield {"type": "error", "message": f"프로세스 실행 중 오류 발생: {str(e)}"}

    def _create_process(self, command: str) -> subprocess.Popen:
        """subprocess.Popen을 사용해서 프로세스를 생성합니다."""
        return subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr를 stdout으로 리다이렉트
            text=True,
            bufsize=1,  # 라인 버퍼링
            universal_newlines=True,
            encoding="utf-8",
            errors="replace"
        )

    async def _read_process_output(self, process: subprocess.Popen) -> AsyncGenerator[Dict[str, Any], None]:
        """프로세스의 실시간 출력을 비동기적으로 읽습니다."""
        loop = asyncio.get_event_loop()

        while True:
            try:
                # 블로킹 readline을 비동기로 실행
                line = await loop.run_in_executor(
                    self.executor,
                    process.stdout.readline
                )

                if not line:  # EOF에 도달하면 종료
                    break

                line = line.strip()
                if line:
                    # 출력 타입을 분류 (선택적)
                    output_type = self._classify_output_type(line)
                    yield {
                        "type": output_type,
                        "message": line
                    }

            except Exception as e:
                yield {
                    "type": "error",
                    "message": f"출력 읽기 중 오류: {str(e)}"
                }
                break

        # 프로세스가 여전히 실행 중인지 확인
        if process.poll() is None:
            yield {"type": "info", "message": "프로세스 종료 대기 중..."}

    def _classify_output_type(self, line: str) -> str:
        """출력 라인의 타입을 분류합니다."""
        line_lower = line.lower()

        if "error" in line_lower or "fail" in line_lower:
            return "error"
        elif "warning" in line_lower or "warn" in line_lower:
            return "warning"
        elif "success" in line_lower or "complete" in line_lower or "done" in line_lower:
            return "success"
        elif "info" in line_lower or "processing" in line_lower:
            return "info"
        elif line.startswith("[") and "]" in line:  # 로그 형태
            return "log"
        else:
            return "stdout"

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
        """명령어 사용 가능 여부 확인 (npm 전역 설치 고려)"""
        try:
            command_path = shutil.which(self.agent_command)
            if command_path is None:
                self.logger.warning(f"'{self.agent_command}' 명령어를 PATH에서 찾을 수 없습니다.")
                return False

            result = subprocess.run(
                [self.agent_command, "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=True
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


    async def async_example():
        runner = VibeCraftAgentRunner()

        print("=== 비동기 실행 예시 ===")
        async for output in runner.run_agent_async(
                sqlite_path="./data-store/test.sqlite",
                visualization_type=VisualizationType.TIME_SERIES,
                user_prompt="월별 매출 추이를 보여주는 대시보드",
                output_dir="./output/test",
                project_name="test-dashboard",
                model="flash",
                debug=True
        ):
            print(f"[{output['type']}] {output['message']}")


    # 실행
    asyncio.run(async_example())
