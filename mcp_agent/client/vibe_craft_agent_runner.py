__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import subprocess
import asyncio
import logging
import shutil
import os
from typing import Dict, Any, Union

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
            result = subprocess.run(command, capture_output=True, text=True, check=True)
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
    ):
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

        yield {"type": "info", "message": "프로세스 시작 중..."}

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 실시간 출력 읽기
            async def read_stream(stream, stream_type):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode().strip()
                    if text:
                        yield {"type": stream_type, "message": text}

            # stdout과 stderr 병합 처리
            async for output in self._merge_streams(
                    read_stream(process.stdout, "stdout"),
                    read_stream(process.stderr, "stderr")
            ):
                yield output

            return_code = await process.wait()

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
            yield {"type": "error", "message": str(e)}

    async def _merge_streams(self, *streams):
        """여러 스트림을 병합하여 순차 처리"""
        queue = asyncio.Queue()

        async def consume(stream):
            try:
                async for item in stream:
                    await queue.put(item)
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(consume(stream)) for stream in streams]
        finished = 0

        while finished < len(streams):
            item = await queue.get()
            if item is None:
                finished += 1
            else:
                yield item

        await asyncio.gather(*tasks, return_exceptions=True)

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
            # shutil.which()를 사용하여 PATH에서 명령어 검색
            command_path = shutil.which(self.agent_command)
            if command_path is None:
                self.logger.warning(f"'{self.agent_command}' 명령어를 PATH에서 찾을 수 없습니다.")
                return False

            # --help 옵션으로 명령어 실행 테스트
            # 참고: 일부 Node.js CLI는 --help에서도 exit code 1을 반환할 수 있음
            result = subprocess.run(
                [self.agent_command, "--help"],
                capture_output=True,
                timeout=10,
                text=True
            )

            # help 텍스트가 출력되었는지 확인 (exit code와 무관하게)
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
            # npm 전역 설치인지 확인
            if "npm" in command_path or "node_modules" in command_path:
                info["installation_method"] = "npm_global"
            elif command_path.startswith("./") or command_path.startswith("/"):
                info["installation_method"] = "local_binary"

        return info


# 사용 예시
if __name__ == "__main__":
    # 기본 인스턴스 생성 (npm 전역 설치 가정, .env 자동 로딩)
    runner = VibeCraftAgentRunner()

    # 설치 정보 확인 (API 키 상태 포함)
    install_info = runner.get_installation_info()
    print(f"설치 정보: {install_info}")

    # GEMINI_API_KEY 단독 확인
    api_key_status = runner.check_gemini_api_key()
    print(f"API 키 상태: {api_key_status}")

    # 사용 가능 여부 확인
    if runner.is_available():
        print("vibecraft-agent 사용 가능")

        # API 키가 없는 경우 경고 출력
        if not api_key_status["exists"]:
            print(f"⚠️  경고: {api_key_status['message']}")
            print(f"💡 해결 방법: {api_key_status['recommendation']}")
            print("API 키 없이 실행하려면 skip_api_key_check=True로 설정하세요.")

        # Enum을 사용한 실행
        result = runner.run_agent(
            sqlite_path="./data-store/383ba7f8-9101-4d20-a3d7-6117a8b54e6c/383ba7f8-9101-4d20-a3d7-6117a8b54e6c.sqlite",
            visualization_type=VisualizationType.TIME_SERIES,
            user_prompt="월별 매출 추이를 보여주는 대시보드",
            output_dir="./output/test",
            project_name="test-dashboard",
            model="flash",
            debug=True
        )

        if result["success"]:
            print("✅ 성공!")
            print(f"출력 디렉토리: {result['output_dir']}")
            print(f"시각화 타입: {result['visualization_type']}")
        else:
            print("❌ 실패!")
            print(f"오류 메시지: {result['message']}")
            if "error_details" in result:
                print(f"상세 오류: {result['error_details']}")
            if "stderr" in result:
                print(f"에러 출력: {result['stderr']}")

        # API 키 체크를 건너뛰는 실행 예시
        result_skip_check = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type="kpi-dashboard",
            user_prompt="KPI 대시보드",
            output_dir="./output",
            skip_api_key_check=True  # API 키 체크 건너뛰기
        )

        # 개발 예정 타입 테스트
        result3 = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type=VisualizationType.GEO_SPATIAL,
            user_prompt="지역별 분석",
            output_dir="./output"
        )
        print(f"개발 예정 타입 결과: {result3['message']}")

    else:
        print("vibecraft-agent 명령어를 찾을 수 없습니다.")
        print("다음 명령어로 설치해주세요: npm install -g vibecraft-agent")

    # 로컬 개발 환경에서 사용할 경우의 예시
    print("\n--- 로컬 개발 환경 예시 ---")
    local_runner = VibeCraftAgentRunner("./vibecraft-agent/vibecraft-agent")
    local_info = local_runner.get_installation_info()
    print(f"로컬 설치 정보: {local_info}")

    # 비동기 실행 예시
    print("\n--- 비동기 실행 예시 ---")


    async def async_example():
        async for output in runner.run_agent_async(
                sqlite_path="/path/to/data.sqlite",
                visualization_type=VisualizationType.TIME_SERIES,
                user_prompt="비동기 테스트",
                output_dir="./output"
        ):
            print(f"[{output['type']}] {output['message']}")

    # asyncio.run(async_example())  # 주석 해제하여 실행
