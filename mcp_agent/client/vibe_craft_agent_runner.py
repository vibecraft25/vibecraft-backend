__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import subprocess
import asyncio
import logging
import shutil
from typing import Dict, Any, Union

# Custom imports
from mcp_agent.schemas import VisualizationType


class VibeCraftAgentRunner:
    """npm으로 전역 설치된 VibeCraft Agent CLI 실행 클래스"""

    def __init__(self, agent_command: str = "vibecraft-agent"):
        """
        초기화

        Args:
            agent_command: 실행할 명령어 (기본값: "vibecraft-agent")
                          npm 전역 설치 시 "vibecraft-agent"
                          로컬 개발 시 "./vibecraft-agent/vibecraft-agent" 등으로 지정 가능
        """
        self.agent_command = agent_command
        self.logger = logging.getLogger(__name__)

    def run_agent(
            self,
            sqlite_path: str,
            visualization_type: Union[str, VisualizationType],
            user_prompt: str,
            output_dir: str = "./output",
            debug: bool = False
    ) -> Dict[str, Any]:
        """동기 방식으로 vibecraft-agent를 실행합니다."""

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
            "--output-dir", output_dir
        ]

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
            debug: bool = False
    ):
        """비동기 방식으로 실행하며 실시간 출력을 yield합니다."""

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
            "--output-dir", output_dir
        ]

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
            result = subprocess.run(
                [self.agent_command, "--help"],
                capture_output=True,
                timeout=10,
                text=True
            )

            if result.returncode == 0:
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
            "installation_method": "unknown"
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
    # 기본 인스턴스 생성 (npm 전역 설치 가정)
    runner = VibeCraftAgentRunner()

    # 설치 정보 확인
    install_info = runner.get_installation_info()
    print(f"설치 정보: {install_info}")

    # 사용 가능 여부 확인
    if runner.is_available():
        print("vibecraft-agent 사용 가능")

        # Enum을 사용한 실행
        result = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type=VisualizationType.TIME_SERIES,
            user_prompt="월별 매출 추이를 보여주는 대시보드",
            output_dir="./output",
            debug=True
        )

        if result["success"]:
            print("성공!")
            print(f"출력 디렉토리: {result['output_dir']}")
            print(f"시각화 타입: {result['visualization_type']}")
        else:
            print("실패!")
            print(result["message"])

        # 문자열을 사용한 실행 (하위 호환성)
        result2 = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type="kpi-dashboard",
            user_prompt="KPI 대시보드",
            output_dir="./output"
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
