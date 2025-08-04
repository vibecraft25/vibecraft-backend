import subprocess
from typing import Optional, Dict, Any
import logging


class VibecraftAgentRunner:
    """Vibecraft Agent CLI 실행 클래스"""

    def __init__(self, agent_command: str = "vibecraft-agent"):
        """
        Args:
            agent_command (str): vibecraft-agent 명령어 경로 (기본: "vibecraft-agent")
        """
        self.agent_command = agent_command
        self.logger = logging.getLogger(__name__)

    def run_agent(
            self,
            sqlite_path: str,
            visualization_type: str,
            user_prompt: str,
            output_dir: str = "./output",
            debug: bool = False
    ) -> Dict[str, Any]:
        """
        vibecraft-agent CLI 명령어를 실행합니다.

        Args:
            sqlite_path (str): SQLite 데이터베이스 파일 경로
            visualization_type (str): 시각화 타입
            user_prompt (str): 사용자의 시각화 요청
            output_dir (str): 출력 디렉토리 (기본: ./output)
            debug (bool): 디버그 모드 (기본: False)

        Returns:
            Dict[str, Any]: 실행 결과
        """

        # 명령어 구성
        command = [
            self.agent_command,
            "--sqlite-path", sqlite_path,
            "--visualization-type", visualization_type,
            "--user-prompt", user_prompt,
            "--output-dir", output_dir
        ]

        if debug:
            command.append("--debug")

        try:
            self.logger.info(f"vibecraft-agent 실행: {' '.join(command)}")

            # 명령어 실행
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info("vibecraft-agent 실행 완료")

            return {
                "success": True,
                "message": "vibecraft-agent 실행 완료",
                "output_dir": output_dir,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.CalledProcessError as e:
            error_msg = f"vibecraft-agent 실행 실패 (exit code: {e.returncode})"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "stdout": e.stdout,
                "stderr": e.stderr
            }

        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg
            }

    async def run_agent_async(
            self,
            sqlite_path: str,
            visualization_type: str,
            user_prompt: str,
            output_dir: str = "./output",
            debug: bool = False
    ) -> Dict[str, Any]:
        """
        vibecraft-agent CLI 명령어를 비동기로 실행합니다.

        Args:
            동기 메서드와 동일

        Returns:
            Dict[str, Any]: 실행 결과
        """
        import asyncio

        # 명령어 구성
        command = [
            self.agent_command,
            "--sqlite-path", sqlite_path,
            "--visualization-type", visualization_type,
            "--user-prompt", user_prompt,
            "--output-dir", output_dir
        ]

        if debug:
            command.append("--debug")

        try:
            self.logger.info(f"vibecraft-agent 비동기 실행: {' '.join(command)}")

            # 비동기 명령어 실행
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.logger.info("vibecraft-agent 비동기 실행 완료")
                return {
                    "success": True,
                    "message": "vibecraft-agent 실행 완료",
                    "output_dir": output_dir,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                }
            else:
                error_msg = f"vibecraft-agent 실행 실패 (exit code: {process.returncode})"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                }

        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg
            }

    def get_available_visualization_types(self) -> list:
        """사용 가능한 시각화 타입 목록"""
        return [
            "time-series",
            "kpi-dashboard",
            "comparison",
            "distribution",
            "correlation"
        ]

    def is_available(self) -> bool:
        """vibecraft-agent 명령어 사용 가능 여부 확인"""
        try:
            result = subprocess.run(
                [self.agent_command, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False


# 사용 예시
if __name__ == "__main__":
    # 인스턴스 생성
    runner = VibecraftAgentRunner()

    # 사용 가능 여부 확인
    if runner.is_available():
        print("vibecraft-agent 사용 가능")

        # 실행
        result = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type="time-series",
            user_prompt="월별 매출 추이를 보여주는 대시보드",
            output_dir="./output",
            debug=True
        )

        if result["success"]:
            print("성공!")
            print(f"출력 디렉토리: {result['output_dir']}")
        else:
            print("실패!")
            print(result["message"])
    else:
        print("vibecraft-agent 명령어를 찾을 수 없습니다.")
