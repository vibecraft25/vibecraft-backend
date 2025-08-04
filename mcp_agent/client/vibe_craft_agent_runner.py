__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import subprocess
from typing import Dict, Any, Union
import logging

# Custom imports
from mcp_agent.schemas import VisualizationType


class VibecraftAgentRunner:
    """Vibecraft Agent CLI 실행 클래스"""

    def __init__(self, agent_command: str = "vibecraft-agent"):
        """
        Args:
            agent_command (str): vibecraft-agent 명령어 경로 (기본: "vibecraft-agent")
        """
        self.agent_command = agent_command
        self.logger = logging.getLogger(__name__)

    # TODO: WIP
    def run_agent(
            self,
            sqlite_path: str,
            visualization_type: Union[str, VisualizationType],
            user_prompt: str,
            output_dir: str = "./output",
            debug: bool = False
    ) -> Dict[str, Any]:
        """
        vibecraft-agent CLI 명령어를 실행합니다.

        Args:
            sqlite_path (str): SQLite 데이터베이스 파일 경로
            visualization_type (Union[str, VisualizationType]): 시각화 타입
            user_prompt (str): 사용자의 시각화 요청
            output_dir (str): 출력 디렉토리 (기본: ./output)
            debug (bool): 디버그 모드 (기본: False)

        Returns:
            Dict[str, Any]: 실행 결과
        """

        # VisualizationType을 문자열로 변환
        viz_type_str = self._normalize_visualization_type(visualization_type)

        # 구현 여부 확인
        if isinstance(visualization_type, VisualizationType):
            if not visualization_type.is_implemented:
                return {
                    "success": False,
                    "message": f"'{viz_type_str}' 타입은 아직 구현되지 않았습니다. 구현된 타입: {self.get_implemented_visualization_types()}",
                    "status": visualization_type.status.value
                }
        elif isinstance(visualization_type, str):
            if not VisualizationType.is_implemented_template_id(visualization_type):
                try:
                    vt = VisualizationType.from_string(visualization_type)
                    return {
                        "success": False,
                        "message": f"'{visualization_type}' 타입은 아직 구현되지 않았습니다. 구현된 타입: {self.get_implemented_visualization_types()}",
                        "status": vt.status.value
                    }
                except ValueError:
                    return {
                        "success": False,
                        "message": f"지원하지 않는 시각화 타입입니다: {visualization_type}. 사용 가능한 타입: {self.get_available_visualization_types()}"
                    }

        # 명령어 구성
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
                "visualization_type": viz_type_str,
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

    # TODO: WIP
    async def run_agent_async(
            self,
            sqlite_path: str,
            visualization_type: Union[str, VisualizationType],
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

        # VisualizationType을 문자열로 변환
        viz_type_str = self._normalize_visualization_type(visualization_type)

        # 구현 여부 확인 (동기 메서드와 동일한 로직)
        if isinstance(visualization_type, VisualizationType):
            if not visualization_type.is_implemented:
                return {
                    "success": False,
                    "message": f"'{viz_type_str}' 타입은 아직 구현되지 않았습니다. 구현된 타입: {self.get_implemented_visualization_types()}",
                    "status": visualization_type.status.value
                }
        elif isinstance(visualization_type, str):
            if not VisualizationType.is_implemented_template_id(visualization_type):
                try:
                    vt = VisualizationType.from_string(visualization_type)
                    return {
                        "success": False,
                        "message": f"'{visualization_type}' 타입은 아직 구현되지 않았습니다. 구현된 타입: {self.get_implemented_visualization_types()}",
                        "status": vt.status.value
                    }
                except ValueError:
                    return {
                        "success": False,
                        "message": f"지원하지 않는 시각화 타입입니다: {visualization_type}. 사용 가능한 타입: {self.get_available_visualization_types()}"
                    }

        # 명령어 구성
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
                    "visualization_type": viz_type_str,
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

    def _normalize_visualization_type(self, visualization_type: Union[str, VisualizationType]) -> str:
        """VisualizationType을 문자열로 정규화"""
        if isinstance(visualization_type, VisualizationType):
            return visualization_type.value
        elif isinstance(visualization_type, str):
            # 유효성 검사
            if not VisualizationType.is_valid_template_id(visualization_type):
                raise ValueError(f"Invalid visualization type: {visualization_type}")
            return visualization_type
        else:
            raise TypeError(f"visualization_type must be str or VisualizationType, got {type(visualization_type)}")

    def get_available_visualization_types(self) -> list[str]:
        """사용 가능한 시각화 타입 목록 (Enum 기반)"""
        return VisualizationType.get_all_values()

    def get_implemented_visualization_types(self) -> list[str]:
        """구현된 시각화 타입 목록만"""
        return VisualizationType.get_implemented_values()

    def get_planned_visualization_types(self) -> list[str]:
        """개발 예정인 시각화 타입 목록"""
        planned_types = VisualizationType.get_planned_types()
        return [vt.value for vt in planned_types]

    def validate_visualization_type(self, visualization_type: Union[str, VisualizationType]) -> Dict[str, Any]:
        """시각화 타입 검증"""
        try:
            if isinstance(visualization_type, str):
                vt = VisualizationType.from_string(visualization_type)
            else:
                vt = visualization_type

            return {
                "valid": True,
                "implemented": vt.is_implemented,
                "type": vt.value,
                "description": vt.description,
                "status": vt.status.value
            }

        except ValueError:
            return {
                "valid": False,
                "message": f"지원하지 않는 시각화 타입입니다: {visualization_type}",
                "available_types": self.get_available_visualization_types()
            }

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

        print(f"구현된 타입: {runner.get_implemented_visualization_types()}")
        print(f"개발 예정 타입: {runner.get_planned_visualization_types()}")

        # Enum을 사용한 실행
        result = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type=VisualizationType.TIME_SERIES,  # Enum 사용
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
            visualization_type="kpi-dashboard",  # 문자열 사용
            user_prompt="KPI 대시보드",
            output_dir="./output"
        )

        # 개발 예정 타입 테스트
        result3 = runner.run_agent(
            sqlite_path="/path/to/data.sqlite",
            visualization_type=VisualizationType.GEO_SPATIAL,  # 개발 예정 타입
            user_prompt="지역별 분석",
            output_dir="./output"
        )
        print(f"개발 예정 타입 결과: {result3['message']}")

    else:
        print("vibecraft-agent 명령어를 찾을 수 없습니다.")
