__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from pathlib import Path

# Third-party imports
from pydantic_settings import BaseSettings
import yaml


class Settings(BaseSettings):
    version: str

    base_url: str
    host: str
    port: int

    mcp_path: str

    log_path: str

    @classmethod
    def load_from_yaml(cls, env: str = "development") -> "Settings":
        config_file = Path(__file__).parent / f"config-{env}.yml"

        # Load YAML configuration
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(
            version=config["version"]["server"],
            base_url=config["base_url"],
            host=config["host"],
            port=config["port"],
            mcp_path=config["resource"]["mcp"],
            log_path=config["log"]["path"],
        )


# Load settings from YAML
settings = Settings.load_from_yaml(env="development")
# settings = Settings.load_from_yaml(env="staging")
