from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # agent setting
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    skills_dir: Path = data_dir / "skills"

    agent_max_iterations: int = 10
    agent_early_stopping_method: str = "generate"
    config_file: str = str(project_root / "config.json")

    # LLM setting
    default_model_id: str = "minimax/MiniMax-M2.7"

    llm_temperature: float = 0.7
    llm_max_tokens: Optional[int] = 1000

    dashscope_llm_api_key: str = ""
    minimax_llm_api_key: str = ""

    # rag setting
    rag_vector_store_type: str = "chroma"
    rag_embedding_model: str = "text-embedding-v2"
    rag_top_k: int = 5
    rag_collection_name: str = "knowledge_base"
    rag_persist_directory: str = str(data_dir / "rag-documents")

    # MCP setting
    mcp_default_timeout: int = 30

    # log setting
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @classmethod
    def validate_config(cls) -> list[str]:
        return settings.validate()

    def validate(self) -> list[str]:
        errors = []
        if not self.dashscope_llm_api_key and not self.minimax_llm_api_key:
            errors.append(
                "LLM API key未配置，请设置 DASHSCOPE_LLM_API_KEY 或 MINIMAX_LLM_API_KEY"
            )
        if self.skills_dir and not self.skills_dir.exists():
            errors.append(f"Skills目录不存在或未配置: {self.skills_dir}")
        return errors


settings = Settings()
