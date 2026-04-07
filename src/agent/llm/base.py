from langchain.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

from src.config import settings
from src.utils.logger import logger


LLM_MAPPING = {
    "dashscope/tongyi-xiaomi-analysis-pro": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": settings.dashscope_llm_api_key,
    },
    "minimax/MiniMax-M2.7": {
        "api_key": settings.minimax_llm_api_key,
    },
}


class LLMManager:
    def __init__(self):
        self.default_model_id = settings.default_model_id
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.current_model: BaseChatModel = self._init_llm()
        logger.info(f"LLMManager initialized with model: {self.default_model_id}")

    def _init_llm(self):
        return self.switch_model(model_id=self.default_model_id)

    def switch_model(self, model_id: str) -> None:
        config = LLM_MAPPING.get(model_id)
        if not config:
            raise ValueError(
                f"不支持的大模型: {model_id}, 可选: {list(LLM_MAPPING.keys())}"
            )

        logger.info(f"Switching LLM to: {model_id}")
        return ChatLiteLLM(
            model=model_id,
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            temperature=float(self.temperature),
            max_tokens=int(self.max_tokens),
        )
