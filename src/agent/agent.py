from typing import Optional

from deepagents import create_deep_agent

from src.agent.skills.base import SkillManager
from src.agent.llm.base import LLMManager
from src.agent.tools.manager import ToolManager
from src.agent.memory.memory_manager import MemoryManager
from src.utils.logger import logger
from src.utils.decorator import retry


class Agent:
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        tool_manager: Optional[ToolManager] = None,
        skill_manager: Optional[SkillManager] = None,
        system_prompt: Optional[str] = "",
    ):
        self.llm_manager = llm_manager or LLMManager()
        self.tool_manager = tool_manager or ToolManager()
        self.skill_manager = skill_manager or SkillManager()
        self.llm = self.llm_manager.current_model
        self.tools = None
        self.skills = None
        self.system_prompt = (
            system_prompt
            or "你是一位优秀的AI助手，能够根据用户输入，进行各种操作，完成任务"
        )
        self.agent = None
        logger.info("Agent initialized")

    async def create_agent(self) -> None:
        logger.info("Creating agent...")
        self.tools = await self.tool_manager.get_all_tools()
        
        logger.info(f"Loaded {len(self.tools)} tools")
        self.skills = await self.skill_manager.get_all_skills()
        
        logger.info(f"Loaded {len(self.skills)} skills")
        self.agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,
            skills=self.skills,
            system_prompt=self.system_prompt,
        )
        
        logger.info("Agent created successfully")
        return self.agent

    def upload_file(self, file_path: str) -> None:
        logger.info(f"Uploading file: {file_path}")
        self.tool_manager.upload_file(file_path)

    def switch_model(self, model_id: str) -> None:
        logger.info(f"Switching model to: {model_id}")
        self.llm_manager.switch_model(model_id)
        self.llm = self.llm_manager.current_model

    @retry(max_attempts=3, exceptions=(Exception,))
    async def run(self, user_input: str) -> str:
        logger.info(f"用户输入: {user_input}")

        if not self.agent:
            await self.create_agent()

        inputs = {"messages": [{"role": "user", "content": user_input}]}

        try:
            result = await self.agent.ainvoke(inputs)
            logger.info("Agent execution completed")

            return result
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
