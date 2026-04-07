import pytest
from unittest.mock import Mock, patch
from src.agent.agent import Agent
from src.agent.llm.base import LLMManager
from src.agent.tools.manager import ToolManager
from src.agent.skills.base import SkillManager


class TestAgent:
    def test_agent_initialization(self):
        agent = Agent()
        assert agent.llm_manager is not None
        assert agent.tool_manager is not None
        assert agent.skill_manager is not None
        assert agent.tools is None
        assert agent.skills is None
        assert agent.agent is None

    def test_agent_initialization_with_custom_managers(self):
        llm = Mock()
        llm.current_model = Mock()
        tools = Mock(spec=ToolManager)
        skills = Mock(spec=SkillManager)

        agent = Agent(
            llm_manager=llm,
            tool_manager=tools,
            skill_manager=skills,
        )

        assert agent.llm_manager is llm
        assert agent.tool_manager is tools
        assert agent.skill_manager is skills

    def test_agent_with_custom_system_prompt(self):
        custom_prompt = "Custom system prompt"
        agent = Agent(system_prompt=custom_prompt)
        assert agent.system_prompt == custom_prompt

    def test_agent_default_system_prompt(self):
        agent = Agent()
        assert (
            agent.system_prompt
            == "你是一位优秀的AI助手，能够根据用户输入，进行各种操作，完成任务"
        )

    def test_switch_model(self):
        agent = Agent()
        with patch.object(agent.llm_manager, "switch_model") as mock_switch:
            agent.switch_model("dashscope/tongyi-xiaomi-analysis-pro")
            mock_switch.assert_called_once_with("dashscope/tongyi-xiaomi-analysis-pro")


class TestLLMManager:
    def test_llm_manager_initialization(self):
        manager = LLMManager()
        assert manager.default_model_id is not None
        assert manager.temperature is not None
        assert manager.current_model is not None

    def test_switch_model_valid(self):
        manager = LLMManager()
        with patch("src.agent.llm.base.ChatLiteLLM"):
            result = manager.switch_model("minimax/MiniMax-M2.7")
            assert result is not None

    def test_switch_model_invalid(self):
        manager = LLMManager()
        with pytest.raises(ValueError) as exc_info:
            manager.switch_model("invalid-model")
        assert "不支持的大模型" in str(exc_info.value)


class TestToolManager:
    def test_tool_manager_initialization(self):
        manager = ToolManager()
        assert manager.mcp_manager is not None
        assert manager.rag_manager is not None
        assert manager.function_manager is not None
        assert manager.tools == []

    def test_get_tool_by_name_not_found(self):
        manager = ToolManager()
        result = manager.get_tool_by_name("nonexistent_tool")
        assert result is None

    def test_add_tool(self):
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        result = manager.add_tool(mock_tool)
        assert result is True
        assert mock_tool in manager.tools

    def test_add_duplicate_tool(self):
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        manager.add_tool(mock_tool)
        result = manager.add_tool(mock_tool)
        assert result is False

    def test_remove_tool(self):
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        manager.tools.append(mock_tool)
        result = manager.remove_tool(mock_tool)
        assert result is True
        assert mock_tool not in manager.tools

    def test_remove_nonexistent_tool(self):
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.name = "nonexistent"
        result = manager.remove_tool(mock_tool)
        assert result is False


class TestSkillManager:
    def test_skill_manager_initialization(self):
        manager = SkillManager()
        assert manager.skills_dir is not None

    @pytest.mark.asyncio
    async def test_get_all_skills_returns_list(self):
        manager = SkillManager()
        skills = await manager.get_all_skills()
        assert isinstance(skills, list)
