from typing import List

from langchain.tools import BaseTool
from src.agent.tools.function_call.base import calculator, get_weather


class FunctionCallManager:
    def get_all_func_call_tools(self) -> List[BaseTool]:
        """获取所有工具。"""
        return [
            get_weather,
            calculator,
        ]
