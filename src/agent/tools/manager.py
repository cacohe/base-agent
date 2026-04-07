from typing import List, Optional

from langchain_core.tools.base import BaseTool

from src.agent.tools.mcp.base import MCPManager
from src.agent.tools.rag.manager import RAGManager
from src.agent.tools.function_call.manager import FunctionCallManager
from src.utils.logger import logger


class ToolManager:
    def __init__(self):
        self.mcp_manager = MCPManager()
        self.rag_manager = RAGManager()
        self.function_manager = FunctionCallManager()
        self.tools: List[BaseTool] = []

    async def get_all_tools(self) -> List[BaseTool]:
        logger.info("Loading all tools...")
        mcp_tools = await self.mcp_manager.get_all_mcp_tools()
        rag_tools = self.rag_manager.get_all_rag_tools()
        function_tools = self.function_manager.get_all_func_call_tools()

        self.tools = function_tools + rag_tools + mcp_tools
        logger.info(f"Loaded {len(self.tools)} tools: {[t.name for t in self.tools]}")
        return self.tools

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        logger.warning(f"Tool not found: {name}")
        return None

    def add_tool(self, tool: BaseTool) -> bool:
        if tool in self.tools:
            logger.warning(f"Tool already exists: {tool.name}")
            return False
        self.tools.append(tool)
        logger.info(f"Tool added: {tool.name}")
        return True

    def remove_tool(self, tool: BaseTool) -> bool:
        if tool not in self.tools:
            logger.warning(f"Tool not found for removal: {tool.name}")
            return False
        self.tools.remove(tool)
        logger.info(f"Tool removed: {tool.name}")
        return True

    def upload_file(self, file_path: str) -> None:
        logger.info(f"Uploading file to RAG: {file_path}")
        self.rag_manager.load_file(file_path)
