import json
import os
from dataclasses import dataclass
from typing import Dict

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import settings
from src.utils.logger import logger


@dataclass
class MCPServerType:
    local = "local"
    remote = "remote"


class MCPManager:
    def __init__(self):
        config_path = settings.config_file
        if not os.path.exists(config_path):
            logger.warning(
                f"MCP config file not found: {config_path}, skipping MCP tools"
            )
            self.mcp_servers = {}
        else:
            self.mcp_servers = self.load_servers_from_json(config_path)
        if not self.mcp_servers:
            logger.info("No MCP servers configured")
            self.mcp_clients = None
        else:
            self.mcp_clients = self.get_mcp_clients()

    def _load_local_server(self, server_name, config: Dict):
        command = config.get("command")
        if not command:
            logger.warning(f"Local server '{server_name}' missing 'command'")
        local_server = {
            "transport": "stdio",
            "command": config.get("command"),
            "env": config.get("env"),
        }
        return local_server

    def _load_remote_server(self, server_name, config: Dict):
        url = config.get("url")
        if not url:
            logger.warning(f"Remote server '{server_name}' missing 'url'")
        remote_server = {
            "transport": "http",
            "url": config.get("url"),
        }
        return remote_server

    def load_servers_from_json(self, file_path: str) -> Dict[str, Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        servers = {}
        mcp_config = data.get("mcp", {})
        for name, config in mcp_config.items():
            server_type = config.get("type")

            if server_type == MCPServerType.local:
                servers[name] = self._load_local_server(name, config)

            elif server_type == MCPServerType.remote:
                servers[name] = self._load_remote_server(name, config)

            else:
                logger.warning(
                    f"Unknown server type '{server_type}' for server '{name}'"
                )

        return servers

    def get_mcp_clients(self) -> MultiServerMCPClient:
        clients = MultiServerMCPClient(self.mcp_servers)
        return clients

    async def get_all_mcp_tools(self):
        if self.mcp_clients is None:
            logger.debug("No MCP clients available, returning empty list")
            return []
        tools = await self.mcp_clients.get_tools()
        return tools
