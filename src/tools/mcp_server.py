from typing import Any, Dict, List
from src.tools.api_wrapper import fetch_patient_live_data
from src.tools.database import query_sql_db, query_vector_db

class MCPServer:
    """
    Exposes internal tools as an MCP-compatible interface (conceptual).
    """
    def __init__(self):
        self.tools = {
            "fetch_patient_data": fetch_patient_live_data,
            "query_sql": query_sql_db,
            "query_vector": query_vector_db
        }

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        func = self.tools[tool_name]
        return await func(**kwargs)

# Singleton instance
mcp_server = MCPServer()
