"""
Local MCP Server for KTern.AI RAID Identification Agent POC.
Exposes standard MCP tools over STDIO:
- select_or_configure_folder
- scan_folder
- read_file
- get_file_status
"""

import sys
import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Redirect console logging handlers to stderr to prevent corrupting stdout JSON-RPC
root_logger = logging.getLogger("raid_agent")
for h in root_logger.handlers:
    if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
        h.stream = sys.stderr

from mcp.server.mcpserver import MCPServer
from src.mcp.tools.local_file_tools import (
    select_or_configure_folder as _tool_select_or_configure_folder,
    scan_folder as _tool_scan_folder,
    read_file as _tool_read_file,
    get_file_status as _tool_get_file_status,
)
from src.utils.logger import get_logger

logger = get_logger("mcp.server")

# Initialize MCP Server instance
server = MCPServer("raid_local_file_server")


@server.tool()
def select_or_configure_folder(folder_path: str) -> Dict[str, Any]:
    """
    Validates and configures the permitted local directory to scan.
    Ensures path exists, is a directory, and does not violate security constraints.
    """
    return _tool_select_or_configure_folder(folder_path)


@server.tool()
def scan_folder(folder_path: Optional[str] = None, user_id: str = "default_user") -> Dict[str, Any]:
    """
    Recursively discovers supported project files under the target directory.
    Categorizes files as DISCOVERED, PROCESSED, MODIFIED, or FAILED.
    Strictly read-only: does not modify the database and does not invoke LLMs.
    """
    return _tool_scan_folder(folder_path=folder_path, user_id=user_id)


@server.tool()
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Safely reads content and metadata of a permitted project file.
    Enforces file size limits (50MB) and path traversal protection.
    Returns UTF-8 text for text files or base64 string for binary documents (.docx, .xlsx).
    """
    return _tool_read_file(file_path)


@server.tool()
def get_file_status(file_path: str, user_id: str = "default_user") -> Dict[str, Any]:
    """
    Retrieves the sync and processing status of a file from SQLite database.
    Returns status (DISCOVERED, PROCESSING, PROCESSED, MODIFIED, FAILED) and metrics.
    """
    return _tool_get_file_status(file_path=file_path, user_id=user_id)


def main():
    """Main entrypoint for running the MCP server process over STDIO."""
    for lgr in (logging.getLogger("raid_agent"), logging.getLogger()):
        for h in lgr.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
                h.stream = sys.stderr
    logger.info("Starting RAID Local MCP Server over STDIO...")
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
