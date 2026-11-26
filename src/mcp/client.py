"""
MCP Client Module for KTern.AI RAID Identification Agent POC.
Connects to the local MCP Server process over STDIO using the official MCP Python SDK.
Maintains a persistent server session across calls to prevent unnecessary subprocess
restarts on Streamlit reruns and preserve session configuration state.
"""

import sys
import json
import asyncio
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import BASE_DIR
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from src.utils.logger import get_logger

logger = get_logger("mcp.client")

# Default timeout for MCP tool calls (seconds)
DEFAULT_MCP_TIMEOUT = 30.0


class MCPClientError(Exception):
    """Base exception for MCP client operations."""
    pass


class MCPConnectionError(MCPClientError):
    """Raised when the MCP server cannot be spawned or fails communication."""
    pass


class MCPToolExecutionError(MCPClientError):
    """Raised when an MCP tool returns an error or execution fails."""
    pass


class MCPClient:
    """
    Persistent client interface that manages connection to the local MCP Server over STDIO.
    Runs a persistent background thread with an active ClientSession, avoiding repetitive
    server process creation on Streamlit reruns and retaining server configuration.
    """

    def __init__(
        self,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        default_timeout: float = DEFAULT_MCP_TIMEOUT
    ):
        self.server_command = server_command or sys.executable
        self.server_args = server_args or ["-m", "src.mcp.server"]
        self.cwd = str((cwd or BASE_DIR).resolve())
        self.default_timeout = default_timeout

        # Persistent connection synchronization primitives
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stack: Optional[AsyncExitStack] = None
        self._close_event: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._init_error: Optional[Exception] = None
        self._is_closed = False

    def _get_server_params(self) -> StdioServerParameters:
        """Constructs StdioServerParameters for launching the local MCP server."""
        return StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            cwd=self.cwd
        )

    def _run_event_loop(self) -> None:
        """Entrypoint for dedicated background thread running the persistent asyncio loop."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_lifecycle())
        except Exception as exc:
            logger.error("MCP background event loop error: %s", exc)
        finally:
            try:
                # Cancel pending tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()
            except Exception:
                pass

    async def _connect_lifecycle(self) -> None:
        """Runs the long-lived MCP client connection lifecycle until closed."""
        self._stack = AsyncExitStack()
        try:
            params = self._get_server_params()
            read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._session = session
            self._init_error = None
            self._close_event = asyncio.Event()
            self._ready_event.set()

            # Keep connection alive until closed
            await self._close_event.wait()

        except Exception as exc:
            logger.error("Failed to connect persistent MCP session: %s", exc)
            self._init_error = exc
            self._session = None
            self._ready_event.set()
        finally:
            if self._stack:
                try:
                    await self._stack.aclose()
                except Exception:
                    pass
            self._session = None

    def _ensure_connected(self, timeout: Optional[float] = None) -> None:
        """Ensures the background MCP server process and session are active and healthy."""
        with self._lock:
            if self._is_closed:
                self._is_closed = False

            if self._session is not None and self._loop is not None and self._loop.is_running():
                return

            # Clean up old thread if any
            self._cleanup_session_internal()

            # Start fresh background thread and loop
            self._ready_event.clear()
            self._init_error = None
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_event_loop,
                name="MCPClientWorker",
                daemon=True
            )
            self._thread.start()

            effective_timeout = timeout or self.default_timeout
            if not self._ready_event.wait(timeout=effective_timeout):
                self._cleanup_session_internal()
                raise MCPConnectionError(f"Timed out after {effective_timeout}s connecting to MCP server.")

            if self._init_error is not None:
                err = self._init_error
                self._cleanup_session_internal()
                raise MCPConnectionError(f"Could not connect to MCP server: {err}") from err

            if self._session is None:
                self._cleanup_session_internal()
                raise MCPConnectionError("MCP server session could not be established.")

    def _cleanup_session_internal(self) -> None:
        """Internal cleanup without locking."""
        if self._loop is not None and self._loop.is_running() and self._close_event is not None:
            try:
                self._loop.call_soon_threadsafe(self._close_event.set)
            except Exception:
                pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None
        self._loop = None
        self._session = None
        self._stack = None
        self._close_event = None

    # -----------------------------------------------------------------------
    # Core MCP Protocol Operations
    # -----------------------------------------------------------------------

    def list_tools(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Synchronously queries the persistent MCP Server for available registered tools.
        """
        effective_timeout = timeout or self.default_timeout
        self._ensure_connected(timeout=effective_timeout)

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._session.list_tools(),
                self._loop
            )
            response = future.result(timeout=effective_timeout)
            tools = []
            for tool in response.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else None
                })
            return tools
        except Exception as exc:
            logger.warning("Error during list_tools, attempting reconnect: %s", exc)
            # Reconnect and retry once
            self._cleanup_session_internal()
            self._ensure_connected(timeout=effective_timeout)
            future = asyncio.run_coroutine_threadsafe(
                self._session.list_tools(),
                self._loop
            )
            response = future.result(timeout=effective_timeout)
            tools = []
            for tool in response.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else None
                })
            return tools

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Synchronously invokes a tool on the persistent MCP server and returns the parsed result.
        """
        effective_timeout = timeout or self.default_timeout
        self._ensure_connected(timeout=effective_timeout)

        def _dispatch():
            future = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name=name, arguments=arguments or {}),
                self._loop
            )
            return future.result(timeout=effective_timeout)

        try:
            res = _dispatch()
        except Exception as exc:
            logger.warning("MCP call_tool('%s') failed (%s), attempting automatic reconnect...", name, exc)
            self._cleanup_session_internal()
            self._ensure_connected(timeout=effective_timeout)
            try:
                res = _dispatch()
            except Exception as retry_exc:
                raise MCPConnectionError(f"Failed to execute MCP tool '{name}': {retry_exc}") from retry_exc

        if getattr(res, "is_error", False):
            err_msg = "Unknown tool execution error"
            if getattr(res, "content", None) and len(res.content) > 0:
                err_msg = getattr(res.content[0], "text", str(res.content[0]))
            return {
                "success": False,
                "error": err_msg
            }

        # Parse response content
        if getattr(res, "content", None) and len(res.content) > 0:
            first_block = res.content[0]
            text_val = getattr(first_block, "text", "")
            if text_val:
                try:
                    parsed = json.loads(text_val)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"success": True, "result": parsed}
                except json.JSONDecodeError:
                    return {"success": True, "result": text_val}

        return {"success": True, "result": None}

    # -----------------------------------------------------------------------
    # High-Level Domain Wrappers for Stage 7C
    # -----------------------------------------------------------------------

    def configure_folder(self, folder_path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Invokes MCP tool 'select_or_configure_folder' to set and validate target local directory.
        """
        return self.call_tool("select_or_configure_folder", {"folder_path": folder_path}, timeout=timeout)

    def scan_folder(
        self,
        folder_path: Optional[str] = None,
        user_id: str = "default_user",
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Invokes MCP tool 'scan_folder' to discover supported files and return sync summary.
        """
        args: Dict[str, Any] = {"user_id": user_id}
        if folder_path is not None:
            args["folder_path"] = folder_path
        return self.call_tool("scan_folder", args, timeout=timeout)

    def read_file(self, file_path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Invokes MCP tool 'read_file' to safely read file contents and metadata.
        """
        return self.call_tool("read_file", {"file_path": file_path}, timeout=timeout)

    def get_file_status(
        self,
        file_path: str,
        user_id: str = "default_user",
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Invokes MCP tool 'get_file_status' to query file status from database sync records.
        """
        return self.call_tool("get_file_status", {"file_path": file_path, "user_id": user_id}, timeout=timeout)

    def is_healthy(self, timeout: float = 5.0) -> bool:
        """Verifies MCP server connectivity and tool readiness."""
        try:
            tools = self.list_tools(timeout=timeout)
            names = {t["name"] for t in tools}
            required = {"select_or_configure_folder", "scan_folder", "read_file", "get_file_status"}
            return required.issubset(names)
        except Exception as exc:
            logger.warning("MCP Client health check failed: %s", exc)
            return False

    def close(self) -> None:
        """Shuts down persistent MCP connection, terminates subprocess, and stops worker thread."""
        with self._lock:
            self._is_closed = True
            self._cleanup_session_internal()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
