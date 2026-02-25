"""
QuantGuard — Pathway MCP Server Integration
=============================================
Exposes QuantGuard's fraud-detection RAG pipeline as a Pathway MCP
(Model Context Protocol) server, allowing any MCP-compatible client
(Claude Desktop, VS Code Copilot, custom agents) to query the live
document store through a standardised tool/resource interface.

MCP Specification: https://modelcontextprotocol.io/
Pathway MCP Docs:  https://pathway.com/developers/user-guide/llm-xpack/pathway_mcp_server

Features:
  • Tool: rag_query — ask regulatory / fraud policy questions (RAG)
  • Tool: generate_report — produce fraud reports (executive, detailed, …)
  • Tool: explain_transaction — explainable AI for a specific transaction
  • Resource: /documents — browse indexed policy documents

Usage:
    # Standalone (starts MCP server on stdio or SSE transport):
    python pathway_mcp_server.py

    # Inside main_api.py (mounted as sub-application):
    from pathway_mcp_server import create_mcp_app
    app.mount("/mcp", create_mcp_app())

Architecture:
    MCP Client  ←→  MCP Server (this file)  ←→  PathwayLLMxPack  ←→  Pathway DocumentStore
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Attempt to import the native Pathway MCP server ────────────────────
PATHWAY_MCP_AVAILABLE = False
try:
    from pathway.xpacks.llm.servers import PathwayMCPServer as _PW_MCP
    PATHWAY_MCP_AVAILABLE = True
    print("[MCP] Native Pathway MCP Server available")
except (ImportError, AttributeError):
    print("[MCP] Native Pathway MCP Server not available — using compat layer")


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Tool Definitions (Model Context Protocol)
# ═══════════════════════════════════════════════════════════════════════════

MCP_SERVER_INFO = {
    "name": "quantguard-mcp",
    "version": "1.0.0",
    "description": (
        "QuantGuard Fraud Detection — Pathway MCP Server. "
        "Provides RAG over regulatory policies, fraud report generation, "
        "and explainable AI for financial transactions."
    ),
}

MCP_TOOLS = [
    {
        "name": "rag_query",
        "description": (
            "Query the QuantGuard regulatory knowledge base using "
            "Retrieval-Augmented Generation (RAG). Searches live-indexed "
            "fraud policies, RBI regulations, and risk guidelines."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question about fraud policies or regulations",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of context chunks to retrieve (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Generate a fraud detection report. Supported types: "
            "executive_summary, detailed_analysis, regulatory_compliance, "
            "risk_assessment, trend_analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "enum": [
                        "executive_summary",
                        "detailed_analysis",
                        "regulatory_compliance",
                        "risk_assessment",
                        "trend_analysis",
                    ],
                    "description": "Type of report to generate",
                },
                "time_range_hours": {
                    "type": "integer",
                    "description": "Hours of data to include (default: 24)",
                    "default": 24,
                },
            },
            "required": ["report_type"],
        },
    },
    {
        "name": "explain_transaction",
        "description": (
            "Get an explainable AI analysis for a specific transaction. "
            "Provides regulatory context, risk factors, and decision rationale."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Transaction ID to analyze",
                },
                "amount": {
                    "type": "number",
                    "description": "Transaction amount",
                },
                "category": {
                    "type": "string",
                    "description": "Transaction category (e.g., wire_transfer, atm_withdrawal)",
                },
            },
            "required": ["transaction_id"],
        },
    },
]

MCP_RESOURCES = [
    {
        "uri": "quantguard://documents",
        "name": "Indexed Documents",
        "description": "Browse all documents indexed in the QuantGuard DocumentStore",
        "mimeType": "application/json",
    },
    {
        "uri": "quantguard://status",
        "name": "System Status",
        "description": "Current QuantGuard system status, indexed doc count, and capabilities",
        "mimeType": "application/json",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Server Implementation (Compat Layer)
# ═══════════════════════════════════════════════════════════════════════════

class QuantGuardMCPServer:
    """
    MCP-compatible server that wraps PathwayLLMxPack.

    Implements the Model Context Protocol with:
      - tools/list, tools/call  — RAG query, reports, explainability
      - resources/list, resources/read — document browsing
      - initialize, ping        — lifecycle management

    When the native Pathway MCP Server is available, delegates to it.
    Otherwise, implements the JSON-RPC 2.0 MCP protocol directly.
    """

    def __init__(self, xpack=None):
        self.xpack = xpack
        self._initialized = False
        self._start_time = datetime.now()

    def _ensure_xpack(self):
        """Lazy-load xpack if not provided."""
        if self.xpack is None:
            try:
                from pathway_llm_xpack import PathwayLLMxPack
                self.xpack = PathwayLLMxPack()
            except Exception as e:
                print(f"[MCP] Failed to initialize xPack: {e}")
                return False
        return True

    # ── JSON-RPC 2.0 MCP Protocol Handler ──────────────────────────────

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single MCP JSON-RPC 2.0 request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        handlers = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
        }

        handler = handlers.get(method)
        if handler is None:
            return self._error_response(req_id, -32601, f"Method not found: {method}")

        try:
            result = await handler(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            return self._error_response(req_id, -32000, str(e))

    async def _handle_initialize(self, params: Dict) -> Dict:
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": MCP_SERVER_INFO,
        }

    async def _handle_ping(self, params: Dict) -> Dict:
        return {}

    async def _handle_tools_list(self, params: Dict) -> Dict:
        return {"tools": MCP_TOOLS}

    async def _handle_tools_call(self, params: Dict) -> Dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not self._ensure_xpack():
            return {
                "content": [{"type": "text", "text": "xPack not available"}],
                "isError": True,
            }

        if tool_name == "rag_query":
            return await self._call_rag_query(arguments)
        elif tool_name == "generate_report":
            return await self._call_generate_report(arguments)
        elif tool_name == "explain_transaction":
            return await self._call_explain_transaction(arguments)
        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

    async def _handle_resources_list(self, params: Dict) -> Dict:
        return {"resources": MCP_RESOURCES}

    async def _handle_resources_read(self, params: Dict) -> Dict:
        uri = params.get("uri", "")

        if uri == "quantguard://documents":
            return await self._read_documents_resource()
        elif uri == "quantguard://status":
            return await self._read_status_resource()
        else:
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": f"Unknown resource: {uri}",
                }],
            }

    # ── Tool Implementations ───────────────────────────────────────────

    async def _call_rag_query(self, args: Dict) -> Dict:
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        result = self.xpack.rag.query(query, top_k=top_k)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2, default=str),
            }],
        }

    async def _call_generate_report(self, args: Dict) -> Dict:
        report_type = args.get("report_type", "executive_summary")
        time_range = args.get("time_range_hours", 24)

        # Load recent alerts for report context
        alerts = []
        alerts_path = os.path.join("data", "high_risk_alerts.jsonl")
        if os.path.exists(alerts_path):
            with open(alerts_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        report = self.xpack.reports.generate_report(
            report_type=report_type,
            data={"alerts": alerts[-50:], "time_range_hours": time_range},
        )
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(report, indent=2, default=str),
            }],
        }

    async def _call_explain_transaction(self, args: Dict) -> Dict:
        tx = {
            "id": args.get("transaction_id", "unknown"),
            "amount": args.get("amount", 0),
            "category": args.get("category", "unknown"),
        }
        qr = {"prediction": "pending", "confidence": 0.0}

        explanation = self.xpack.insights.explain_decision(tx, qr)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(explanation, indent=2, default=str),
            }],
        }

    # ── Resource Implementations ───────────────────────────────────────

    async def _read_documents_resource(self) -> Dict:
        if not self._ensure_xpack():
            docs = []
        else:
            docs = [
                {
                    "id": d.get("id", ""),
                    "metadata": d.get("metadata", {}),
                    "content_preview": d.get("content", "")[:200],
                }
                for d in self.xpack.store.documents
            ]
        return {
            "contents": [{
                "uri": "quantguard://documents",
                "mimeType": "application/json",
                "text": json.dumps({"count": len(docs), "documents": docs}, indent=2),
            }],
        }

    async def _read_status_resource(self) -> Dict:
        status = {
            "service": "quantguard-mcp",
            "status": "healthy",
            "pathway_native": False,
            "xpack_ready": self.xpack is not None,
            "documents_indexed": self.xpack.store.size if self.xpack else 0,
            "uptime": str(datetime.now() - self._start_time),
            "tools": [t["name"] for t in MCP_TOOLS],
        }
        try:
            from pathway_llm_xpack import XPACK_NATIVE
            status["pathway_native"] = XPACK_NATIVE
        except ImportError:
            pass
        return {
            "contents": [{
                "uri": "quantguard://status",
                "mimeType": "application/json",
                "text": json.dumps(status, indent=2),
            }],
        }

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _error_response(req_id, code, message):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI Sub-Application (mount on main API)
# ═══════════════════════════════════════════════════════════════════════════

def create_mcp_app(xpack=None):
    """
    Create a FastAPI sub-app that serves the MCP protocol over HTTP/SSE.

    Usage in main_api.py:
        from pathway_mcp_server import create_mcp_app
        app.mount("/mcp", create_mcp_app(xpack=xpack_instance))
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    mcp_app = FastAPI(
        title="QuantGuard MCP Server",
        description="Pathway MCP Server for fraud detection RAG",
        version="1.0.0",
    )
    server = QuantGuardMCPServer(xpack=xpack)

    @mcp_app.post("/")
    async def mcp_endpoint(request: dict):
        """MCP JSON-RPC 2.0 endpoint."""
        response = await server.handle_request(request)
        return JSONResponse(content=response)

    @mcp_app.get("/health")
    async def mcp_health():
        return {"status": "healthy", "mcp_server": "quantguard"}

    @mcp_app.get("/tools")
    async def list_tools():
        """List available MCP tools (convenience endpoint)."""
        return {"tools": MCP_TOOLS}

    return mcp_app


# ═══════════════════════════════════════════════════════════════════════════
#  Stdio Transport (for CLI / Claude Desktop MCP config)
# ═══════════════════════════════════════════════════════════════════════════

async def run_stdio_server():
    """Run MCP server over stdin/stdout (JSON-RPC 2.0 line-delimited)."""
    import asyncio

    server = QuantGuardMCPServer()
    print("[MCP] QuantGuard MCP Server running on stdio", file=sys.stderr)
    print(f"[MCP] Tools: {[t['name'] for t in MCP_TOOLS]}", file=sys.stderr)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            request = json.loads(line.decode().strip())
            response = await server.handle_request(request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue


# ═══════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    if "--http" in sys.argv:
        # HTTP mode: run as standalone FastAPI server on port 8001
        import uvicorn
        mcp_app = create_mcp_app()
        print("[MCP] Starting HTTP MCP server on port 8001")
        uvicorn.run(mcp_app, host="0.0.0.0", port=8001)
    else:
        # Default: stdio transport (for Claude Desktop / MCP clients)
        asyncio.run(run_stdio_server())
