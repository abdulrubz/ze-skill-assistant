"""name -> (tool, ToolSpec), built from domain/tools.yaml and nothing else.

Every tool a skill can reach is an entry in domain/tools.yaml, mapped to an
operation of a service's OpenAPI spec (`<service>.<operationId>`). There are
no hand-written tools: skills grant names from that file, the boot self-test
(app/skills/registry.py) rejects any other name, and the tool objects are
generated from the specs.

`declare_tools()` gives every configured tool a placeholder with its real
name, description and argument schema, so skills compile and prompts render
without a network connection; `install_tool()` swaps in the generated
implementation at startup (app/tools/openapi/server.py).

The tool layer is READ-ONLY: the drift check rejects a write in tools.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool

Access = Literal["read"]
CostClass = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    group: str
    service: str
    operation_id: str
    access: Access = "read"
    cost_class: CostClass = "low"


class ToolNotInstalled(RuntimeError):
    pass


_TOOLS: dict[str, BaseTool] = {}
_SPECS: dict[str, ToolSpec] = {}
_INSTALLED: set[str] = set()


def _placeholder(name: str, description: str, args_schema: dict) -> BaseTool:
    async def call(**_kwargs):
        raise ToolNotInstalled(f"tool '{name}' is not installed: the generated tool backend has not been started "
                               "(app.tools.openapi.server.install)")

    return StructuredTool(name=name, description=description, args_schema=args_schema, coroutine=call)


def declare_tools() -> None:
    """Specs and placeholders for every tools.yaml entry. Idempotent; never
    replaces an installed tool."""
    from app.tools.openapi.config import load_tools, model_args_schema, model_description, operation
    from app.tools.openapi.generate import load_generated

    generated, tools = load_generated(), load_tools()
    for name, tool in tools.tools.items():
        op = operation(generated, tool.operation)
        if op is None:
            raise KeyError(f"tools.yaml: '{name}' maps to unknown operation '{tool.operation}' "
                           "(run scripts/tool_spec.py check)")
        if op["kind"] != "read":
            raise ValueError(f"tools.yaml: '{name}' maps to a write ({tool.operation}); the tool layer is read-only")
        _SPECS[name] = ToolSpec(name=name, group=tool.group, service=tool.service, operation_id=tool.operation,
                                cost_class=tool.cost_class)
        if name not in _INSTALLED:
            _TOOLS[name] = _placeholder(name, model_description(generated, tool), model_args_schema(generated, tool))


def install_tool(name: str, tool_obj: BaseTool) -> None:
    """The generated implementation of a configured tool. Must run before
    skill subgraphs are compiled -- they bind tool objects at build."""
    get_spec(name)
    if tool_obj.name != name:
        raise ValueError(f"tool is named '{tool_obj.name}', expected '{name}'")
    _TOOLS[name] = tool_obj
    _INSTALLED.add(name)


def is_installed(name: str) -> bool:
    return name in _INSTALLED


def get_tool(name: str) -> BaseTool:
    if name not in _TOOLS:
        raise KeyError(f"unknown tool: {name}")
    return _TOOLS[name]


def get_spec(name: str) -> ToolSpec:
    if name not in _SPECS:
        raise KeyError(f"unknown tool: {name}")
    return _SPECS[name]


def all_tool_names() -> list[str]:
    return list(_TOOLS)


def all_specs() -> dict[str, ToolSpec]:
    return dict(_SPECS)


def tools_for(names: list[str]) -> list[BaseTool]:
    """Tool objects for a skill's grants."""
    return [get_tool(n) for n in names]
