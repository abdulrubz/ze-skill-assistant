"""The drift check: tools.yaml against the generated files. CI runs it after
every spec sync, so a spec change that breaks a tool fails the build with one
line naming the tool, instead of quietly degrading answers."""

from __future__ import annotations

import copy

from app.tools.openapi.config import ToolConfig, ToolsFile, load_tools
from app.tools.openapi.drift import check, errors
from app.tools.openapi.generate import load_generated


def _check(tool: ToolConfig, generated: dict | None = None):
    return errors(check(generated or load_generated(), ToolsFile(tools={"t": tool})))


def test_the_committed_tools_yaml_fits_the_specs():
    assert errors(check(load_generated(), load_tools())) == []


def test_a_write_is_rejected_because_the_tool_layer_is_read_only():
    problems = _check(ToolConfig(operation="orders.cancelOrder", description="x"))
    assert any("read-only" in p.message for p in problems)


def test_an_operation_must_be_namespaced_by_service():
    problems = _check(ToolConfig(operation="getOrder", description="x"))
    assert any("<service>.<operationId>" in p.message for p in problems)


def test_a_removed_operation_names_the_service():
    problems = _check(ToolConfig(operation="orders.getOrderV0", description="x"))
    assert any("orders spec" in p.message for p in problems)


def test_an_undocumented_operation_is_refused():
    g = copy.deepcopy(load_generated())
    g["operations"]["orders.getOrder"]["documented"] = False
    problems = _check(ToolConfig(operation="orders.getOrder", description="x"), g)
    assert any("not documented" in p.message for p in problems)


def test_an_output_projection_onto_a_vanished_field_is_caught():
    g = copy.deepcopy(load_generated())
    del g["schemas"]["orders.Order"]["properties"]["total"]
    tool = ToolConfig.model_validate({"operation": "orders.getOrder", "description": "x",
                                      "output": {"fields": {"total": "total"}}})
    problems = _check(tool, g)
    assert any("'total' no longer exists" in p.message for p in problems)
