"""Unit tests for MSAF Content -> A2A Part conversion."""

from typing import Any

from a2a.types import DataPart, TextPart
from agent_framework import Content as MsafContent
from pydantic import BaseModel

from agenticlayer.msaf.agent_to_a2a import _msaf_content_to_a2a_part


def _data(content: MsafContent) -> tuple[dict[str, Any], dict[str, Any] | None]:
    part = _msaf_content_to_a2a_part(content)
    root = part.root
    assert isinstance(root, DataPart)
    return root.data, root.metadata


class TestTextContent:
    def test_text_content_returns_text_part(self) -> None:
        content = MsafContent.from_text("hello world")
        part = _msaf_content_to_a2a_part(content)
        assert isinstance(part.root, TextPart)
        assert part.root.text == "hello world"


class TestFunctionCall:
    def test_dict_arguments_pass_through(self) -> None:
        content = MsafContent.from_function_call(
            call_id="call_1",
            name="search",
            arguments={"q": "Anna"},
        )
        data, metadata = _data(content)
        assert data == {"id": "call_1", "name": "search", "args": {"q": "Anna"}}
        assert metadata == {"msaf_type": "function_call"}

    def test_none_arguments_become_empty_dict(self) -> None:
        content = MsafContent.from_function_call(call_id="call_2", name="ping")
        data, _ = _data(content)
        assert data["args"] == {}

    def test_string_arguments_pass_through_as_string(self) -> None:
        content = MsafContent.from_function_call(
            call_id="call_3",
            name="search",
            arguments='{"q": "Anna"}',
        )
        data, _ = _data(content)
        assert data["args"] == '{"q": "Anna"}'

    def test_exception_surfaces_in_metadata(self) -> None:
        content = MsafContent.from_function_call(
            call_id="call_x",
            name="search",
            arguments={},
            exception="bad args",
        )
        _, metadata = _data(content)
        assert metadata == {"msaf_type": "function_call", "exception": "bad args"}


class TestFunctionResult:
    def test_dict_result_passes_through(self) -> None:
        content = MsafContent.from_function_result(call_id="call_1", result={"ok": True})
        data, metadata = _data(content)
        assert data == {"id": "call_1", "response": {"ok": True}}
        assert metadata == {"msaf_type": "function_result"}

    def test_pydantic_result_is_json_decoded(self) -> None:
        # MSAF serialises Pydantic models via str() → '"value=7"' (a JSON string).
        # _to_jsonable JSON-decodes that string back to "value=7".
        class Out(BaseModel):
            value: int

        content = MsafContent.from_function_result(call_id="call_p", result=Out(value=7))
        data, _ = _data(content)
        assert data["response"] == "value=7"

    def test_non_jsonable_result_is_stringified(self) -> None:
        class Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        content = MsafContent.from_function_result(call_id="call_o", result=Opaque())
        data, _ = _data(content)
        assert data["response"] == "<opaque>"

    def test_exception_with_no_result_becomes_error_response(self) -> None:
        content = MsafContent.from_function_result(call_id="call_e", exception="boom")
        data, metadata = _data(content)
        assert data == {"id": "call_e", "response": {"error": "boom"}}
        # exception is consumed by data.response and must NOT be duplicated in metadata
        assert metadata == {"msaf_type": "function_result"}

    def test_exception_with_result_preserved_in_metadata(self) -> None:
        content = MsafContent.from_function_result(
            call_id="call_b",
            result={"ok": True},
            exception="warning",
        )
        data, metadata = _data(content)
        assert data == {"id": "call_b", "response": {"ok": True}}
        assert metadata == {"msaf_type": "function_result", "exception": "warning"}

    def test_result_omits_name_key(self) -> None:
        content = MsafContent.from_function_result(call_id="call_n", result={"ok": True})
        data, _ = _data(content)
        assert "name" not in data


class TestMcpServerToolCall:
    def test_includes_server_name_in_metadata(self) -> None:
        content = MsafContent.from_mcp_server_tool_call(
            call_id="call_m",
            tool_name="search_customer_by_name",
            server_name="crm",
            arguments={"name": "Anna"},
        )
        data, metadata = _data(content)
        assert data == {
            "id": "call_m",
            "name": "search_customer_by_name",
            "args": {"name": "Anna"},
        }
        assert metadata == {"msaf_type": "mcp_server_tool_call", "server_name": "crm"}

    def test_omits_server_name_when_absent(self) -> None:
        content = MsafContent.from_mcp_server_tool_call(
            call_id="call_m2",
            tool_name="ping",
        )
        _, metadata = _data(content)
        assert metadata == {"msaf_type": "mcp_server_tool_call"}


class TestMcpServerToolResult:
    def test_output_becomes_response(self) -> None:
        content = MsafContent.from_mcp_server_tool_result(
            call_id="call_m",
            output={"customers": []},
        )
        data, metadata = _data(content)
        assert data == {"id": "call_m", "response": {"customers": []}}
        assert metadata == {"msaf_type": "mcp_server_tool_result"}


class TestFallback:
    def test_unmapped_type_uses_legacy_flat_shape(self) -> None:
        # `error` content is not in the four covered types; it should still
        # produce a DataPart and must not gain `msaf_type`.
        content = MsafContent.from_error(message="kaboom", error_code="E1")
        data, metadata = _data(content)
        assert data["type"] == "error"
        assert metadata is None
