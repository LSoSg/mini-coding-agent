"""Consistency checks between LLM schemas and local tool functions."""

import inspect

from tool_schemas import TOOLS
from tools import TOOL_REGISTRY


def test_schema_names_match_local_registry() -> None:
    schema_names = {schema["function"]["name"] for schema in TOOLS}

    assert schema_names == set(TOOL_REGISTRY)


def test_schema_parameters_match_function_signatures() -> None:
    for schema in TOOLS:
        function_schema = schema["function"]
        function = TOOL_REGISTRY[function_schema["name"]]
        signature = inspect.signature(function)
        parameters = function_schema["parameters"]
        required = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        }

        assert set(parameters["properties"]) == set(signature.parameters)
        assert set(parameters.get("required", [])) == required
        assert parameters["additionalProperties"] is False
