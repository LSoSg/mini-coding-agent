"""OpenAI-compatible descriptions of tools exposed to the LLM."""

from typing import Any


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List one directory inside the workspace and distinguish files "
                "from directories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative directory path. Use '.' for the root."
                        ),
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Recursively search UTF-8 code and text files in the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Case-insensitive text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative directory path. Use '.' for the root."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of matches to return.",
                    },
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite one UTF-8 text file inside the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative destination file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 text content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "Run one allowed command in the workspace. Command must be a JSON "
                "array matching exactly: ['pytest'], ['python', '-m', 'pytest'], "
                "['python', '<script.py>'], ['git', 'status'], or ['git', 'diff']."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Command and arguments as separate strings.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]
