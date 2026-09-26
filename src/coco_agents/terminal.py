import os
import sys

RESET = "\033[0m"
USER_COLOR = "\033[37m"
AGENT_COLOR = "\033[1;34m"


def colors_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def format_user_prompt() -> str:
    prompt = ">>> "
    return f"{USER_COLOR}{prompt}{RESET}" if colors_enabled() else prompt


def format_agent_response(response: str) -> str:
    if not colors_enabled():
        return response
    return f"{AGENT_COLOR}{response}{RESET}"
