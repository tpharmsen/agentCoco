from pathlib import Path

from .graph import create_graph
from .terminal import format_agent_response, format_user_prompt


def approve_command(command: str, workspace: Path) -> bool:
    print(f"\nThe coding agent wants to run:\n  {command}\nIn: {workspace}")
    return input("Approve this command? [y/N] ").strip().lower() in {"y", "yes"}


def main() -> None:
    graph = create_graph(approve_command=approve_command)
    print("Lewis agents ready. Type 'exit' to quit.")
    while True:
        prompt = input(format_user_prompt())
        if prompt.strip().lower() in {"exit", "quit"}:
            return
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": prompt}],
                "route": None,
            },
            config={"configurable": {"thread_id": "cli"}},
        )
        print(format_agent_response(result["messages"][-1].content))


if __name__ == "__main__":
    main()
