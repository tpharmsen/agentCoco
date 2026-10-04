"""Export the Lewis LangGraph as a standard PNG image and optional Mermaid source."""

from argparse import ArgumentParser
from pathlib import Path

from lewis_agents.config import Settings
from lewis_agents.graph import create_graph


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lewis-graph.png"),
        help="PNG output path (default: artifacts/lewis-graph.png)",
    )
    parser.add_argument("--mermaid", type=Path, help="Optional Mermaid source output path")
    args = parser.parse_args()

    # Graph construction does not call a model or external API.
    graph = create_graph(Settings(_env_file=".env"))
    mermaid = graph.get_graph().draw_mermaid()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Wrote PNG graph to {args.output}")

    if args.mermaid:
        args.mermaid.parent.mkdir(parents=True, exist_ok=True)
        args.mermaid.write_text(mermaid)
        print(f"Wrote Mermaid graph to {args.mermaid}")


if __name__ == "__main__":
    main()
