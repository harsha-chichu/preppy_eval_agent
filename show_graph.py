"""Render the evaluation agent's LangGraph structure.

Run:  uv run python show_graph.py

Writes ``graph.mmd`` (Mermaid source) and tries ``graph.png`` (needs network -
falls back gracefully), and prints an ASCII diagram to the terminal.
"""

from pathlib import Path

from src.agent.graph import _GRAPH

ROOT = Path(__file__).parent


def main() -> None:
    g = _GRAPH.get_graph()

    mermaid = g.draw_mermaid()
    (ROOT / "graph.mmd").write_text(mermaid, encoding="utf-8")
    print("wrote graph.mmd")

    try:
        png = g.draw_mermaid_png()
        (ROOT / "graph.png").write_bytes(png)
        print("wrote graph.png")
    except Exception as exc:  # network / renderer unavailable
        print(f"skipped graph.png ({exc.__class__.__name__}: {exc})")

    print("\n--- ASCII ---")
    try:
        print(g.draw_ascii())
    except ImportError:
        print("(install 'grandalf' for ASCII: uv add --dev grandalf)")

    print("\n--- Mermaid (paste into https://mermaid.live) ---")
    print(mermaid)


if __name__ == "__main__":
    main()
