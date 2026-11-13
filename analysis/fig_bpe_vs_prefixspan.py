"""Render the saved essay measurements using the shared presentation pipeline."""

from render_essay_figures import render_one


def main() -> int:
    print(render_one("fig_bpe_vs_prefixspan"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
