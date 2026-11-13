"""Render the saved essay measurements using the shared presentation pipeline."""

from render_essay_figures import render_one


def main() -> int:
    print(render_one("failure_by_family"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
