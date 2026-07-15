"""Isaac Sim launcher guard."""


def main() -> int:
    try:
        import isaacsim  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Isaac Sim 6.0.1 is not available. "
            "Launch this script with the Isaac Sim Python runtime."
        ) from exc
    raise SystemExit("Merge the local official Isaac Sim 6.0.1 UI template before GUI use.")


if __name__ == "__main__":
    raise SystemExit(main())
