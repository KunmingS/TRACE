"""Development backend runner for TRACE.

This module exists instead of using ``python -m uvicorn ... --loop none`` from
the CLI. Some uvicorn releases support ``loop="none"`` in ``Config`` but do not
expose it as a valid CLI choice. Calling uvicorn programmatically keeps reload
mode compatible while letting Windows keep its Proactor event-loop policy for
ffmpeg subprocess streaming.
"""
import argparse
import asyncio
import sys


def _set_windows_subprocess_policy() -> None:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the TRACE dev backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    _set_windows_subprocess_policy()
    import uvicorn

    uvicorn.run(
        "trace_tad.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="none",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
