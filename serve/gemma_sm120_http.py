"""Compatibility CLI delegating to the shared HTTP/SSE/Responses server."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--lazy', action='store_true')
    args = parser.parse_args()
    from serve.server import main as shared_main
    previous = sys.argv
    sys.argv = [previous[0], '--engine', 'strata', '--config', str(args.config),
                '--port', str(args.port)]
    if args.host:
        sys.argv += ['--host', args.host]
    if args.lazy:
        sys.argv += ['--lazy']
    try:
        return shared_main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
