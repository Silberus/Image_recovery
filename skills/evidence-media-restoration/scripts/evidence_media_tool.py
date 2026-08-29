#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from evidence_media.io_media import inspect_source
from evidence_media.pipeline import run_pipeline
from evidence_media.selftest import run_selftest


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence-preserving image/video restoration")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect", help="Inventory codec/container or image sequence")
    inspect_p.add_argument("input", type=Path)
    restore_p = sub.add_parser("restore", help="Run the complete restoration pipeline")
    restore_p.add_argument("input", type=Path)
    restore_p.add_argument("output", type=Path)
    restore_p.add_argument("--config", type=Path, required=True)
    self_p = sub.add_parser("selftest", help="Run a deterministic synthetic regression test")
    self_p.add_argument("output", type=Path)
    self_p.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_source(args.input), indent=2, ensure_ascii=False))
        return 0
    config = load_config(args.config)
    if args.command == "restore":
        print(json.dumps(run_pipeline(args.input, args.output, config), indent=2, ensure_ascii=False))
        return 0
    if args.command == "selftest":
        print(json.dumps(run_selftest(args.output, config), indent=2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
