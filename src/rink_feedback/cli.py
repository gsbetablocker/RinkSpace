"""Explicit-input command line interface; standard library only."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .core import Store, ValidationError


def _read(path):
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8-sig") as handle:
        if Path(path).suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent, source-bound feedback ledger. Does not fit or train.")
    parser.add_argument("--store", required=True, help="Explicit SQLite output path")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "import-cases", "query", "finding", "use", "resolve", "export"):
        command = commands.add_parser(name)
        if name not in {"init", "status", "export"}:
            command.add_argument("--input", required=True, help="JSON file, or - for stdin")
        if name == "export":
            command.add_argument("--output", help="Optional explicit JSON output file; otherwise stdout")
    args = parser.parse_args(argv)
    try:
        with Store(args.store) as store:
            if args.command in {"init", "status"}:
                result = store.stats()
            elif args.command == "export":
                result = store.export()
            else:
                payload = _read(args.input)
                if args.command == "import-cases":
                    cases = payload.get("cases", [payload]) if isinstance(payload, dict) else payload
                    if not isinstance(cases, list):
                        raise ValidationError("import-cases input must be a case, list, or {cases:[...]}")
                    result = [store.upsert_case(case) for case in cases]
                else:
                    method = {"query": store.query, "finding": store.record_finding, "use": store.record_use, "resolve": store.record_resolution}[args.command]
                    result = method(payload)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            print(json.dumps({"output": str(destination), "written": True}, ensure_ascii=False))
        else:
            print(rendered, end="")
        return 0
    except (ValidationError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc), "action": "Correct the explicit input or select a compatible store; no fit/training was run."}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
