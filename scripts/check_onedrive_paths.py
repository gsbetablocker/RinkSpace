"""Read-only Windows path check; includes ignored environment/cache contents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from rinkspace.path_policy import check_project_path, scan_project_paths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读检查 OneDrive 项目的 Windows 路径限制")
    parser.add_argument("--root", type=Path, default=PROJECT, help="项目根目录（默认当前项目）")
    parser.add_argument("--candidate", type=Path, action="append", default=[], help="额外检查拟创建路径，可重复；相对路径以项目根为基准")
    parser.add_argument("--candidate-only", action="store_true", help="仅检查 --candidate，不扫描现有目录")
    parser.add_argument("--json", action="store_true", help="在标准输出提供 JSON，不写入文件")
    args = parser.parse_args(argv)
    if args.candidate_only and not args.candidate:
        parser.error("--candidate-only 必须同时提供 --candidate")
    root = args.root.resolve()
    report = {
        "root": str(root), "checked_paths": 0, "violations": [],
        "skipped_links": [], "errors": [], "ok": True,
    } if args.candidate_only else scan_project_paths(root).as_dict()
    for candidate in args.candidate:
        physical = (candidate if candidate.is_absolute() else root / candidate).resolve()
        report["checked_paths"] += 1
        report["violations"].extend(
            {"path": row.path, "reason": row.reason, "actual_units": row.actual_units,
             "allowed_units": row.allowed_units, "component": row.component}
            for row in check_project_path(physical, project_root=root)
        )
    report["ok"] = not report["violations"] and not report["errors"]
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{'通过' if report['ok'] else '失败'}：检查 {report['checked_paths']} 项路径；"
              f"超限 {len(report['violations'])} 项，读取错误 {len(report['errors'])} 项，"
              f"跳过链接 {len(report['skipped_links'])} 项。")
        for row in report["violations"]:
            kind = "绝对路径" if row["reason"] == "absolute_path" else "路径组件"
            print(f"{kind} {row['actual_units']}/{row['allowed_units']} UTF-16 单元：{row['path']}")
        for row in report["errors"]:
            print(f"读取失败：{row['path']}：{row['error']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
