"""Run the new local review/control surface; no numerical solver is started."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rinksolve.server import make_server
from rinksolve.store import checked_path, CurrentStore


def main():
    parser = argparse.ArgumentParser(description="RinkSolve 2 本地审阅与约束控制界面（不启动求解）")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data", type=Path, default=Path("D:/CodexData/RinkSpace_Data"))
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--event", default="ankara_2026")
    parser.add_argument("--programme", default="okada_mei_sp")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    log_path = checked_path(args.log or args.data / "runs" / "solve2_control" / "server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    programme=CurrentStore(args.root,args.data,event=args.event,programme=args.programme).programme_input()
    if programme.get("control_surface",{}).get("enabled") is False:
        logging.info("用户已停用此控制台；不启动服务、不打开界面。")
        return
    server = make_server(args.root, args.data, args.host, args.port, args.event, args.programme)
    logging.info("本地控制页面 http://%s:%s/ 主日志 %s；控制服务不启动数值任务，实际数值状态由 /api/state 报告", args.host, server.server_address[1], log_path)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        logging.info("本地控制服务停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
