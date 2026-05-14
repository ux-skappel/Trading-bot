"""CLI entry point.

Usage:
    python -m trading_bot.main run            # one ingestion+scoring cycle
    python -m trading_bot.main loop --every 300   # repeat every 300s
    python -m trading_bot.main approve <trade_id>
    python -m trading_bot.main backtest --events events.csv --prices prices.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .app import TradingBotApp
from .backtesting.backtest import run_backtest
from .config import get_settings
from .trading.paper_broker import PaperBroker

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _print_safety_banner() -> None:
    s = get_settings()
    print("=" * 78)
    print(" TRADING BOT — SAFETY BANNER")
    print(f"   mode               : {s.trading_mode}")
    print(f"   live_trading flag  : {s.live_trading}  (broker={s.broker})")
    print(f"   human_approval     : {s.human_approval_mode}")
    print(f"   require_manual_conf: {s.require_manual_confirmation}")
    print(" This MVP only ships a paper broker. Live trading is structurally")
    print(" disabled — no code path exists to send a real order.")
    print(" This software is NOT financial advice.")
    print("=" * 78)


def cmd_run(args: argparse.Namespace) -> int:
    app = TradingBotApp()
    result = app.run_once()
    print(f"ingested={result.ingested}  normalized={result.normalized}  "
          f"opportunities={result.opportunities}  high_impact={result.high_impact_hits}")
    for rec in sorted(result.recommendations, key=lambda r: r.score, reverse=True)[:15]:
        print(
            f"  {rec.direction.upper():5} {rec.symbol:10} "
            f"score={rec.score:6.2f} conf={rec.confidence:.2f} "
            f"pending={rec.approval_pending} trade_id={rec.trade_id}"
        )
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    app = TradingBotApp()
    while True:
        try:
            result = app.run_once()
            logger.info("Cycle: %s", result)
        except KeyboardInterrupt:
            logger.info("interrupted")
            return 0
        except Exception:
            logger.exception("cycle failed")
        time.sleep(args.every)


def cmd_approve(args: argparse.Namespace) -> int:
    broker = PaperBroker()
    res = broker.approve_pending(args.trade_id, approver=args.approver)
    print(res)
    return 0 if res.accepted else 1


def cmd_backtest(args: argparse.Namespace) -> int:
    result = run_backtest(args.events, args.prices, horizon_bars=args.horizon)
    print(
        f"n_trades={result.n_trades} win_rate={result.win_rate} "
        f"avg_return={result.avg_return} max_dd={result.max_drawdown} "
        f"sharpe={result.sharpe}"
    )
    if result.best_trade:
        print(f"best : {result.best_trade}")
    if result.worst_trade:
        print(f"worst: {result.worst_trade}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading_bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run").set_defaults(func=cmd_run)

    loop = sub.add_parser("loop")
    loop.add_argument("--every", type=int, default=300)
    loop.set_defaults(func=cmd_loop)

    appr = sub.add_parser("approve")
    appr.add_argument("trade_id", type=int)
    appr.add_argument("--approver", default="cli")
    appr.set_defaults(func=cmd_approve)

    bt = sub.add_parser("backtest")
    bt.add_argument("--events", required=True)
    bt.add_argument("--prices", required=True)
    bt.add_argument("--horizon", type=int, default=5)
    bt.set_defaults(func=cmd_backtest)

    return p


def main(argv=None) -> int:
    s = get_settings()
    _configure_logging(s.log_level)
    _print_safety_banner()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
