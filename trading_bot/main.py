"""CLI entry point.

Usage:
    python -m trading_bot.main run            # one ingestion+scoring cycle
    python -m trading_bot.main loop --every 300   # repeat every 300s
    python -m trading_bot.main approve <trade_id>
    python -m trading_bot.main backtest --events events.csv --prices prices.csv
    python -m trading_bot.main levels AAPL BTC     # support/resistance
    python -m trading_bot.main levels --csv chart.csv
    python -m trading_bot.main strategy --market us crypto   # support-bounce test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .analysis.support_resistance import SupportResistance, bars_from_csv_text, detect_levels
from .app import TradingBotApp
from .backtesting.backtest import run_backtest
from .backtesting.support_bounce import BounceParams, BounceResult, backtest_support_bounce, summarize
from .config import get_settings
from .connectors.market_data import MarketDataProvider, infer_asset_class
from .trading.paper_broker import PaperBroker

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    logging.getLogger("telethon").setLevel(logging.WARNING)   # no per-download chatter


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


def _print_levels(sr: SupportResistance, source: str, n_bars: int) -> None:
    print(
        f"\n{sr.symbol}  price {sr.current_price:.4f}  "
        f"({source}, {n_bars} bars, tolerance {sr.tolerance_pct:.1%})"
    )
    if not sr.supports and not sr.resistances:
        print("  (not enough data to find levels)")
        return
    for lv in sorted(sr.resistances, key=lambda lv: lv.price, reverse=True):
        print(f"  RESISTANCE {lv.price:>12.4f}  {lv.distance_pct:+7.1%}  touches {lv.touches}  strength {lv.strength:.2f}")
    print(f"  -- price -- {sr.current_price:>12.4f}")
    for lv in sr.supports:
        print(f"  SUPPORT    {lv.price:>12.4f}  {lv.distance_pct:+7.1%}  touches {lv.touches}  strength {lv.strength:.2f}")


def cmd_levels(args: argparse.Namespace) -> int:
    s = get_settings()
    if args.csv:
        with open(args.csv, encoding="utf-8-sig") as f:
            bars = bars_from_csv_text(f.read())
        sr = detect_levels(bars, symbol=os.path.basename(args.csv),
                           window=s.sr_window, max_levels=s.sr_max_levels)
        _print_levels(sr, "csv", len(bars))
        return 0

    market_data = MarketDataProvider(s.market_data_provider)
    for symbol in args.symbols or (s.stock_watchlist + s.crypto_watchlist):
        history = market_data.get_history(symbol, infer_asset_class(symbol))
        sr = detect_levels(history.bars, symbol=symbol,
                           window=s.sr_window, max_levels=s.sr_max_levels)
        _print_levels(sr, history.source, len(history.bars))
    return 0


def _pct(value) -> str:
    return "-" if value is None else f"{value:+.1%}"


def _print_bounce_row(r: BounceResult, source: str) -> None:
    win = "-" if r.win_rate is None else f"{r.win_rate:.0%}"
    print(
        f"{r.symbol:<10} {source:>8} {len(r.trades):>6} {win:>5} {_pct(r.avg_return):>9} "
        f"{_pct(r.total_return):>9} {_pct(r.buy_hold_return):>9} {_pct(r.max_drawdown):>7} "
        f"{r.time_in_market:>6.0%} {r.liquidations:>4}"
    )


def cmd_strategy(args: argparse.Namespace) -> int:
    s = get_settings()
    if not 1 <= args.leverage <= 50:
        print("--leverage must be between 1 and 50 (Liquid allows up to 40x on crypto)")
        return 2
    if args.risk is not None and not 0 < args.risk <= 20:
        print("--risk is a percent of the account, between 0 and 20 (e.g. 1)")
        return 2
    params = BounceParams.from_settings(
        s, min_touches=args.min_touches, min_reward_risk=args.min_rr,
        leverage=args.leverage, max_hold_days=args.max_days, trailing=args.trailing,
        risk_per_trade=args.risk / 100 if args.risk else None,
    )
    hold = f"max {params.max_hold_days} days" if params.max_hold_days else "no time limit"
    size = (f"risk {args.risk:g}% per trade (max {params.leverage:g}x)" if params.risk_per_trade
            else f"{params.leverage:g}x leverage")
    trail = "trailing stop" if params.trailing else "fixed stop"
    print(
        f"\nSupport-bounce backtest · daily candles · min {params.min_touches} touches · "
        f"min reward/risk {params.min_reward_risk} · {size} · {trail} · {hold}\n"
        f"Costs like Liquid perps: fee {params.fee_pct:.3%} per side, funding "
        f"{params.funding_pct_per_day:.2%}/day, liquidation at {params.maintenance_margin:.2%} "
        f"maintenance margin\n"
    )
    print(f"{'SYMBOL':<10} {'DATA':>8} {'TRADES':>6} {'WIN':>5} {'AVG/TRADE':>9} "
          f"{'STRATEGY':>9} {'BUY&HOLD':>9} {'MAX DD':>7} {'IN MKT':>6} {'LIQ':>4}")

    if args.csv:
        with open(args.csv, encoding="utf-8-sig") as f:
            jobs = [(os.path.basename(args.csv), bars_from_csv_text(f.read()), "csv")]
    else:
        market_data = MarketDataProvider(s.market_data_provider)
        jobs = []
        for symbol in args.symbols or s.symbols_for(args.market):
            history = market_data.get_history(symbol, infer_asset_class(symbol), period=args.period)
            jobs.append((symbol, history.bars, history.source))

    results = []
    for symbol, bars, source in jobs:
        r = backtest_support_bounce(bars, symbol, params)
        results.append(r)
        _print_bounce_row(r, source)
        if args.trades:
            for t in r.trades:
                print(f"    {t.entry_time:%Y-%m-%d} buy {t.entry_price:.4g} (support {t.support:.4g}, "
                      f"{t.support_touches}x, size {t.position_size:.2f}x) -> {t.exit_time:%Y-%m-%d} "
                      f"{t.exit_reason} {t.exit_price:.4g}  {t.return_pct:+.1%}  ({t.days_held}d)")

    total = summarize(results)
    if total["trades"]:
        pf = "-" if total["profit_factor"] is None else f"{total['profit_factor']:.2f}"
        print(
            f"\nALL: {total['trades']} trades · win {total['win_rate']:.0%} · "
            f"avg {total['avg_return']:+.2%} per trade · profit factor {pf} · "
            f"avg {total['avg_days_held']:.1f} days held · "
            f"beat buy & hold in {total['beat_buy_hold']} of {total['symbols']}\n"
            f"     one account taking every trade: {total['account_return']:+.1%} "
            f"(biggest fall {total['account_max_drawdown']:+.1%})"
        )
        if params.leverage > 1:
            print(f"     {total['liquidations']} liquidations · "
                  f"{total['wiped_out']} of {total['symbols']} accounts wiped out")
    else:
        print("\nNo trades — the strategy never triggered on this data.")
    print("Past results do not guarantee future returns.")
    return 0


def _print_copy_report(result) -> None:
    trade_posts = [o for o in result.posts if o.signal.action != "info"]
    print(f"\nPaper copy of {len(result.posts)} posts: {len(trade_posts)} trade instructions, "
          f"{len(result.posts) - len(trade_posts)} other posts\n")
    for o in result.posts:
        if o.signal.action == "info":
            continue
        print(f"  {o.post.time:%Y-%m-%d %H:%M}  {o.signal.summary()}\n      -> {o.outcome}")
    print()
    for t in result.trades:
        exits = ", ".join(f"{f.reason} {f.price:g}" for f in t.fills) or "-"
        entry = f"{t.entry_price:g}" if t.entry_price else "-"
        delay = f" ({t.delay_cost:+.2%} vs his entry)" if t.delay_cost is not None else ""
        ret = f" ({t.return_on_margin:+.1%} on margin)" if t.return_on_margin is not None else ""
        money = t.pnl + t.unrealized
        print(f"  #{t.number:<3} {t.symbol:<6} {t.side:<5} {t.leverage:g}x  {t.status:<9} entry {entry}{delay}"
              f"  exits: {exits}  -> ${money:+,.2f}{ret}{'  [open]' if t.status == 'open' else ''}"
              f"{'  ' + t.note if t.status in ('skipped', 'expired', 'cancelled') else ''}")
    win = "-" if result.win_rate is None else f"{result.win_rate:.0%}"
    delay = "-" if result.avg_delay_cost is None else f"{result.avg_delay_cost:+.2%}"
    print(f"\nRealized ${result.realized_pnl:+,.2f} ({result.total_return:+.1%} of ${result.start_equity:,.0f}) · "
          f"win rate {win} · max drawdown {result.max_drawdown:+.1%} · avg entry vs his {delay}")
    print("Paper only: no real orders. Past results do not guarantee future returns.")


def _write_copy_review(result, path: Path) -> int:
    """Every post of his with what the bot understood and did, for someone to check."""
    lines = []
    for o in result.posts:
        post, sig = o.post, o.signal
        if not post.text.strip() or (post.sell_only and sig.action == "info"):
            continue   # bare images, and chat in the discussion topics
        lines.append(f"--- {post.time:%Y-%m-%d %H:%M} UTC · message {post.message_id}"
                     f"{' · reply to ' + str(post.reply_to) if post.reply_to else ''}"
                     f"{' · discussion topic (sells only)' if post.sell_only else ''}")
        lines.append(post.text.strip())
        if post.image_text:
            lines.append("[image] " + post.image_text.replace("\n", " | ")[:200])
        lines.append(f">>> understood: {sig.summary()}" + (f"  ({'; '.join(sig.notes)})" if sig.notes else ""))
        lines.append(f">>> bot did: {o.outcome}\n")
    lines.append("=== Copied trades (test on old posts, each sized from the account at that moment)")
    for t in result.trades:
        exits = ", ".join(f"{f.reason} {f.price:g}" for f in t.fills) or "-"
        lines.append(f"#{t.number} {t.symbol} {t.side} {t.leverage:g}x {t.status} · margin ${t.margin:,.0f} · "
                     f"entry {t.entry_price or '-'} · exits {exits} · P/L ${t.pnl + t.unrealized:+,.2f} · {t.note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(result.posts)


def cmd_copy(args: argparse.Namespace) -> int:
    from .copytrade.parser import parse_signal

    if args.copy_cmd in ("login", "chats", "authors", "topics", "backfill", "listen"):
        from .copytrade.telegram_reader import run
        run(args.copy_cmd, days=getattr(args, "days", 90), limit=getattr(args, "limit", 500),
            fresh=getattr(args, "fresh", False))
        return 0
    if args.copy_cmd == "posts":
        from .copytrade.store import load_posts
        posts = load_posts()[-args.limit:]
        if not posts:
            print("No posts stored yet. Run `copy backfill` first.")
            return 1
        for post in posts:
            sig = parse_signal(post.text)
            print(f"\n--- {post.time:%Y-%m-%d %H:%M} UTC · message {post.message_id}"
                  f"{' · reply to ' + str(post.reply_to) if post.reply_to else ''}")
            print(post.text.strip() or "(image)")
            print(f">>> {sig.summary() if post.text.strip() else 'image: content cannot be read'}")
        return 0
    if args.copy_cmd == "images":
        from .copytrade.store import load_posts
        with_images = [p for p in load_posts() if p.image_text is not None][-args.limit:]
        if not with_images:
            print("No image text yet. On a Mac, run `copy backfill --days 90` after updating; "
                  "images are read with the Mac's built-in text recognition.")
            return 1
        for post in with_images:
            print(f"\n--- {post.time:%Y-%m-%d %H:%M} UTC · message {post.message_id}"
                  f"{' · reply to ' + str(post.reply_to) if post.reply_to else ''}")
            print(f"caption: {post.text.strip() or '(none)'}")
            print("image text: " + (post.image_text.replace(chr(10), ' | ')[:300] or "(nothing readable)"))
        return 0
    if args.copy_cmd == "parse":
        sig = parse_signal(" ".join(args.text))
        print(sig.summary())
        for note in sig.notes:
            print(f"  - {note}")
        return 0
    if args.copy_cmd == "import":
        from .copytrade.store import export_authors, import_telegram_export
        if not args.author and not args.channel:
            print("Who should be copied? Most active senders in the export:")
            for name, sender_id, count in export_authors(args.file)[:15]:
                print(f"  {count:>5}  {name}  ({sender_id})")
            print("Run again with --author \"<name or id>\" (or --channel if every post is his).")
            return 1
        counts = import_telegram_export(args.file, args.author)
        print(f"Imported {counts['new']} new posts ({counts['edited']} edited, "
              f"{counts['skipped']} other messages skipped) from chat {counts['chat_id']}.")
        return 0
    if args.copy_cmd == "report":
        from .copytrade.engine import CopyConfig, replay
        from .copytrade.prices import CandleFeed
        from .copytrade.store import get_live_since, load_posts
        posts = load_posts()
        if not posts:
            print("No posts stored yet. Run `copy backfill` (Telegram API) or `copy import` (export file).")
            return 1
        config = CopyConfig.from_settings(get_settings())
        if args.history:
            config.share_margin = False
            print("\nTEST ON HIS OLD POSTS (not bought: shows how copying him would have gone; "
                  "each trade sized from the account at that moment)")
        else:
            config.live_since = get_live_since()
            if config.live_since is None:
                print("The live paper account starts when you run `copy listen`. "
                      "Use `copy report --history` to test his old posts.")
                return 1
            print(f"\nLIVE PAPER ACCOUNT since {config.live_since:%Y-%m-%d %H:%M} UTC")
        from .copytrade.engine import feed_snapshot
        from .copytrade.store import load_snapshots
        feed = CandleFeed()
        _print_copy_report(replay(posts, feed.candles, config, max_leverage=feed.max_leverage,
                                  snapshot=feed_snapshot(feed, load_snapshots())))
        return 0
    if args.copy_cmd == "review":
        from .copytrade.engine import CopyConfig, feed_snapshot, replay
        from .copytrade.prices import CandleFeed
        from .copytrade.store import load_posts, load_snapshots
        posts = load_posts()
        if not posts:
            print("No posts stored yet. Run `copy backfill` first.")
            return 1
        config = CopyConfig.from_settings(get_settings())
        config.share_margin = False
        print(f"Replaying {len(posts)} stored posts against real prices (takes a minute)...")
        feed = CandleFeed()
        result = replay(posts, feed.candles, config, max_leverage=feed.max_leverage,
                        snapshot=feed_snapshot(feed, load_snapshots()))
        path = Path(__file__).resolve().parents[1] / "review.txt"
        _write_copy_review(result, path)
        print(f"Wrote {path}\nIt stays on this computer (never uploaded to GitHub). "
              "Drag it into the Claude chat to have every post checked.")
        return 0
    if args.copy_cmd == "reset-live":
        from .copytrade.store import set_live_since
        when = set_live_since()
        print(f"Live paper account restarted {when:%Y-%m-%d %H:%M} UTC: only posts from now on are bought.")
        return 0
    return 2


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

    lv = sub.add_parser("levels", help="support/resistance (default: whole watchlist)")
    lv.add_argument("symbols", nargs="*")
    lv.add_argument("--csv", help="OHLC or close-only CSV file")
    lv.set_defaults(func=cmd_levels)

    strat = sub.add_parser("strategy", help="backtest the support-bounce strategy")
    strat.add_argument("symbols", nargs="*")
    strat.add_argument("--market", nargs="+", default=["all"],
                    choices=["us", "oslo", "europe", "crypto", "all"])
    strat.add_argument("--period", default="5y", help="1y, 2y, 5y, 10y, max")
    strat.add_argument("--csv", help="OHLC or close-only CSV file")
    strat.add_argument("--min-touches", type=int, default=2)
    strat.add_argument("--min-rr", type=float, default=1.5, help="minimum reward/risk")
    strat.add_argument("--leverage", type=float, default=1.0,
                       help="fixed leverage, or the cap with --risk; max 50 (backtest only)")
    strat.add_argument("--risk", type=float, default=None,
                       help="percent of the account lost if the stop is hit, e.g. 1")
    strat.add_argument("--trailing", action="store_true",
                       help="move the stop to break-even at +1R, then trail 1 ATR below the high")
    strat.add_argument("--max-days", type=int, default=None, help="sell at the close after N days")
    strat.add_argument("--trades", action="store_true", help="list every trade")
    strat.set_defaults(func=cmd_strategy)

    cp = sub.add_parser("copy", help="paper-copy a trader's Telegram posts (read-only)")
    cp.set_defaults(func=cmd_copy)
    csub = cp.add_subparsers(dest="copy_cmd", required=True)
    csub.add_parser("login", help="log in to Telegram once (asks for the code)")
    csub.add_parser("chats", help="list your chats with their ids")
    au = csub.add_parser("authors", help="most active senders in COPY_CHAT")
    au.add_argument("--limit", type=int, default=500)
    tp = csub.add_parser("topics", help="topics in COPY_CHAT and how many posts he has in each")
    tp.add_argument("--limit", type=int, default=1000)
    bf = csub.add_parser("backfill", help="fetch his past posts")
    bf.add_argument("--days", type=int, default=90)
    bf.add_argument("--fresh", action="store_true", help="forget earlier fetched posts first")
    csub.add_parser("listen", help="save his new posts as they arrive")
    im = csub.add_parser("import", help="import a Telegram Desktop export (result.json)")
    im.add_argument("file")
    im.add_argument("--author", help="his name or id as shown in the export")
    im.add_argument("--channel", action="store_true", help="every post in the export is his")
    ig = csub.add_parser("images", help="text the bot read from his images")
    ig.add_argument("--limit", type=int, default=10)
    po = csub.add_parser("posts", help="print his stored posts and how each is understood")
    po.add_argument("--limit", type=int, default=40)
    pa = csub.add_parser("parse", help="show how a post would be understood")
    pa.add_argument("text", nargs="+")
    rp = csub.add_parser("report", help="the live paper account (or --history: test on old posts)")
    rp.add_argument("--history", action="store_true", help="test on all his stored posts instead")
    csub.add_parser("review", help="write review.txt: every post, what the bot understood and did")
    csub.add_parser("reset-live", help="restart the live paper account from now")

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
