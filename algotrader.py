#!/usr/bin/env python3
"""
AlgoTrader CLI — quantitative trading for Indian equity markets.

    ./run.sh universe                      fetch index constituents from NSE
    ./run.sh login                         start Zerodha auth (daily)
    ./run.sh token <request_token>         finish Zerodha auth
    ./run.sh warm --days 90                pre-download history into the cache
    ./run.sh backtest --start 2026-01-01   simulate on historical data
    ./run.sh run --mode paper              trade with simulated fills, real prices
    ./run.sh run --mode live               trade with real money
    ./run.sh report                        post-trade analytics
    ./run.sh check <file.py>               scan trading code for known failure patterns
    ./run.sh wizard                        generate a new bot

Read NUANCES.md before going live. Every check in this codebase is there because
its absence cost someone money.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_DIR))


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[90m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    width = 66
    print(f"\n{Colors.CYAN}╔{'═' * width}╗{Colors.ENDC}")
    padding = (width - len(text)) // 2
    print(f"{Colors.CYAN}║{' ' * padding}{Colors.BOLD}{text}{Colors.ENDC}"
          f"{' ' * (width - padding - len(text))}{Colors.CYAN}║{Colors.ENDC}")
    print(f"{Colors.CYAN}╚{'═' * width}╝{Colors.ENDC}\n")


def ok(message: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")


def warn(message: str) -> None:
    print(f"{Colors.YELLOW}!{Colors.ENDC} {message}")


def fail(message: str) -> None:
    print(f"{Colors.RED}✗{Colors.ENDC} {message}")


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        for candidate in (Path.cwd() / ".env", SKILL_DIR / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                return
    except ImportError:
        pass


DEFAULT_CONFIG_PATH = SKILL_DIR / "config" / "default.json"


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Merge a user config over the shipped defaults."""
    config: dict[str, Any] = {}

    if DEFAULT_CONFIG_PATH.exists():
        config.update(json.loads(DEFAULT_CONFIG_PATH.read_text()))

    if path:
        user_path = Path(path)
        if not user_path.exists():
            raise SystemExit(f"config not found: {user_path}")
        config.update(json.loads(user_path.read_text()))

    return config


# ============================================================================
# universe
# ============================================================================

def cmd_universe(args: argparse.Namespace) -> int:
    from engine.universe import INDEX_FILES, UniverseFetchError, fetch_index_constituents, save_universe

    print_header("UNIVERSE FETCHER")

    indices = [i.strip() for i in args.indices.split(",")] if args.indices else ["nifty50"]
    out_dir = Path(args.out)
    failures = 0

    for index in indices:
        if index not in INDEX_FILES:
            fail(f"unknown index {index!r}. Known: {', '.join(sorted(INDEX_FILES))}")
            failures += 1
            continue

        try:
            data = fetch_index_constituents(index)
        except UniverseFetchError as exc:
            fail(f"{index}: {exc}")
            failures += 1
            continue

        path = save_universe(data, out_dir / f"{index}.json")
        ok(f"{data['index']}: {data['count']} stocks → {path}")

        sample = ", ".join(s["symbol"] for s in data["stocks"][:5])
        print(f"  {Colors.DIM}{sample}, ...{Colors.ENDC}")

    if failures:
        print(f"\n{Colors.YELLOW}{failures} index/indices failed. "
              f"No stale fallback was substituted — refetch before trading.{Colors.ENDC}")
        return 1

    print(f"\n{Colors.GREEN}Done.{Colors.ENDC} Universe files are live NSE data, "
          f"refetch after every index rebalance.")
    return 0


# ============================================================================
# auth
# ============================================================================

def cmd_login(args: argparse.Namespace) -> int:
    load_env()
    from engine.broker import AuthError, KiteBroker

    print_header("ZERODHA LOGIN")
    try:
        url = KiteBroker.login_url()
    except AuthError as exc:
        fail(str(exc))
        print("\nCreate a .env file with:\n  KITE_API_KEY=...\n  KITE_API_SECRET=...")
        return 1

    if args.manual:
        print("1. Open this URL and log in:\n")
        print(f"   {Colors.CYAN}{url}{Colors.ENDC}\n")
        print("2. After login you land on a redirect URL containing request_token=XXXX")
        print("3. Run:\n")
        print(f"   {Colors.BOLD}./run.sh token XXXX{Colors.ENDC}\n")
        warn("Kite tokens expire daily around 07:30 IST — this is a daily ritual.")
        return 0

    return _login_with_listener(url, args.port)


def _login_with_listener(url: str, port: int) -> int:
    """
    Catch the redirect automatically.

    The manual flow asks you to copy a request_token out of a browser error page.
    That token is single-use and expires in minutes, so every fumble — a reload,
    a stale URL, a truncated paste — burns it and produces an error that looks
    like a credentials problem. Listening on the redirect URL removes the whole
    class of failure: the token never touches the clipboard.
    """
    import socket
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    from engine.broker import KiteBroker

    # macOS runs AirPlay Receiver on 5000 by default, which will answer the
    # browser instead of us and silently swallow the token.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.4)
    busy = probe.connect_ex(("127.0.0.1", port)) == 0
    probe.close()

    if busy:
        fail(f"port {port} is already in use")
        print("  On macOS this is usually AirPlay Receiver:")
        print("    System Settings -> General -> AirDrop & Handoff -> AirPlay Receiver: off")
        print(f"  Or pick another port and set the app's Redirect URL to match:")
        print(f"    ./run.sh login --port 5555   (redirect URL http://127.0.0.1:5555/)")
        print(f"  Or fall back to copy-paste:  ./run.sh login --manual")
        return 1

    captured: dict[str, str] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            params = parse_qs(urlparse(self.path).query)
            token = (params.get("request_token") or [""])[0]
            status = (params.get("status") or [""])[0]

            if token:
                captured["request_token"] = token
                captured["status"] = status
                body = (
                    b"<html><body style='font-family:system-ui;padding:3rem'>"
                    b"<h2>Token received.</h2>"
                    b"<p>You can close this tab and return to the terminal.</p>"
                    b"</body></html>"
                )
            else:
                captured["error"] = self.path
                body = (
                    b"<html><body style='font-family:system-ui;padding:3rem'>"
                    b"<h2>No request_token in the redirect.</h2>"
                    b"<p>Check the terminal for details.</p>"
                    b"</body></html>"
                )

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, *args: Any) -> None:
            pass  # keep the console clean

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Listening on {Colors.BOLD}http://127.0.0.1:{port}/{Colors.ENDC} "
          f"for the Kite redirect.\n")
    print("Opening your browser. If it does not open, use this URL:\n")
    print(f"   {Colors.CYAN}{url}{Colors.ENDC}\n")
    print(f"{Colors.DIM}Log in as the Zerodha account the app is bound to. "
          f"TOTP is required for API logins.{Colors.ENDC}\n")

    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless or no browser configured
        pass

    print("Waiting (Ctrl+C to cancel)...")
    try:
        received = done.wait(timeout=300)
    except KeyboardInterrupt:
        server.shutdown()
        print("\nCancelled.")
        return 130

    server.shutdown()

    if not received:
        fail("timed out after 5 minutes")
        return 1

    if "request_token" not in captured:
        fail("the redirect carried no request_token")
        print(f"  raw redirect path: {captured.get('error', '?')}")
        print("\n  That means Kite did not mint a token. Usual causes:")
        print("    - logged in as a different Zerodha account than the app is bound to")
        print("    - TOTP 2FA not enrolled on the account")
        print("    - a stale login URL (always start from ./run.sh login)")
        return 1

    print()
    ok(f"request_token captured ({captured['status'] or 'no status'})")

    try:
        KiteBroker.exchange_session(captured["request_token"])
    except Exception as exc:  # noqa: BLE001
        fail(f"token exchange failed: {exc}")
        message = str(exc).lower()
        if "invalid" in message or "expired" in message:
            # Request tokens are single-use and short-lived, so this is usually a
            # replayed or stale token rather than a credentials problem.
            print("\n  The request_token was rejected. They are single-use and expire")
            print("  within minutes, so this happens if the login was retried, the")
            print("  page reloaded, or the token reused. Run ./run.sh login again.")
        else:
            print("\n  The token was accepted but the exchange failed — check that")
            print("  KITE_API_SECRET in .env matches this app's secret exactly.")
        return 1

    ok("access token saved to .kite_session.json (mode 600)")
    print(f"  {Colors.DIM}valid until ~07:30 IST tomorrow{Colors.ENDC}\n")

    try:
        broker = KiteBroker()
        print(f"  Connected as {Colors.BOLD}{broker.profile.get('user_name')}{Colors.ENDC} "
              f"({broker.profile.get('user_id')})")
        print(f"  Available margin: Rs {broker.available_margin():,.2f}")
    except Exception as exc:  # noqa: BLE001
        warn(f"saved, but a verification call failed: {exc}")

    return 0


def cmd_token(args: argparse.Namespace) -> int:
    load_env()
    from engine.broker import AuthError, KiteBroker

    try:
        KiteBroker.exchange_session(args.request_token)
    except (AuthError, Exception) as exc:  # noqa: BLE001
        fail(f"token exchange failed: {exc}")
        return 1

    ok("access token saved to .kite_session.json (mode 600)")
    print(f"  {Colors.DIM}valid until ~07:30 IST tomorrow{Colors.ENDC}")
    return 0


# ============================================================================
# warm cache
# ============================================================================

def cmd_warm(args: argparse.Namespace) -> int:
    load_env()
    from engine.broker import BrokerError, KiteBroker
    from engine.data import DataManager
    from engine.universe import load_universe

    print_header("CACHE WARM")
    config = load_config(args.config)

    try:
        broker = KiteBroker()
    except BrokerError as exc:
        fail(str(exc))
        return 1

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else load_universe(args.universe, max_age_days=None)
    )
    if args.limit:
        symbols = symbols[: args.limit]

    data = DataManager(broker, interval=args.interval,
                       cache_dir=config.get("cache_dir", ".cache") + "/ohlcv")

    print(f"Downloading {args.days}d of {args.interval} bars for {len(symbols)} symbols")
    print(f"{Colors.DIM}Kite allows 3 historical requests/second — this is rate limited.{Colors.ENDC}\n")

    counts = data.warm_cache(symbols, days=args.days)
    total = sum(counts.values())
    failed = [s for s, n in counts.items() if n == 0]

    print()
    ok(f"cached {total:,} bars across {len(counts) - len(failed)} symbols")
    if failed:
        warn(f"{len(failed)} symbols returned nothing: {', '.join(failed[:10])}")
    return 0


# ============================================================================
# backtest
# ============================================================================

def cmd_backtest(args: argparse.Namespace) -> int:
    load_env()
    from engine import backtest
    from engine.analytics import format_report
    from engine.broker import BrokerError, KiteBroker
    from engine.data import DataManager
    from engine.universe import load_universe

    print_header("BACKTEST")
    config = load_config(args.config)

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) if args.end else datetime.now()

    try:
        broker = KiteBroker()
    except BrokerError as exc:
        fail(str(exc))
        print("\nBacktesting needs historical data from Kite. Run ./run.sh login first.")
        return 1

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else load_universe(args.universe, max_age_days=None)
    )
    if args.limit:
        symbols = symbols[: args.limit]

    interval = args.interval or config.get("interval", "minute")
    data = DataManager(broker, interval=interval,
                       cache_dir=config.get("cache_dir", ".cache") + "/ohlcv")

    print(f"Strategy   {args.strategy}")
    print(f"Universe   {len(symbols)} symbols")
    print(f"Period     {start:%Y-%m-%d} → {end:%Y-%m-%d}  ({interval} bars)")
    print(f"Capital    Rs {args.capital:,.0f}\n")

    frames: dict[str, Any] = {}
    tick_sizes: dict[str, float] = {}

    for index, symbol in enumerate(symbols, 1):
        try:
            frame = data.with_indicators(symbol, start, end, config)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{index}/{len(symbols)}] {symbol}: skipped — {exc}")
            continue

        if frame.height:
            frames[symbol] = frame
            tick_sizes[symbol] = broker.tick_size(symbol)
        print(f"  [{index}/{len(symbols)}] {symbol}: {frame.height:,} bars", end="\r")

    print(" " * 70, end="\r")
    if not frames:
        fail("no data — run ./run.sh warm first")
        return 1

    ok(f"loaded {sum(f.height for f in frames.values()):,} bars\n")

    result = backtest.run(
        frames,
        config,
        strategy=args.strategy,
        starting_capital=args.capital,
        slippage_bps=args.slippage,
        tick_sizes=tick_sizes,
        intraday=interval != "day",
    )

    print(f"{Colors.BOLD}Results{Colors.ENDC}")
    print(format_report(result.metrics))

    if result.rejections:
        print(f"\n{Colors.BOLD}Why signals were not taken{Colors.ENDC}")
        for reason, count in list(result.rejections.items())[:8]:
            print(f"    {reason:45s} {count:>6,}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if result.trades:
        trades_path = out_dir / f"trades_{args.strategy}_{stamp}.parquet"
        result.trades_frame().write_parquet(trades_path)
        metrics_path = out_dir / f"metrics_{args.strategy}_{stamp}.json"
        metrics_path.write_text(json.dumps(
            {"metrics": result.metrics, "config": result.config,
             "rejections": result.rejections}, indent=2, default=str))
        print(f"\n  saved {trades_path}")
        print(f"  saved {metrics_path}")

    print(f"\n{Colors.DIM}Backtest results are not a forecast. Paper trade before going live.{Colors.ENDC}")
    return 0


# ============================================================================
# live / paper run
# ============================================================================

def cmd_run(args: argparse.Namespace) -> int:
    load_env()
    from engine.broker import BrokerError
    from engine.runner import PreflightError, TradingRunner

    print_header(f"{args.mode.upper()} TRADING")
    config = load_config(args.config)
    config["capital"] = args.capital or config.get("capital", 1_000_000.0)
    if args.strategy:
        config["strategy"] = args.strategy
    if args.universe:
        config["universe_file"] = args.universe

    if args.mode == "live":
        print(f"{Colors.RED}{Colors.BOLD}  LIVE MODE — REAL ORDERS, REAL MONEY{Colors.ENDC}\n")
        print(f"  Strategy   {config.get('strategy')}")
        print(f"  Capital    Rs {config['capital']:,.0f}")
        print(f"  Risk/trade {config.get('risk_pct', 1.0)}%  "
              f"(max {config.get('max_positions', 5)} positions, "
              f"{config.get('max_portfolio_heat_pct', 5.0)}% portfolio heat)")
        print(f"  Daily stop {config.get('daily_loss_limit_pct', 3.0)}% "
              f"or {config.get('max_consecutive_losses', 3)} losses in a row\n")

        confirmation = input(f"  Type {Colors.BOLD}LIVE{Colors.ENDC} to confirm: ").strip()
        if confirmation != "LIVE":
            print("  Cancelled.")
            return 1
        print()

    try:
        runner = TradingRunner(config, mode=args.mode, state_dir=args.state_dir)
    except (PreflightError, BrokerError, Exception) as exc:  # noqa: BLE001
        fail(str(exc))
        return 1

    try:
        runner.run()
    except PreflightError as exc:
        fail(f"preflight failed: {exc}")
        print(f"\n{Colors.DIM}Preflight failures are not warnings. Fix the cause.{Colors.ENDC}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    return 0


# ============================================================================
# report
# ============================================================================

def cmd_report(args: argparse.Namespace) -> int:
    from engine.analytics import daily_report, format_report, signal_attribution

    print_header("POST-TRADE REPORT")
    config = load_config(args.config)

    metrics = daily_report(
        args.trades, starting_capital=config.get("capital", 1_000_000.0), day=args.date
    )
    if "error" in metrics:
        fail(metrics["error"])
        return 1

    print(f"{Colors.BOLD}{metrics.get('date', 'latest')}{Colors.ENDC}")
    print(format_report(metrics))

    if metrics.get("by_hour"):
        print(f"\n{Colors.BOLD}By entry hour{Colors.ENDC}")
        for hour, stats in metrics["by_hour"].items():
            print(f"    {hour}:00   {stats['trades']:3d} trades  "
                  f"win {stats['win_rate']:.0%}  Rs {stats['pnl']:>10,.0f}")

    attribution = signal_attribution(args.signals, args.trades)
    if attribution.get("rejection_reasons"):
        print(f"\n{Colors.BOLD}Signals blocked by risk limits{Colors.ENDC}")
        for reason, count in list(attribution["rejection_reasons"].items())[:8]:
            print(f"    {reason:45s} {count:>5}")

    if attribution.get("factor_attribution"):
        print(f"\n{Colors.BOLD}Factor attribution{Colors.ENDC}")
        for factor, stats in attribution["factor_attribution"].items():
            without = stats["win_rate_without"]
            comparison = f" vs {without:.0%} without" if without is not None else ""
            print(f"    {factor:16s} win {stats['win_rate_with']:.0%}{comparison}  "
                  f"(n={stats['trades_with']})")
        print(f"    {Colors.DIM}{attribution.get('note', '')}{Colors.ENDC}")

    return 0


# ============================================================================
# check — static scan for known failure patterns
# ============================================================================

CHECKS = [
    {
        "id": "tick_size",
        "nuance": 1,
        "severity": "critical",
        "pattern": r"place_order\s*\([^)]*price\s*=",
        "absent": r"round_to_tick|round\s*\(\s*\w+\s*/\s*tick",
        "message": "order price may not be tick-aligned",
        "fix": "price = round_to_tick(price, tick_size)  # 90% of rejections",
    },
    {
        "id": "reconciliation",
        "nuance": 2,
        "severity": "critical",
        "pattern": r"kite\.|KiteConnect",
        "absent": r"reconcil|positions\s*\(\s*\)",
        "message": "no position reconciliation against the broker on startup",
        "fix": "reconcile broker positions at startup — broker is the source of truth",
    },
    {
        "id": "sl_cancel_first",
        "nuance": 3,
        "severity": "critical",
        "pattern": r"cancel_order\s*\([^)]*\)[\s\S]{0,400}?place_order\s*\([^)]*SL",
        "message": "cancel-then-place stop loss leaves the position naked if the place fails",
        "fix": "place the new SL first, then cancel the old one",
    },
    {
        "id": "vwap_reset",
        "nuance": 4,
        "severity": "critical",
        "pattern": r"vwap",
        "absent": r"over\s*\(\s*[\"']session_date|groupby|group_by|reset|\.dt\.date",
        "message": "VWAP may not reset daily",
        "fix": "anchor VWAP per session: cum_sum().over('session_date')",
    },
    {
        "id": "symbol_cooldown",
        "nuance": 5,
        "severity": "high",
        "pattern": r"place_order|generate_signal|def\s+\w*signal",
        "absent": r"cooldown|last_exit|can_trade_symbol",
        "message": "no symbol cooldown — risk of re-entering the name that just stopped you out",
        "fix": "block re-entry for 45 minutes after an exit",
    },
    {
        "id": "candle_complete",
        "nuance": 6,
        "severity": "high",
        "pattern": r"iloc\[-1\]|row\(-1|\[-1\]\[[\"']close",
        "absent": r"is_candle_complete|complete|row\(-2",
        "message": "signals may be evaluated on an incomplete candle",
        "fix": "use the last COMPLETE candle (row -2 live), or check completion with a 500ms buffer",
    },
    {
        "id": "opening_balance",
        "nuance": 7,
        "severity": "high",
        "pattern": r"opening_balance",
        "message": "margin read from opening_balance ignores margin already blocked",
        "fix": "use margins('equity')['net']",
    },
    {
        "id": "adx_direction",
        "nuance": 8,
        "severity": "medium",
        "pattern": r"adx[\"'\]\s\)]*[<>]=?\s*\d+[\s\S]{0,120}?(direction|signal)\s*=\s*[\"']?(LONG|SHORT|BUY|SELL)",
        "message": "ADX may be used for direction — it is directionless",
        "fix": "ADX filters strength; take direction from EMA or +DI/-DI",
    },
    {
        "id": "json_ohlcv",
        "nuance": 10,
        "severity": "medium",
        "pattern": r"json\.(dump|load)\w*\([^)]*(?:candle|ohlc|historical|bars)",
        "message": "OHLCV stored as JSON — Parquet is ~28x faster to load",
        "fix": "df.write_parquet(path)",
    },
    {
        "id": "iterrows",
        "nuance": 11,
        "severity": "medium",
        "pattern": r"\.iterrows\s*\(|for\s+\w+\s+in\s+range\s*\(\s*len\s*\(\s*(?:df|candles)",
        "message": "row-by-row iteration over price data",
        "fix": "vectorise with Polars expressions (~37x)",
    },
    {
        "id": "risk_per_trade",
        "nuance": 19,
        "severity": "high",
        "pattern": r"quantity\s*=\s*(?:int\s*\()?\s*(?:capital|funds|balance)\s*(?://|/)",
        "absent": r"stop_loss|risk_per_share|entry\s*-\s*stop",
        "message": "position size derived from capital, not from stop distance",
        "fix": "quantity = (capital * risk_pct/100) / abs(entry - stop)",
    },
    {
        "id": "bare_except",
        "nuance": None,
        "severity": "medium",
        "pattern": r"except\s*:\s*(?:\n|#)|except\s+Exception\s*:\s*(?:\n\s*pass)",
        "message": "silent exception handler — a swallowed order failure is invisible",
        "fix": "log the exception; never pass silently around order calls",
    },
]


def cmd_check(args: argparse.Namespace) -> int:
    import re

    print_header("CODE CHECK")

    target = Path(args.path)
    files = (
        sorted(target.rglob("*.py"))
        if target.is_dir()
        else [target] if target.exists() else []
    )
    files = [f for f in files if not any(
        part in {"venv", ".venv", "__pycache__", "node_modules", ".git", "engine"}
        for part in f.parts
    )]

    if not files:
        fail(f"no Python files found at {target}")
        return 1

    print(f"Scanning {len(files)} file(s) against {len(CHECKS)} known failure patterns\n")

    findings: list[dict[str, Any]] = []

    for file_path in files:
        try:
            source = file_path.read_text(errors="ignore")
        except OSError:
            continue

        for check in CHECKS:
            match = re.search(check["pattern"], source, re.IGNORECASE)
            if not match:
                continue

            # 'absent' means: the risky pattern is present AND the mitigation is not.
            if check.get("absent") and re.search(check["absent"], source, re.IGNORECASE):
                continue

            line_number = source[: match.start()].count("\n") + 1
            findings.append({
                "file": file_path,
                "line": line_number,
                "check": check,
            })

    if not findings:
        ok("no known failure patterns detected")
        print(f"\n{Colors.DIM}This is a heuristic scan, not a proof of correctness. "
              f"Work through the NUANCES.md checklist before going live.{Colors.ENDC}")
        return 0

    order = {"critical": 0, "high": 1, "medium": 2}
    findings.sort(key=lambda f: order.get(f["check"]["severity"], 3))

    colors = {"critical": Colors.RED, "high": Colors.YELLOW, "medium": Colors.CYAN}

    for finding in findings:
        check = finding["check"]
        color = colors.get(check["severity"], "")
        nuance = f" (NUANCE #{check['nuance']})" if check["nuance"] else ""

        print(f"{color}{check['severity'].upper():8s}{Colors.ENDC} "
              f"{finding['file']}:{finding['line']}{nuance}")
        print(f"         {check['message']}")
        print(f"         {Colors.DIM}fix: {check['fix']}{Colors.ENDC}\n")

    critical = sum(1 for f in findings if f["check"]["severity"] == "critical")
    print(f"{len(findings)} finding(s), {critical} critical")
    print(f"\n{Colors.DIM}Heuristic scan — verify each finding by reading the code. "
          f"False positives are expected.{Colors.ENDC}")
    return 1 if critical else 0


# ============================================================================
# wizard
# ============================================================================

def cmd_wizard(args: argparse.Namespace) -> int:
    from wizard import run_wizard

    return run_wizard(SKILL_DIR)


# ============================================================================
# entry point
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algotrader",
        description="Quantitative trading for Indian equity markets (NSE / Zerodha Kite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    universe = subparsers.add_parser("universe", help="fetch index constituents from NSE")
    universe.add_argument("--indices", default="nifty50",
                          help="comma separated: nifty50,nifty100,midcap150,smallcap250")
    universe.add_argument("--out", default="universe", help="output directory")
    universe.set_defaults(func=cmd_universe)

    login = subparsers.add_parser(
        "login", help="Zerodha auth: opens the browser and catches the redirect"
    )
    login.add_argument("--port", type=int, default=5000,
                       help="local port matching the app's Redirect URL (default 5000)")
    login.add_argument("--manual", action="store_true",
                       help="just print the URL; paste the token via ./run.sh token")
    login.set_defaults(func=cmd_login)

    token = subparsers.add_parser("token", help="exchange a request_token for an access token")
    token.add_argument("request_token")
    token.set_defaults(func=cmd_token)

    warm = subparsers.add_parser("warm", help="pre-download history into the Parquet cache")
    warm.add_argument("--universe", default="universe/nifty50.json")
    warm.add_argument("--symbols", help="comma separated, overrides --universe")
    warm.add_argument("--days", type=int, default=90)
    warm.add_argument("--interval", default="minute")
    warm.add_argument("--limit", type=int, help="only the first N symbols")
    warm.add_argument("--config")
    warm.set_defaults(func=cmd_warm)

    backtest_parser = subparsers.add_parser("backtest", help="simulate on historical data")
    backtest_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    backtest_parser.add_argument("--end", help="YYYY-MM-DD (default: today)")
    backtest_parser.add_argument("--strategy", default="fortress",
                                 choices=["fortress", "momentum", "mean_reversion"])
    backtest_parser.add_argument("--universe", default="universe/nifty50.json")
    backtest_parser.add_argument("--symbols", help="comma separated, overrides --universe")
    backtest_parser.add_argument("--limit", type=int)
    backtest_parser.add_argument("--interval", help="minute, 5minute, 15minute, day")
    backtest_parser.add_argument("--capital", type=float, default=1_000_000.0)
    backtest_parser.add_argument("--slippage", type=float, default=5.0, help="basis points")
    backtest_parser.add_argument("--out", default="backtests")
    backtest_parser.add_argument("--config")
    backtest_parser.set_defaults(func=cmd_backtest)

    run_parser = subparsers.add_parser("run", help="trade (paper or live)")
    run_parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    run_parser.add_argument("--config")
    run_parser.add_argument("--universe")
    run_parser.add_argument("--strategy")
    run_parser.add_argument("--capital", type=float)
    run_parser.add_argument("--state-dir", default="state")
    run_parser.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="post-trade analytics")
    report.add_argument("--trades", default="logs/trades.jsonl")
    report.add_argument("--signals", default="logs/signals.jsonl")
    report.add_argument("--date", help="YYYY-MM-DD (default: most recent)")
    report.add_argument("--config")
    report.set_defaults(func=cmd_report)

    check = subparsers.add_parser("check", help="scan trading code for known failure patterns")
    check.add_argument("path", nargs="?", default=".")
    check.set_defaults(func=cmd_check)

    wizard = subparsers.add_parser("wizard", help="generate a new trading bot")
    wizard.set_defaults(func=cmd_wizard)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        return cmd_wizard(args)

    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Cancelled.{Colors.ENDC}")
        raise SystemExit(130)
