"""
QuantGuard - Live Market Data → Transaction Stream
====================================================
Subscribes to live market APIs (Alpha Vantage, Polygon.io) and converts
real-time stock/crypto price movements into synthetic financial transactions.

Data Sources (configurable):
  • Alpha Vantage  – Real-time stock & forex quotes (free API key)
  • Polygon.io     – Stock/crypto WebSocket stream (free tier)
  • Socket stream  – Raw TCP/WebSocket event ingestion
  • Demo mode      – Simulates live events via Pathway-style demo utilities

Each price tick is transformed into a transaction event:
  - Price spikes   → high-value suspicious transactions
  - Volume bursts  → velocity fraud patterns
  - Cross-market   → geographic anomaly patterns

Usage:
  python live_data_source.py                   # Demo mode (no API key needed)
  python live_data_source.py --source alpha    # Alpha Vantage live
  python live_data_source.py --source polygon  # Polygon.io WebSocket
  python live_data_source.py --source socket   # TCP socket listener
"""

import json
import os
import time
import random
import asyncio
import argparse
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────

DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.jsonl")

# Alpha Vantage (free: 25 requests/day standard, 5/min)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

# Polygon.io (free: 5 API calls/min, delayed WebSocket)
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")

# Socket streaming
SOCKET_HOST = os.getenv("STREAM_SOCKET_HOST", "127.0.0.1")
SOCKET_PORT = int(os.getenv("STREAM_SOCKET_PORT", "9999"))

# Mapping: stock symbols → transaction categories
SYMBOL_MAP = {
    "AAPL": {"category": "Electronics", "location": "New York"},
    "GOOGL": {"category": "Technology", "location": "San Francisco"},
    "MSFT": {"category": "Technology", "location": "Seattle"},
    "AMZN": {"category": "Shopping", "location": "New York"},
    "TSLA": {"category": "Automotive", "location": "Austin"},
    "NFLX": {"category": "Entertainment", "location": "Los Angeles"},
    "JPM": {"category": "Banking", "location": "New York"},
    "RELIANCE.BSE": {"category": "Energy", "location": "Mumbai"},
    "TCS.BSE": {"category": "Technology", "location": "Mumbai"},
    "INFY": {"category": "Technology", "location": "Bangalore"},
}

USER_POOL = [f"USER_{i}" for i in range(100, 115)]

# Rolling price state for anomaly creation
_price_history = {}
_volume_history = {}


# ═══════════════════════════════════════════════════════════════════════════
#  Alpha Vantage — Live Stock / Forex Quotes
# ═══════════════════════════════════════════════════════════════════════════

class AlphaVantageSource:
    """
    Fetches real-time market data from Alpha Vantage API and transforms
    price movements into transaction events for the fraud detection pipeline.

    API docs: https://www.alphavantage.co/documentation/
    Free tier: 25 requests/day (standard), 5/min
    """

    def __init__(self, api_key=None, symbols=None):
        self.api_key = api_key or ALPHA_VANTAGE_KEY
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
        self.session = None
        self._price_cache = {}
        print(f"[LiveData] Alpha Vantage source initialized")
        print(f"[LiveData] Symbols: {', '.join(self.symbols)}")
        print(f"[LiveData] API Key: {'***' + self.api_key[-4:] if len(self.api_key) > 4 else 'demo'}")

    async def _ensure_session(self):
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()

    async def fetch_quote(self, symbol):
        """Fetch real-time quote for a single symbol."""
        import aiohttp
        await self._ensure_session()
        url = (
            f"{ALPHA_VANTAGE_BASE}?function=GLOBAL_QUOTE"
            f"&symbol={symbol}&apikey={self.api_key}"
        )
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                quote = data.get("Global Quote", {})
                if quote:
                    price = float(quote.get("05. price", 0))
                    volume = int(quote.get("06. volume", 0))
                    change_pct = float(quote.get("10. change percent", "0%").rstrip("%"))
                    return {
                        "symbol": symbol,
                        "price": price,
                        "volume": volume,
                        "change_pct": change_pct,
                        "timestamp": datetime.now().isoformat(),
                    }
        except Exception as e:
            print(f"[AlphaVantage] Error fetching {symbol}: {e}")
        return None

    async def fetch_intraday(self, symbol, interval="5min"):
        """Fetch intraday time series."""
        import aiohttp
        await self._ensure_session()
        url = (
            f"{ALPHA_VANTAGE_BASE}?function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}&interval={interval}"
            f"&apikey={self.api_key}&outputsize=compact"
        )
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                ts_key = f"Time Series ({interval})"
                series = data.get(ts_key, {})
                points = []
                for ts, vals in list(series.items())[:10]:
                    points.append({
                        "symbol": symbol,
                        "timestamp": ts,
                        "open": float(vals["1. open"]),
                        "high": float(vals["2. high"]),
                        "low": float(vals["3. low"]),
                        "close": float(vals["4. close"]),
                        "volume": int(vals["5. volume"]),
                    })
                return points
        except Exception as e:
            print(f"[AlphaVantage] Error fetching intraday {symbol}: {e}")
        return []

    def market_to_transaction(self, market_data):
        """Transform a market data point into a financial transaction event."""
        symbol = market_data.get("symbol", "AAPL")
        price = market_data.get("price", 0) or market_data.get("close", 0)
        volume = market_data.get("volume", 0)
        change_pct = market_data.get("change_pct", 0)

        mapping = SYMBOL_MAP.get(symbol, {"category": "Trading", "location": "New York"})

        # Track rolling price for anomaly detection
        if symbol not in _price_history:
            _price_history[symbol] = deque(maxlen=50)
            _volume_history[symbol] = deque(maxlen=50)
        _price_history[symbol].append(price)
        _volume_history[symbol].append(volume)

        # Transform price into transaction-like amount
        # Scale: $100-$5000 range based on price movement
        base_amount = price * random.uniform(0.5, 2.0)
        amount = round(min(max(base_amount, 50), 8000), 2)

        # Detect suspicious patterns from market data
        is_suspicious = False
        suspicious_reasons = []

        # Price spike detection (> 3% change → suspicious high-value tx)
        if abs(change_pct) > 3.0:
            is_suspicious = True
            amount = round(amount * random.uniform(2.0, 4.0), 2)
            suspicious_reasons.append(f"price_spike_{change_pct:+.1f}%")

        # Volume anomaly (compare to rolling average)
        if len(_volume_history[symbol]) > 5:
            avg_vol = sum(_volume_history[symbol]) / len(_volume_history[symbol])
            if avg_vol > 0 and volume > avg_vol * 2.5:
                is_suspicious = True
                suspicious_reasons.append(f"volume_burst_{volume/avg_vol:.1f}x")

        # Price deviation from rolling mean
        if len(_price_history[symbol]) > 5:
            import numpy as np
            prices = list(_price_history[symbol])
            mean_p = np.mean(prices)
            std_p = np.std(prices)
            if std_p > 0 and abs(price - mean_p) > 2.0 * std_p:
                is_suspicious = True
                suspicious_reasons.append(f"price_deviation_{((price-mean_p)/std_p):.1f}σ")

        # Pick user and add geographic variation
        user_id = random.choice(USER_POOL)
        locations = ["New York", "London", "Tokyo", "Mumbai", "Singapore",
                     "Dubai", "Berlin", "Sydney", "Toronto", "Paris"]
        location = mapping["location"] if random.random() < 0.6 else random.choice(locations)

        return {
            "user_id": user_id,
            "amount": amount,
            "location": location,
            "category": mapping["category"],
            "timestamp": datetime.now().isoformat(),
            "is_suspicious_flag": is_suspicious,
            "market_source": {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "change_pct": change_pct,
                "suspicious_reasons": suspicious_reasons,
            },
        }

    async def stream(self, callback, interval=12):
        """
        Continuously fetch live quotes and emit transaction events.
        Rotates through symbols to stay within rate limits.
        """
        print(f"\n[AlphaVantage] Starting live market stream (interval: {interval}s)")
        idx = 0
        while True:
            symbol = self.symbols[idx % len(self.symbols)]
            quote = await self.fetch_quote(symbol)
            if quote:
                tx = self.market_to_transaction(quote)
                await callback(tx)
                print(
                    f"  📈 {symbol}: ${quote['price']:.2f} ({quote['change_pct']:+.1f}%) "
                    f"→ {tx['user_id']} ${tx['amount']:.2f} [{tx['category']}]"
                    f"{' 🚨' if tx['is_suspicious_flag'] else ''}"
                )
            idx += 1
            await asyncio.sleep(interval)

    async def close(self):
        if self.session:
            await self.session.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Polygon.io — WebSocket Streaming
# ═══════════════════════════════════════════════════════════════════════════

class PolygonWebSocketSource:
    """
    Connects to Polygon.io real-time WebSocket feed for stock/crypto trades.

    API docs: https://polygon.io/docs/stocks/ws_stocks_trades
    Free tier: delayed data (15 min), 5 API calls/min
    """

    def __init__(self, api_key=None, symbols=None):
        self.api_key = api_key or POLYGON_KEY
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL"]
        self.ws = None
        print(f"[LiveData] Polygon.io WebSocket source initialized")

    async def stream(self, callback, max_events=None):
        """Connect to Polygon.io WebSocket and stream trade events."""
        try:
            import websockets
        except ImportError:
            print("[Polygon] websockets package required: pip install websockets")
            return

        ws_url = "wss://delayed.polygon.io/stocks"
        print(f"[Polygon] Connecting to {ws_url}")

        try:
            async with websockets.connect(ws_url) as ws:
                # Authenticate
                await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                auth_resp = await ws.recv()
                print(f"[Polygon] Auth: {auth_resp}")

                # Subscribe to trades
                subs = ",".join([f"T.{s}" for s in self.symbols])
                await ws.send(json.dumps({"action": "subscribe", "params": subs}))
                print(f"[Polygon] Subscribed: {subs}")

                count = 0
                async for message in ws:
                    try:
                        events = json.loads(message)
                        for ev in (events if isinstance(events, list) else [events]):
                            if ev.get("ev") == "T":  # Trade event
                                tx = self._trade_to_transaction(ev)
                                await callback(tx)
                                count += 1
                                if max_events and count >= max_events:
                                    return
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Polygon] WebSocket error: {e}")

    def _trade_to_transaction(self, trade_event):
        """Convert a Polygon trade event to a transaction."""
        symbol = trade_event.get("sym", "AAPL")
        price = float(trade_event.get("p", 0))
        size = int(trade_event.get("s", 0))
        mapping = SYMBOL_MAP.get(symbol, {"category": "Trading", "location": "New York"})

        amount = round(price * min(size, 10) * random.uniform(0.1, 0.5), 2)
        is_suspicious = size > 1000 or price * size > 100000

        return {
            "user_id": random.choice(USER_POOL),
            "amount": max(amount, 50),
            "location": mapping["location"],
            "category": mapping["category"],
            "timestamp": datetime.now().isoformat(),
            "is_suspicious_flag": is_suspicious,
            "market_source": {
                "symbol": symbol,
                "price": price,
                "trade_size": size,
                "exchange": trade_event.get("x", "unknown"),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Socket/Kafka Stream — TCP & WebSocket Ingestion
# ═══════════════════════════════════════════════════════════════════════════

class SocketStreamSource:
    """
    Reads transaction events from a TCP socket or WebSocket.
    Compatible with Kafka consumers that write to sockets.

    Start a producer: echo '{"amount":500,"user_id":"USER_100"}' | nc localhost 9999
    Or use Kafka → socket bridge for real-time streams.
    """

    def __init__(self, host=None, port=None):
        self.host = host or SOCKET_HOST
        self.port = port or SOCKET_PORT
        print(f"[LiveData] Socket stream source: {self.host}:{self.port}")

    async def stream(self, callback):
        """Listen for incoming transaction events on a TCP socket."""
        server = await asyncio.start_server(
            lambda r, w: self._handle_client(r, w, callback),
            self.host, self.port,
        )
        addr = server.sockets[0].getsockname()
        print(f"[Socket] Listening on {addr[0]}:{addr[1]}")
        print(f"[Socket] Send JSON transactions via: echo '{{\"amount\":500}}' | nc {addr[0]} {addr[1]}")

        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader, writer, callback):
        """Handle incoming socket connection."""
        try:
            data = await reader.read(65536)
            text = data.decode("utf-8").strip()
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    tx = self._normalize_event(event)
                    await callback(tx)
                except json.JSONDecodeError:
                    pass
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            print(f"[Socket] Client error: {e}")

    @staticmethod
    def _normalize_event(event):
        """Normalize an incoming event into our transaction schema."""
        return {
            "user_id": event.get("user_id", random.choice(USER_POOL)),
            "amount": float(event.get("amount", random.uniform(100, 3000))),
            "location": event.get("location", random.choice(
                ["New York", "Mumbai", "London", "Tokyo", "Singapore"]
            )),
            "category": event.get("category", "Trading"),
            "timestamp": event.get("timestamp", datetime.now().isoformat()),
            "is_suspicious_flag": event.get("is_suspicious_flag", False),
            "market_source": event.get("market_source", {"source": "socket"}),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Demo Mode — Pathway-Style Live Event Simulation
# ═══════════════════════════════════════════════════════════════════════════

class DemoLiveSource:
    """
    Simulates live market events using Pathway's demo utilities pattern.
    Generates realistic transaction streams with market-correlated patterns:
      - Intraday price curves (sine + noise)
      - Volume clustering
      - Cross-market correlation events
      - Flash crash / spike simulations

    No API key required. Runs entirely locally.
    """

    def __init__(self):
        self.symbols = list(SYMBOL_MAP.keys())[:6]
        self._sim_prices = {s: random.uniform(80, 400) for s in self.symbols}
        self._sim_volumes = {s: random.randint(50000, 500000) for s in self.symbols}
        self._tick = 0
        print("[LiveData] Demo mode — simulated live market events (no API key needed)")

    def _simulate_tick(self):
        """Simulate one market tick with realistic price movement."""
        import math
        self._tick += 1

        # Pick a symbol
        symbol = self.symbols[self._tick % len(self.symbols)]
        price = self._sim_prices[symbol]
        volume = self._sim_volumes[symbol]

        # Intraday pattern: sine curve + random walk
        intraday_factor = math.sin(self._tick * 0.05) * 0.02  # ±2% intraday swing
        random_walk = random.gauss(0, 0.005)  # ~0.5% random noise
        regime_shift = 0
        if random.random() < 0.03:  # 3% chance of a big move
            regime_shift = random.choice([-1, 1]) * random.uniform(0.03, 0.08)

        change_pct = (intraday_factor + random_walk + regime_shift) * 100
        new_price = price * (1 + change_pct / 100)
        self._sim_prices[symbol] = max(new_price, 5.0)

        # Volume clustering
        vol_mult = 1.0
        if abs(change_pct) > 3.0:
            vol_mult = random.uniform(2.5, 5.0)  # High volume on big moves
        elif abs(change_pct) > 1.5:
            vol_mult = random.uniform(1.5, 2.5)
        new_volume = int(volume * vol_mult * random.uniform(0.8, 1.2))
        self._sim_volumes[symbol] = max(new_volume, 10000)

        return {
            "symbol": symbol,
            "price": round(new_price, 2),
            "volume": new_volume,
            "change_pct": round(change_pct, 2),
            "timestamp": datetime.now().isoformat(),
        }

    async def stream(self, callback, interval=1.0):
        """Emit simulated market events as transaction stream."""
        print(f"[Demo] Streaming simulated market data (interval: {interval}s)")
        alpha_src = AlphaVantageSource.__new__(AlphaVantageSource)
        alpha_src.api_key = "demo"
        alpha_src.symbols = self.symbols
        alpha_src._price_cache = {}

        count = 0
        fraud_count = 0

        while True:
            tick = self._simulate_tick()
            tx = alpha_src.market_to_transaction(tick)
            await callback(tx)

            count += 1
            if tx["is_suspicious_flag"]:
                fraud_count += 1

            flag = "🚨" if tx["is_suspicious_flag"] else "✅"
            if count % 5 == 0 or tx["is_suspicious_flag"]:
                reasons = tx.get("market_source", {}).get("suspicious_reasons", [])
                reason_str = f" ({', '.join(reasons)})" if reasons else ""
                print(
                    f"  {flag} [{count:>5}] {tick['symbol']:>6} "
                    f"${tick['price']:>8.2f} ({tick['change_pct']:+5.1f}%) "
                    f"→ {tx['user_id']} ${tx['amount']:>8.2f} [{tx['category']}]"
                    f"{reason_str}"
                )

            if count % 50 == 0:
                rate = fraud_count / max(count, 1) * 100
                print(
                    f"\n  ┌{'─'*55}┐"
                    f"\n  │ 📊 LIVE MARKET STREAM: {count:,} events │ "
                    f"{fraud_count} suspicious │ Rate: {rate:.1f}%"
                    f"\n  └{'─'*55}┘\n"
                )

            await asyncio.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════
#  Transaction Writer (shared callback)
# ═══════════════════════════════════════════════════════════════════════════

async def write_transaction(tx):
    """Write a transaction event to the JSONL file (shared by all sources)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRANSACTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(tx, default=str) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Combined Multi-Source Stream
# ═══════════════════════════════════════════════════════════════════════════

class MultiSourceStream:
    """
    Aggregates multiple data sources into a single transaction stream.
    Implements the Pathway demo utility pattern for composable data ingestion.
    """

    def __init__(self, sources=None):
        self.sources = sources or []

    def add_source(self, source):
        self.sources.append(source)

    async def run(self):
        """Run all sources concurrently."""
        if not self.sources:
            print("[MultiSource] No sources configured, using Demo mode")
            self.sources.append(DemoLiveSource())

        tasks = []
        for src in self.sources:
            tasks.append(asyncio.create_task(src.stream(write_transaction)))

        print(f"[MultiSource] Running {len(tasks)} data sources concurrently")
        await asyncio.gather(*tasks, return_exceptions=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QuantGuard Live Market Data → Transaction Stream"
    )
    parser.add_argument(
        "--source",
        choices=["alpha", "polygon", "socket", "demo", "multi"],
        default="demo",
        help="Data source: alpha (Alpha Vantage), polygon (Polygon.io), "
             "socket (TCP), demo (simulated), multi (all sources)",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Stock symbols to track")
    parser.add_argument("--interval", type=float, default=None, help="Polling interval (seconds)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  QuantGuard Live Market Data Stream")
    print("=" * 60)
    print(f"  Source   : {args.source}")
    print(f"  Output   : {TRANSACTIONS_FILE}")
    print(f"  Symbols  : {args.symbols or 'default'}")
    print("  Press Ctrl+C to stop\n")

    async def run_source():
        if args.source == "alpha":
            src = AlphaVantageSource(symbols=args.symbols)
            await src.stream(write_transaction, interval=args.interval or 12)
            await src.close()

        elif args.source == "polygon":
            if not POLYGON_KEY:
                print("[Error] Set POLYGON_API_KEY in .env for Polygon.io")
                return
            src = PolygonWebSocketSource(symbols=args.symbols)
            await src.stream(write_transaction)

        elif args.source == "socket":
            src = SocketStreamSource()
            await src.stream(write_transaction)

        elif args.source == "multi":
            ms = MultiSourceStream()
            ms.add_source(DemoLiveSource())
            if ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != "demo":
                ms.add_source(AlphaVantageSource(symbols=args.symbols))
            await ms.run()

        else:  # demo
            src = DemoLiveSource()
            await src.stream(write_transaction, interval=args.interval or 1.0)

    try:
        asyncio.run(run_source())
    except KeyboardInterrupt:
        print("\n[LiveData] Stream stopped.")


if __name__ == "__main__":
    main()
