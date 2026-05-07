"""
Test: multi-tick simulation of the LOB matching engine.

Tick 0 — seed the book with resting limit orders on both sides (no crosses).
Tick 1 — aggressive orders that cross; partial fills; market orders.
Tick 2 — unfilled orders from tick 1 carry over; new wave of orders.
Tick 3 — thin tick: only 2 orders, tests empty-side behaviour.
Tick 4 — final flush: market orders sweep residual resting liquidity.

After all ticks: print full trade log, final book state, and price history.
"""

import json
from engine import Order, OrderBook, OrderType, Side


def make_limit(side: Side, qty: float, price: float) -> Order:
    return Order(side, OrderType.LIMIT, qty=qty, price=price)


def make_market(side: Side, qty: float) -> Order:
    return Order(side, OrderType.MARKET, qty=qty, price=None)


def run_test() -> None:
    book = OrderBook(symbol="AAPL")

    # -----------------------------------------------------------------------
    # TICK 0 — seed resting liquidity; no crosses expected.
    # -----------------------------------------------------------------------
    tick0_orders = [
        # Bids: 98.00 – 99.50 (below any ask)
        make_limit(Side.BUY,  qty=10, price=99.50),
        make_limit(Side.BUY,  qty=8,  price=99.00),
        make_limit(Side.BUY,  qty=5,  price=99.00),   # same price, time 2nd
        make_limit(Side.BUY,  qty=12, price=98.50),
        make_limit(Side.BUY,  qty=6,  price=98.00),
        # Asks: 101.00 – 103.00 (above any bid)
        make_limit(Side.SELL, qty=8,  price=101.00),
        make_limit(Side.SELL, qty=10, price=101.50),
        make_limit(Side.SELL, qty=4,  price=102.00),
        make_limit(Side.SELL, qty=7,  price=102.50),
        make_limit(Side.SELL, qty=5,  price=103.00),
    ]
    s0 = book.advance_tick(tick0_orders)
    print(f"Tick 0 closed: {s0}")

    # -----------------------------------------------------------------------
    # TICK 1 — aggressive orders cross the spread; partial fills occur.
    # -----------------------------------------------------------------------
    tick1_orders = [
        # Aggressive buys that cross into asks at 101.00 and 101.50
        make_limit(Side.BUY,  qty=5,  price=101.50),   # fills 5 of the 8 @ 101.00
        make_limit(Side.BUY,  qty=6,  price=101.50),   # fills remaining 3 @ 101.00 + 3 @ 101.50
        # Aggressive sell that crosses into bids at 99.50
        make_limit(Side.SELL, qty=7,  price=99.50),    # fills 7 of the 10 @ 99.50
        # Market orders
        make_market(Side.BUY,  qty=4),                 # sweeps next best ask (101.50 residual)
        make_market(Side.SELL, qty=3),                 # sweeps next best bid (99.50 residual)
        # New resting orders (won't cross after above fills)
        make_limit(Side.BUY,  qty=9,  price=100.00),
        make_limit(Side.SELL, qty=6,  price=103.50),
    ]
    s1 = book.advance_tick(tick1_orders)
    print(f"Tick 1 closed: {s1}")

    # -----------------------------------------------------------------------
    # TICK 2 — carryover orders are in the book; fresh wave tightens spread.
    # -----------------------------------------------------------------------
    tick2_orders = [
        make_limit(Side.BUY,  qty=15, price=100.50),   # crosses asks still resting at 100.xx? no
        make_limit(Side.BUY,  qty=4,  price=101.00),   # crosses into residual 101.50 ask
        make_limit(Side.SELL, qty=10, price=99.50),    # crosses into bids carried from tick 1
        make_limit(Side.SELL, qty=5,  price=100.00),   # crosses resting 100.00 bids
        make_limit(Side.BUY,  qty=3,  price=99.00),    # won't cross — rests
        make_limit(Side.SELL, qty=3,  price=104.00),   # won't cross — rests
    ]
    s2 = book.advance_tick(tick2_orders)
    print(f"Tick 2 closed: {s2}")

    # -----------------------------------------------------------------------
    # TICK 3 — thin tick: only 2 orders, tests zero-trade case.
    # -----------------------------------------------------------------------
    tick3_orders = [
        make_limit(Side.BUY,  qty=2, price=97.00),   # deep bid, no cross
        make_limit(Side.SELL, qty=2, price=105.00),  # deep ask, no cross
    ]
    s3 = book.advance_tick(tick3_orders)
    print(f"Tick 3 closed: {s3}")

    # -----------------------------------------------------------------------
    # TICK 4 — market orders sweep all remaining resting liquidity.
    # -----------------------------------------------------------------------
    tick4_orders = [
        make_market(Side.BUY,  qty=50),   # eats all asks
        make_market(Side.SELL, qty=50),   # eats all bids
    ]
    s4 = book.advance_tick(tick4_orders)
    print(f"Tick 4 closed: {s4}")

    # -----------------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------------
    book.print_trade_log()
    book.print_book(levels=8)
    book.print_price_history()

    total_volume = sum(t.qty for t in book.trade_log)
    avg_price = (
        sum(t.price * t.qty for t in book.trade_log) / total_volume
        if total_volume else 0.0
    )
    print(f"  Session VWAP      : {avg_price:.4f}")
    print(f"  Session volume    : {total_volume:.2f}")
    print(f"  Last clearing px  : {book.last_clearing_price:.2f}")
    print(f"  Ticks completed   : {book.current_tick}")
    print()

    # -----------------------------------------------------------------------
    # JSON export — write file and do a quick structural sanity check.
    # -----------------------------------------------------------------------
    out_path = "session.json"
    raw = book.to_json(path=out_path)
    parsed = json.loads(raw)

    print(f"JSON written to: {out_path}")
    print(f"  Top-level keys    : {list(parsed.keys())}")
    print(f"  tick_history len  : {len(parsed['tick_history'])}")
    print(f"  trade_log len     : {len(parsed['trade_log'])}")
    print(f"  order_history len : {len(parsed['order_history'])}")
    print(f"  bids depth        : {len(parsed['order_book']['bids'])} levels")
    print(f"  asks depth        : {len(parsed['order_book']['asks'])} levels")
    print(f"  mid_price         : {parsed['mid_price']}")
    print(f"  last_clearing_px  : {parsed['last_clearing_price']}")

    # Spot-check: every trade_id in trade_log is unique.
    ids = [t["trade_id"] for t in parsed["trade_log"]]
    assert len(ids) == len(set(ids)), "duplicate trade_ids in export"

    # Spot-check: order statuses are valid strings.
    valid_statuses = {"OPEN", "PARTIAL", "FILLED", "CANCELLED"}
    for o in parsed["order_history"]:
        assert o["status"] in valid_statuses, f"bad status: {o['status']}"

    print("\nAll export assertions passed.")


if __name__ == "__main__":
    run_test()
