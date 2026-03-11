#!/usr/bin/env python3
"""Fetch NSE option-chain snapshots for all expiries of a symbol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

NSE_BASE_URL = "https://www.nseindia.com"
OPTION_CHAIN_PAGE_URL = f"{NSE_BASE_URL}/option-chain"
OPTION_CHAIN_API_URL = f"{NSE_BASE_URL}/api/option-chain-v3"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch full NSE option-chain snapshots (all available expiries) "
            "and save CE/PE rows to CSV."
        )
    )
    parser.add_argument(
        "--symbol",
        default="NIFTY",
        help="NSE derivative symbol. Examples: NIFTY, BANKNIFTY, RELIANCE.",
    )
    parser.add_argument(
        "--segment",
        default="indices",
        choices=["indices", "equity"],
        help="Use 'indices' for index options, 'equity' for stock options.",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=150,
        help=(
            "Days to probe for a valid seed expiry if direct expiry discovery fails. "
            "Default: 150"
        ),
    )
    parser.add_argument(
        "--request-delay-ms",
        type=int,
        default=120,
        help="Delay between API calls in milliseconds. Default: 120",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds. Default: 15",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output CSV files. Default: output",
    )
    parser.add_argument(
        "--save-raw-json",
        action="store_true",
        help="Also save one JSON file per expiry under output/raw_option_chain/",
    )
    return parser.parse_args()


def to_segment_value(segment: str) -> str:
    return "Indices" if segment == "indices" else "Equity"


def parse_expiry(expiry: str) -> datetime:
    return datetime.strptime(expiry, "%d-%b-%Y")


def sort_expiries(expiries: list[str]) -> list[str]:
    unique = list(dict.fromkeys(expiries))
    return sorted(unique, key=parse_expiry)


class NSEOptionChainClient:
    def __init__(self, timeout_seconds: float, request_delay_ms: int):
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = request_delay_ms / 1000.0
        self.session = requests.Session()
        self.html_headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.json_headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": OPTION_CHAIN_PAGE_URL,
            "X-Requested-With": "XMLHttpRequest",
        }

    def warm_up(self) -> None:
        response = self.session.get(
            OPTION_CHAIN_PAGE_URL,
            headers=self.html_headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def fetch_chain(
        self, *, symbol: str, segment: str, expiry: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, str] = {
            "type": to_segment_value(segment),
            "symbol": symbol.upper(),
        }
        if expiry:
            params["expiry"] = expiry
        response = self.session.get(
            OPTION_CHAIN_API_URL,
            params=params,
            headers=self.json_headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("NSE API did not return a JSON object.")
        return payload

    def discover_expiries(
        self, *, symbol: str, segment: str, lookahead_days: int
    ) -> list[str]:
        initial_payload = self.fetch_chain(symbol=symbol, segment=segment, expiry=None)
        initial_expiries = (
            initial_payload.get("records", {}).get("expiryDates", [])
            if isinstance(initial_payload, dict)
            else []
        )
        if initial_expiries:
            return sort_expiries(initial_expiries)

        start = date.today()
        for offset in range(lookahead_days + 1):
            candidate = (start + timedelta(days=offset)).strftime("%d-%b-%Y")
            try:
                payload = self.fetch_chain(
                    symbol=symbol, segment=segment, expiry=candidate
                )
            except requests.RequestException:
                time.sleep(self.request_delay_seconds)
                continue

            expiries = payload.get("records", {}).get("expiryDates", [])
            if expiries:
                return sort_expiries(expiries)

            time.sleep(self.request_delay_seconds)

        raise RuntimeError(
            "Could not discover expiry dates. NSE may be blocking the session or "
            "the symbol/segment is invalid."
        )


def flatten_chain_rows(
    payload: dict[str, Any], requested_expiry: str, segment: str
) -> list[dict[str, Any]]:
    records = payload.get("records", {})
    data_rows = records.get("data", [])
    snapshot_timestamp = records.get("timestamp")
    underlying_value = records.get("underlyingValue")

    flattened: list[dict[str, Any]] = []
    for strike_row in data_rows:
        strike_price = strike_row.get("strikePrice")
        for option_type in ("CE", "PE"):
            leg = strike_row.get(option_type)
            if not leg:
                continue
            # NSE may return placeholder empty legs for missing contracts.
            if not leg.get("identifier") or not leg.get("expiryDate"):
                continue
            flattened.append(
                {
                    "segment": segment,
                    "requested_expiry": requested_expiry,
                    "snapshot_timestamp": snapshot_timestamp,
                    "contract_expiry": leg.get("expiryDate"),
                    "underlying": leg.get("underlying"),
                    "underlying_value": leg.get("underlyingValue", underlying_value),
                    "strike_price": leg.get("strikePrice", strike_price),
                    "option_type": option_type,
                    "identifier": leg.get("identifier"),
                    "last_price": leg.get("lastPrice"),
                    "change": leg.get("change"),
                    "pchange": leg.get("pchange"),
                    "open_interest": leg.get("openInterest"),
                    "change_in_open_interest": leg.get("changeinOpenInterest"),
                    "pchange_in_open_interest": leg.get("pchangeinOpenInterest"),
                    "volume": leg.get("totalTradedVolume"),
                    "implied_volatility": leg.get("impliedVolatility"),
                    "total_buy_quantity": leg.get("totalBuyQuantity"),
                    "total_sell_quantity": leg.get("totalSellQuantity"),
                    "best_bid_price": leg.get("buyPrice1"),
                    "best_bid_quantity": leg.get("buyQuantity1"),
                    "best_ask_price": leg.get("sellPrice1"),
                    "best_ask_quantity": leg.get("sellQuantity1"),
                }
            )
    return flattened


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = NSEOptionChainClient(
        timeout_seconds=args.timeout_seconds,
        request_delay_ms=args.request_delay_ms,
    )

    try:
        client.warm_up()
        expiries = client.discover_expiries(
            symbol=args.symbol,
            segment=args.segment,
            lookahead_days=args.lookahead_days,
        )
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Error during expiry discovery: {exc}", file=sys.stderr)
        return 1

    all_rows: list[dict[str, Any]] = []
    raw_json_dir = output_dir / "raw_option_chain"
    if args.save_raw_json:
        raw_json_dir.mkdir(parents=True, exist_ok=True)

    for expiry in expiries:
        try:
            payload = client.fetch_chain(
                symbol=args.symbol, segment=args.segment, expiry=expiry
            )
        except requests.RequestException as exc:
            print(f"Warning: skipping expiry {expiry} ({exc})", file=sys.stderr)
            continue

        rows = flatten_chain_rows(payload=payload, requested_expiry=expiry, segment=args.segment)
        all_rows.extend(rows)

        if args.save_raw_json:
            safe_expiry = expiry.replace("-", "")
            raw_path = raw_json_dir / f"{args.symbol.upper()}_{safe_expiry}.json"
            raw_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

        time.sleep(client.request_delay_seconds)

    if not all_rows:
        print(
            "No option rows were downloaded. Try increasing --lookahead-days or rerun later.",
            file=sys.stderr,
        )
        return 1

    frame = pd.DataFrame(all_rows)
    frame["contract_expiry_sort"] = pd.to_datetime(
        frame["contract_expiry"], dayfirst=True, errors="coerce"
    )
    frame.sort_values(
        by=["contract_expiry_sort", "strike_price", "option_type"],
        inplace=True,
        ignore_index=True,
    )
    frame.drop(columns=["contract_expiry_sort"], inplace=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_upper = args.symbol.upper()
    options_csv = output_dir / f"{symbol_upper}_{args.segment}_option_chain_{run_stamp}.csv"
    expiries_csv = output_dir / f"{symbol_upper}_{args.segment}_expiries_{run_stamp}.csv"

    frame.to_csv(options_csv, index=False)
    pd.DataFrame({"expiry": expiries}).to_csv(expiries_csv, index=False)

    print(f"Symbol: {symbol_upper} ({args.segment})")
    print(f"Expiries found: {len(expiries)}")
    print(f"Rows saved: {len(frame)}")
    print(f"Options CSV: {options_csv.resolve()}")
    print(f"Expiries CSV: {expiries_csv.resolve()}")
    if args.save_raw_json:
        print(f"Raw JSON directory: {raw_json_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
