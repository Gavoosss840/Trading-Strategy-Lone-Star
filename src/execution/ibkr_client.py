"""
IBKR TWS Client
────────────────
Manages connection to Interactive Brokers Trader Workstation (TWS) or
IB Gateway using ib_insync.

Paper trading ports  (API must be enabled in TWS/Gateway settings):
  TWS:        7497
  IB Gateway: 4002

Live trading ports:
  TWS:        7496
  IB Gateway: 4001

Usage:
    client = IBKRClient.from_config(cfg["ibkr"])
    with client:
        nav           = client.get_nav()
        buying_power  = client.get_buying_power()
        positions     = client.get_positions()
        client.place_bracket_order("FCX", "LONG", 10, 45.20, 48.50, 43.80)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

try:
    from ib_insync import IB, Contract, util
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False


DEFAULT_PAPER_PORT_TWS     = 7497
DEFAULT_PAPER_PORT_GATEWAY = 4002
DEFAULT_LIVE_PORT_TWS      = 7496
DEFAULT_LIVE_PORT_GATEWAY  = 4001


class IBKRConnectionError(RuntimeError):
    pass


class IBKRClient:
    """
    Thin wrapper around ib_insync.IB for Lone Star order routing.

    Supports paper and live modes. Switch by changing ibkr.mode in config.yaml
    or by passing --ibkr-mode live on the CLI (no code change required).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PAPER_PORT_TWS,
        client_id: int = 1,
        mode: str = "paper",
        timeout: float = 30.0,
    ) -> None:
        if not HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is not installed. Run: pip install ib_insync"
            )
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self.mode      = mode.lower()
        self.timeout   = timeout
        self._ib: Optional[IB] = None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, ibkr_cfg: dict, mode_override: Optional[str] = None) -> "IBKRClient":
        """
        Build from config/config.yaml ibkr section.

        mode_override lets --ibkr-mode live CLI flag switch without editing config.
        """
        mode = (mode_override or ibkr_cfg.get("mode", "paper")).lower()
        if "port" in ibkr_cfg:
            port = int(ibkr_cfg["port"])
        else:
            port = DEFAULT_PAPER_PORT_TWS if mode == "paper" else DEFAULT_LIVE_PORT_TWS
        return cls(
            host      = ibkr_cfg.get("host", "127.0.0.1"),
            port      = port,
            client_id = int(ibkr_cfg.get("client_id", 1)),
            mode      = mode,
            timeout   = float(ibkr_cfg.get("timeout_sec", 30.0)),
        )

    # ── Connection context manager ────────────────────────────────────────────

    def connect(self) -> None:
        self._ib = IB()
        try:
            self._ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=False,
            )
        except Exception as exc:
            raise IBKRConnectionError(
                f"Cannot connect to IBKR TWS/Gateway at {self.host}:{self.port} "
                f"(mode={self.mode}). "
                "Ensure TWS or IB Gateway is running and API connections are enabled "
                "in TWS → Edit → Global Configuration → API → Settings.\n"
                f"Original error: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    @property
    def ib(self) -> "IB":
        if self._ib is None or not self._ib.isConnected():
            raise IBKRConnectionError(
                "Not connected. Call connect() first or use IBKRClient as a context manager."
            )
        return self._ib

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    # ── Account data ──────────────────────────────────────────────────────────

    def get_nav(self) -> float:
        """Return account NetLiquidation (total NAV) in USD."""
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        # Fallback: any currency
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation":
                return float(av.value)
        raise IBKRConnectionError(
            "Could not retrieve NetLiquidation from IBKR account values. "
            "Check TWS account permissions."
        )

    def get_buying_power(self) -> float:
        """Return AvailableFunds (settled cash + margin headroom) in USD."""
        for av in self.ib.accountValues():
            if av.tag == "AvailableFunds" and av.currency == "USD":
                return float(av.value)
        for av in self.ib.accountValues():
            if av.tag == "AvailableFunds":
                return float(av.value)
        return 0.0

    def get_positions(self) -> List[dict]:
        """
        Return open positions as dicts compatible with the rebalancer format:
          ticker, commodity, direction, shares, entry_price, entry_date
        """
        result = []
        for pos in self.ib.positions():
            qty = float(pos.position)
            if qty == 0:
                continue
            result.append({
                "ticker":      pos.contract.symbol,
                "commodity":   "",             # IBKR doesn't store this field
                "direction":   "LONG" if qty > 0 else "SHORT",
                "shares":      int(abs(qty)),
                "entry_price": float(pos.avgCost),
                "entry_date":  "",
            })
        return result

    # ── Market data snapshot ──────────────────────────────────────────────────

    def get_last_price(self, ticker: str, timeout_sec: float = 5.0) -> Optional[float]:
        """
        Fetch last traded / close price for a US stock via snapshot market data.
        Returns None if the price cannot be retrieved within timeout_sec.
        """
        contract = self._stock_contract(ticker)
        self.ib.qualifyContracts(contract)
        ticker_data = self.ib.reqMktData(contract, "", True, False)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.ib.sleep(0.25)
            price = ticker_data.last or ticker_data.close
            if price and price > 0:
                self.ib.cancelMktData(contract)
                return float(price)
        self.ib.cancelMktData(contract)
        return None

    # ── Order placement ───────────────────────────────────────────────────────

    @staticmethod
    def _stock_contract(ticker: str) -> "Contract":
        c = Contract()
        c.symbol   = ticker
        c.secType  = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    def place_bracket_order(
        self,
        ticker: str,
        direction: str,
        quantity: int,
        entry_limit_price: float,
        tp_price: float,
        sl_price: float,
    ) -> Tuple:
        """
        Place an IBKR bracket order:
          Parent  — LMT entry (BUY or SELL SHORT)
          TP leg  — LMT exit at take-profit price (attached child)
          SL leg  — STP exit at stop-loss price   (attached child)

        Bracket legs are transmitted together. TP and SL are One-Cancels-Other.

        Returns tuple of ib_insync Trade objects (parent, tp, sl).
        Raises IBKRConnectionError if not connected.
        """
        if quantity < 1:
            raise ValueError(
                f"IBKR does not support fractional shares — quantity must be >= 1, got {quantity}"
            )

        contract = self._stock_contract(ticker)
        self.ib.qualifyContracts(contract)

        action_enter = "BUY"  if direction.upper() == "LONG" else "SELL"
        bracket = self.ib.bracketOrder(
            action_enter,
            quantity,
            entry_limit_price,
            tp_price,
            sl_price,
        )

        trades = []
        for order in bracket:
            trade = self.ib.placeOrder(contract, order)
            trades.append(trade)

        # Give TWS time to confirm
        self.ib.sleep(1)
        return tuple(trades)

    def cancel_order_by_id(self, order_id: int) -> None:
        """Cancel a live order by its IBKR order ID."""
        for trade in self.ib.trades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                return
