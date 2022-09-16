import asyncio
import logging
import statistics
from decimal import Decimal
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.connector.exchange.binance.binance_exchange import BinanceExchange
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.utils.estimate_fee import build_trade_fee
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.rsi_bb.data_types import BollingerBands, PriceSize, Proposal
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_py_base import StrategyPyBase
from hummingbot.strategy.utils import order_age

NaN = float("nan")
s_decimal_zero = Decimal(0)
s_decimal_nan = Decimal("NaN")
lms_logger = None


class RSIBBStrategy(StrategyPyBase):

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global lms_logger
        if lms_logger is None:
            lms_logger = logging.getLogger(__name__)
        return lms_logger

    def init_params(self,
                    exchange: ExchangeBase,
                    order_amount: Decimal,
                    market_infos: Dict[str, MarketTradingPairTuple],
                    base_amount_percentage: Decimal = Decimal(20.0),
                    volatility_hours: int = 100,
                    bb_upper_band: Decimal = Decimal('3'),
                    bb_lower_band: Decimal = Decimal('3'),
                    rsi_upper_threshold: Decimal = Decimal('70'),
                    rsi_lower_threshold: Decimal = Decimal('30'),
                    converge_time: float = 60.,
                    status_report_interval: float = 900,
                    hb_app_notification: bool = False):
        self._exchange = exchange
        self._order_amount = order_amount
        self._market_infos = market_infos
        self._base_amount_percentage = base_amount_percentage
        self._candle_hours = volatility_hours
        self._bb_upper_band = bb_upper_band
        self._bb_lower_band = bb_lower_band
        self._rsi_upper_threshold = rsi_upper_threshold
        self._rsi_lower_threshold = rsi_lower_threshold
        self._ev_loop = asyncio.get_event_loop()
        self._status_report_interval = status_report_interval
        self._ready_to_trade = False
        self._candles_ready = False

        self._converge_time = converge_time
        self._order_refresh_time = converge_time
        self._max_order_age = converge_time

        self._token_balances = {}
        self._sell_budgets = {}
        self._buy_budgets = {}

        self._refresh_times = {market: 0 for market in market_infos}
        self._candles = {market: [] for market in market_infos}
        self._bolinger_bands = {market: object() for market in market_infos}
        self._rsi = {market: s_decimal_zero for market in market_infos}

        self._last_timestamp = 0
        self._last_converge_time = 0
        self._hb_app_notification = hb_app_notification

        self.add_markets([exchange])

        try:
            market = list(self._market_infos.keys())[0]
            exchange = self._market_infos[market]
            exchange.get_historical(market, self._candle_hours)
            self._candlesExchange = exchange
        except NotImplementedError:
            self.logger().warning(
                # TODO: Add the actual exchange name to this message after some investigation
                f"{self.active_markets[0].display_name} does not support historical data. Falling back to Binance.")
            self._candlesExchange = BinanceExchange(
                client_config_map=HummingbotApplication.main_application().client_config_map,
                binance_api_key="",
                binance_api_secret="",
                trading_required=False)

    @property
    def active_orders(self):
        """
        List active orders (they have been sent to the market and have not been canceled yet)
        """
        limit_orders = self.order_tracker.active_limit_orders
        return [o[1] for o in limit_orders]

    @property
    def sell_budgets(self):
        return self._sell_budgets

    @property
    def buy_budgets(self):
        return self._buy_budgets

    def all_base_tokens(self) -> Set[str]:
        """
        Get the base token (left-hand side) from all markets in this strategy
        """
        tokens = set()
        for market in self._market_infos:
            tokens.add(market.split("-")[0])
        return tokens

    def quote_token(self) -> str:
        """
        Get the quote token (right-hand side) from the first market in this strategy
        """
        for market in self._market_infos:
            return market.split("-")[1]

    def all_tokens(self) -> Set[str]:
        """
        Return a list of all tokens involved in this strategy (base and quote)
        """
        tokens = set()
        for market in self._market_infos:
            tokens.update(market.split("-"))
        return tokens

    def total_port_value_in_token(self) -> Decimal:
        """
        Total portfolio value in quote token amount
        """
        all_bals = self.adjusted_available_balances()
        port_value = all_bals.get(self.quote_token, s_decimal_zero)
        for market, market_info in self._market_infos.items():
            base, _ = market.split("-")
            port_value += all_bals[base] * market_info.get_mid_price()
        return port_value

    def adjusted_available_balances(self) -> Dict[str, Decimal]:
        """
        Calculates all available balances, account for amount attributed to orders and reserved balance.
        :return: a dictionary of token and its available balance
        """
        tokens = self.all_tokens()
        adjusted_bals = {t: s_decimal_zero for t in tokens}
        total_bals = {t: s_decimal_zero for t in tokens}
        total_bals.update(self._exchange.get_all_balances())
        for token in tokens:
            adjusted_bals[token] = self._exchange.get_available_balance(token)
        for order in self.active_orders:
            base, quote = order.trading_pair.split("-")
            if order.is_buy:
                adjusted_bals[quote] += order.quantity * order.price
            else:
                adjusted_bals[base] += order.quantity
        return adjusted_bals

    def _is_strategy_ready(self):
        """
        Check if the strategy is ready to trade after all historical candles are retrieved
        :return: True if the strategy is ready to trade, False otherwise
        """
        candles = [True for candles in self._candles.values() if len(candles) == self._candle_hours]

        return len(candles) == len(self._market_infos)

    def base_order_size(self, trading_pair: str, price: Decimal = s_decimal_zero):
        if price == s_decimal_zero:
            price = self._market_infos[trading_pair].get_mid_price()
        return self._order_amount / price

    def start(self, clock: Clock, timestamp: float):
        restored_orders = self._exchange.limit_orders
        for order in restored_orders:
            self._exchange.cancel(order.trading_pair, order.client_order_id)

    def stop(self, clock: Clock):
        pass

    def _run_preflights(self):
        """
        Run preflights to make sure all markets and historical data is ready to go
        :return True if all preflights are successful, False otherwise.
        """
        # 1. Verify we have obtained all the historical data to be able to calculate BB
        if not self._is_strategy_ready():
            self.logger().warning("Historical data not yet obtained. Please wait...")
            self.update_klines_data()
            return False
        # 2. Make sure there are no active orders in the exchange and all connectors are ready to trade
        if not self._ready_to_trade:
            # Check if there are restored orders, they should be canceled before strategy starts.
            self._ready_to_trade = self._exchange.ready and len(self._exchange.limit_orders) == 0
            if not self._exchange.ready:
                self.logger().warning(f"{self._exchange.name} is not ready. Please wait...")
                return False
            else:
                self.logger().info("Strategy and exchange are ready. Allocating budgets and kicking off trading")
                self.create_budget_allocation()
                return True
        return True

    # MAIN Strategy entry point
    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        :param timestamp: current tick timestamp
        """
        if not self._run_preflights():
            # Not ready to trade yet. Wait for next tick.
            return
        # Update the historical data; both bollinger bands and RSIs every interval
        if (timestamp - self._last_converge_time) > self._converge_time:
            # Update the historical data
            self.update_klines_data()
            # Update Bollinger bands and prices
            self.update_bollinger_bands()
            self.logger().info(f"Bollinger bands: {self._bolinger_bands}")
            # Update RSI coefficient
            self.update_rsi()
            self.logger().info(f"RSIs: {self._rsi}")
            self._last_converge_time = timestamp

        proposals = self.create_base_proposals()
        self._token_balances = self.adjusted_available_balances()
        self.apply_budget_constraint(proposals)
        self.cancel_active_orders(proposals)
        self.execute_orders_proposal(proposals)

        self._last_timestamp = timestamp

    async def active_orders_df(self) -> pd.DataFrame:
        """
        Return the active orders in a DataFrame.
        """
        size_q_col = f"Amount({self.quote_token()})"
        columns = ["Market", "Side", "Price", "Spread", "Amount", size_q_col, "Age"]
        data = []
        for order in self.active_orders:
            mid_price = self._market_infos[order.trading_pair].get_mid_price()
            spread = 0 if mid_price == 0 else abs(order.price - mid_price) / mid_price
            size_q = order.quantity * mid_price
            age = order_age(order, self.current_timestamp)
            # // indicates order is a paper order so 'n/a'. For real orders, calculate age.
            age_txt = "n/a" if age <= 0. else pd.Timestamp(age, unit='s').strftime('%H:%M:%S')
            data.append([
                order.trading_pair,
                "buy" if order.is_buy else "sell",
                float(order.price),
                f"{spread:.2%}",
                float(order.quantity),
                float(size_q),
                age_txt
            ])
        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by=["Market", "Side"], inplace=True)
        return df

    def budget_status_df(self) -> pd.DataFrame:
        """
        Return the trader's budget in a DataFrame
        """
        data = []
        columns = ["Market", f"Budget({self.quote_token()})", "Base bal", "Quote bal", "Base/Quote"]
        for market, market_info in self._market_infos.items():
            mid_price = market_info.get_mid_price()
            base_bal = self._sell_budgets[market]
            quote_bal = self._buy_budgets[market]
            total_bal_in_quote = (base_bal * mid_price) + quote_bal
            total_bal_in_token = total_bal_in_quote
            base_pct = (base_bal * mid_price) / total_bal_in_quote if total_bal_in_quote > 0 else s_decimal_zero
            quote_pct = quote_bal / total_bal_in_quote if total_bal_in_quote > 0 else s_decimal_zero
            data.append([
                market,
                float(total_bal_in_token),
                float(base_bal),
                float(quote_bal),
                f"{base_pct:.0%} / {quote_pct:.0%}"
            ])
        df = pd.DataFrame(data=data, columns=columns).replace(np.nan, '', regex=True)
        df.sort_values(by=["Market"], inplace=True)
        return df

    def market_status_df(self) -> pd.DataFrame:
        """
        Return the market status (prices, volatility) in a DataFrame
        """
        data = []
        # TODO: Add Bolinger bands to state
        columns = ["Market", f"Upper Price ({self.quote_token()})", f"Mid price ({self.quote_token()})",
                   f"Lower Price ({self.quote_token()})", "RSI"]
        for market, market_info in self._market_infos.items():
            mid_price = market_info.get_mid_price()
            data.append([
                market,
                f"{self._bolinger_bands[market].upper_band:.4f}",
                f"{float(mid_price):.2f}",
                f"{self._bolinger_bands[market].lower_band:.4f}",
                f"{self._rsi[market]:.2f}"
            ])
        df = pd.DataFrame(data=data, columns=columns).replace(np.nan, '', regex=True)
        df.sort_values(by=["Market"], inplace=True)
        return df

    async def format_status(self) -> str:
        """
        Return the budget, market and order statuses.
        """
        if not self._ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(list(self._market_infos.values())))

        budget_df = self.budget_status_df()
        lines.extend(["", "  Budget:"] + ["    " + line for line in budget_df.to_string(index=False).split("\n")])

        market_df = self.market_status_df()
        lines.extend(["", "  Markets:"] + ["    " + line for line in market_df.to_string(index=False).split("\n")])

        # See if there are any open orders.
        if len(self.active_orders) > 0:
            df = await self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active maker orders."])

        warning_lines.extend(self.balance_warning(list(self._market_infos.values())))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)

    def create_base_proposals(self):
        """
        Each tick this strategy creates a set of proposals based on the market_info and the parameters from the
        constructor based on the bollinger bands entry and exit price marks.
        """
        proposals = []
        for market, market_info in self._market_infos.items():
            buy_price = self._exchange.quantize_order_price(market, self._bolinger_bands[market].lower_band)
            buy_size = self.base_order_size(market, buy_price)
            sell_price = self._exchange.quantize_order_price(market, self._bolinger_bands[market].upper_band)
            sell_size = self.base_order_size(market, sell_price)
            proposals.append(Proposal(market, PriceSize(buy_price, buy_size), PriceSize(sell_price, sell_size)))
        return proposals

    def create_budget_allocation(self):
        """
        Create buy and sell budgets for every market
        """
        self._sell_budgets = {m: s_decimal_zero for m in self._market_infos}
        self._buy_budgets = {m: s_decimal_zero for m in self._market_infos}
        portfolio_value = self.total_port_value_in_token()
        market_portion = portfolio_value / len(self._market_infos)
        balances = self.adjusted_available_balances()
        for market, market_info in self._market_infos.items():
            base, quote = market.split("-")
            self._sell_budgets[market] = balances[base]
            buy_budget = market_portion - (balances[base] * market_info.get_mid_price())
            if buy_budget > s_decimal_zero:
                self._buy_budgets[market] = buy_budget
        self.logger().info(f"BUY Budget allocation: {self._buy_budgets}")
        self.logger().info(f"SELL Budget allocation: {self._sell_budgets}")

    def apply_budget_constraint(self, proposals: List[Proposal]) -> List[Proposal]:
        """
        Apply budget constraints to the buy and sell proposals.
        :param list(Proposals): The proposals to constrain.
        """
        balances = self._token_balances.copy()
        for proposal in proposals:
            if balances[proposal.base()] < proposal.sell.size:
                proposal.sell.size = balances[proposal.base()]
            proposal.sell.size = self._exchange.quantize_order_amount(proposal.market, proposal.sell.size)
            balances[proposal.base()] -= proposal.sell.size

            quote_size = proposal.buy.size * proposal.buy.price
            quote_size = balances[proposal.quote()] if balances[proposal.quote()] < quote_size else quote_size

            buy_fee = build_trade_fee(
                self._exchange.name,
                is_maker=True,
                base_currency="",
                quote_currency="",
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=Decimal("0"),
                price=Decimal("0"),
            )
            buy_size = quote_size / (proposal.buy.price * (Decimal("1") + buy_fee.percent))
            proposal.buy.size = self._exchange.quantize_order_amount(proposal.market, buy_size)
            balances[proposal.quote()] -= quote_size

    def cancel_active_orders(self, proposals: List[Proposal]):
        """
        Cancel any orders that have an order age greater than self._max_order_age
        """
        for proposal in proposals:
            to_cancel = False
            cur_orders = [o for o in self.active_orders if o.trading_pair == proposal.market]
            if cur_orders and any(order_age(o, self.current_timestamp) > self._max_order_age for o in cur_orders):
                to_cancel = True
            elif self._refresh_times[proposal.market] <= self.current_timestamp and cur_orders:
                to_cancel = True
            if to_cancel:
                for order in cur_orders:
                    self.cancel_order(self._market_infos[proposal.market], order.client_order_id)
                    # To place new order on the next tick
                    self._refresh_times[order.trading_pair] = self.current_timestamp + 0.1

    def execute_orders_proposal(self, proposals: List[Proposal]):
        """
        Execute a list of proposals if:
            - the current timestamp is less than its refresh timestamp.
            - <Only for buy orders>, the upper RSI threshold is not surpassed (overbought scenario)
            - <Only for sell orders>, the lower RSI threshold is not reached (oversold scenario)
        Adjust the buy and sell price to the mid price if the latter is higher (sell) or lower (sell) than the Bollinger
          entry/exit prices.
        Update the refresh timestamp in any circumstance.
        """
        for proposal in proposals:
            maker_order_type: OrderType = self._exchange.get_maker_order_type()
            cur_orders = [o for o in self.active_orders if o.trading_pair == proposal.market]
            if cur_orders or self._refresh_times[proposal.market] > self.current_timestamp:
                continue
            mid_price = self._market_infos[proposal.market].get_mid_price()
            if proposal.buy.size > 0:
                # 1 Filter out those buy orders with prices lower than the midprice
                if proposal.buy.price - mid_price > 0:
                    self.logger().info(f"Buy price {proposal.buy.price} is above the mid price {mid_price}")
                    proposal.buy.price = mid_price
                # 2 Filter out those when RSI is above the upper threshold
                if self._rsi[proposal.market] >= self._rsi_upper_threshold:
                    self._refresh_times[proposal.market] = self.current_timestamp + self._order_refresh_time
                    continue
                self.logger().info(f"({proposal.market}) Creating a bid order {proposal.buy} value: "
                                   f"{proposal.buy.size * proposal.buy.price:.2f} {proposal.quote()} ")
                self.buy_with_specific_market(
                    self._market_infos[proposal.market],
                    proposal.buy.size,
                    order_type=maker_order_type,
                    price=proposal.buy.price
                )
            if proposal.sell.size > 0:
                # 1 Filter out those sell orders with prices higher than the mid price
                if proposal.sell.price - mid_price < 0:
                    self.logger().info(f"Buy price {proposal.sell.price} is below the mid price {mid_price}")
                    proposal.sell.price = mid_price
                # 2 Filter out those when RSI is below the upper threshold
                if Decimal(self._rsi[proposal.market]) <= self._rsi_lower_threshold:
                    self._refresh_times[proposal.market] = self.current_timestamp + self._order_refresh_time
                    continue
                self.logger().info(f"({proposal.market}) Creating an ask order at {proposal.sell} value: "
                                   f"{proposal.sell.size * proposal.sell.price:.2f} {proposal.quote()} ")
                self.sell_with_specific_market(
                    self._market_infos[proposal.market],
                    proposal.sell.size,
                    order_type=maker_order_type,
                    price=proposal.sell.price
                )
            if proposal.buy.size > 0 or proposal.sell.size > 0:
                self._refresh_times[proposal.market] = self.current_timestamp + self._order_refresh_time

    def did_fill_order(self, event):
        """
        Check if order has been completed, log it, notify the hummingbot application, and update budgets.
        """
        order_id = event.order_id
        market_info = self.order_tracker.get_shadow_market_pair_from_order_id(order_id)
        if market_info is not None:
            if event.trade_type is TradeType.BUY:
                msg = f"({market_info.trading_pair}) Maker BUY order (price: {event.price}) of {event.amount} " \
                      f"{market_info.base_asset} is filled."
                self.log_with_clock(logging.INFO, msg)
                self.notify_hb_app_with_timestamp(msg)
                self._buy_budgets[market_info.trading_pair] -= (event.amount * event.price)
                self._sell_budgets[market_info.trading_pair] += event.amount
            else:
                msg = f"({market_info.trading_pair}) Maker SELL order (price: {event.price}) of {event.amount} " \
                      f"{market_info.base_asset} is filled."
                self.log_with_clock(logging.INFO, msg)
                self.notify_hb_app_with_timestamp(msg)
                self._sell_budgets[market_info.trading_pair] -= event.amount
                self._buy_budgets[market_info.trading_pair] += (event.amount * event.price)

    def update_klines_data(self):
        for market, _ in self._market_infos.items():
            self._candles[market] = self._candlesExchange.get_historical(market, self._candle_hours)

    def update_bollinger_bands(self):
        """
        Calculate and update the Bollinger bands for each market in this strategy and its current mid price
        """
        for market in self._market_infos:
            close_prices = list(map(lambda x: x.close, self._candles[market]))
            mean = statistics.mean(close_prices)
            stdev = statistics.stdev(close_prices)
            upper_band = mean + (stdev * self._bb_upper_band)
            lower_band = mean - (stdev * self._bb_lower_band)
            self._bolinger_bands[market] = BollingerBands(std_dev=stdev,
                                                          upper_band=upper_band,
                                                          lower_band=lower_band,
                                                          volatility_hours=self._candle_hours
                                                          )

    def update_rsi(self):
        """
        The Relative Strength Index (RSI) is a momentum indicator that describes the current price relative to average
        high and low prices over a previous trading period. This indicator estimates overbought or oversold status and
        helps spot trend reversals, price pullbacks, and the emergence of bullish or bearish markets. It does not use
        external libraries.
        """
        for market in self._market_infos:
            close_prices = pd.Series(list(map(lambda x: x.close, self._candles[market])))
            close_prices_delta = close_prices.diff().dropna()

            ups = close_prices_delta.clip(lower=0)
            downs = close_prices_delta.clip(upper=0) * -1

            ma_up = ups.rolling(window=self._candle_hours - 1).mean()
            ma_down = downs.rolling(window=self._candle_hours - 1).mean()

            rsi = ma_up / ma_down
            rsi = 100 - (100 / (1 + rsi))

            self._rsi[market] = rsi[self._candle_hours - 1]

    def notify_hb_app(self, msg: str):
        """
        Send a message to the hummingbot application
        """
        if self._hb_app_notification:
            super().notify_hb_app(msg)
