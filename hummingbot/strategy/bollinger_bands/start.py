from decimal import Decimal

from hummingbot.strategy.bollinger_bands import BollingerBandsStrategy
from hummingbot.strategy.bollinger_bands.bollinger_bands_config_map import bollinger_bands_config_map as c_map
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


def start(self):
    try:
        # 1. Prepare input data to pass to the strategy initialisation method.
        exchange = c_map.get("exchange").value.lower()
        markets = [market.strip() for market in c_map.get("markets").value.split(",")]
        order_amount = c_map.get("order_amount").value
        volatility_hours = c_map.get("volatility_hours").value
        bb_upper_band = c_map.get("bb_upper_band").value
        bb_lower_band = c_map.get("bb_lower_band").value
        rsi_upper_threshold = c_map.get("rsi_upper_threshold").value
        rsi_lower_threshold = c_map.get("rsi_lower_threshold").value
        base_amount_percentage = c_map.get("base_amount_percentage").value

        # 2. Prepare markets and exchange attributes
        self._initialize_markets([(exchange, markets)])
        exchange = self.markets[exchange]
        market_infos = {}
        for market in markets:
            base, quote = market.split("-")
            market_infos[market] = MarketTradingPairTuple(exchange, market, base, quote)

        # 3. Pass all pre-calculated input parameters and kick the strategy off.
        self.strategy = BollingerBandsStrategy()
        self.strategy.init_params(
            exchange=exchange,
            order_amount=order_amount,
            market_infos=market_infos,
            base_amount_percentage=base_amount_percentage,
            volatility_hours=volatility_hours,
            bb_upper_band=Decimal(bb_upper_band),
            bb_lower_band=Decimal(bb_lower_band),
            rsi_upper_threshold=Decimal(rsi_upper_threshold),
            rsi_lower_threshold=Decimal(rsi_lower_threshold),
            hb_app_notification=True
        )

    except Exception as e:
        self.logger().error(str(e))
        self.logger().error("Unknown error during initialization.", exc_info=True)
