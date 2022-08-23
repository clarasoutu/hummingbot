"""
The configuration parameters for a user made Bolinger Bands strategy.
"""

import re
from decimal import Decimal
from typing import Optional

from hummingbot.client.config.config_validators import validate_decimal, validate_exchange, validate_market_trading_pair
from hummingbot.client.config.config_var import ConfigVar
from hummingbot.client.settings import AllConnectorSettings, required_exchanges


def exchange_on_validated(value: str) -> None:
    required_exchanges.add(value)


def market_validate(value: str) -> Optional[str]:
    pairs = list()
    quote_assets = list()
    if len(value.strip()) == 0:
        # Whitespace
        return "Invalid market(s). The given entry is empty."
    markets = list(value.upper().split(","))
    for market in markets:
        if len(market.strip()) == 0:
            return "Invalid markets. The given entry contains an empty market."
        tokens = market.strip().split("-")
        if len(tokens) != 2:
            return f"Invalid market. {market} doesn't contain exactly 2 tickers."
        for token in tokens:
            # Check allowed ticker lengths
            if len(token.strip()) == 0:
                return f"Invalid market. Ticker {token} has an invalid length."
            if bool(re.search('^[a-zA-Z0-9]*$', token) is False):
                return f"Invalid market. Ticker {token} contains invalid characters."
        # The pair is valid
        pair = f"{tokens[0]}-{tokens[1]}"
        if pair in pairs:
            return f"Duplicate market {pair}."
        pairs.append(pair)
        if tokens[1] not in quote_assets and len(quote_assets) > 0:
            return f"Quote asset {tokens[1]} differs from previous ones"
        quote_assets.append(tokens[1])


def maker_trading_pair_prompt():
    exchange = bollinger_bands_config_map.get("exchange").value
    example = AllConnectorSettings.get_example_pairs().get(exchange)
    return "Enter the token trading pair you would like to trade on %s%s >>> " \
           % (exchange, f" (e.g. {example})" if example else "")


# strategy specific validators
def validate_exchange_trading_pair(value: str) -> Optional[str]:
    exchange = bollinger_bands_config_map.get("exchange").value
    return validate_market_trading_pair(exchange, value)


async def order_amount_prompt() -> str:
    trading_pair = bollinger_bands_config_map["market"].value
    base_asset, _ = trading_pair.split("-")
    return f"What is the amount of {base_asset} per order? >>> "


def on_validated_price_source_exchange(value: str):
    if value is None:
        bollinger_bands_config_map["price_source_market"].value = None


def order_size_prompt() -> str:
    quote_token = bollinger_bands_config_map["markets"].value.split(",")[0].split("-")[1]
    return f"What is the size of each order (in {quote_token} amount)? >>> "


bollinger_bands_config_map = {
    "strategy":
        ConfigVar(key="strategy",
                  prompt=None,
                  default="bollinger_bands"),
    "exchange":
        ConfigVar(key="exchange",
                  prompt="Enter your maker spot connector >>> ",
                  validator=validate_exchange,
                  on_validated=exchange_on_validated,
                  prompt_on_new=True),
    "markets":
        ConfigVar(key="markets",
                  prompt="Enter a list of markets (comma separated, e.g. LTC-USDT,ETH-USDT) that share quote asset >>> ",
                  type_str="str",
                  validator=market_validate,
                  prompt_on_new=True),
    "order_amount":
        ConfigVar(key="order_amount",
                  prompt=order_size_prompt,
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 0, inclusive=False),
                  prompt_on_new=True),
    "base_amount_percentage":
        ConfigVar(key="base_amount_percentage",
                  prompt="What percentage of the base amount do you want to use for each order? "
                         "(Enter 1 to indicate 1%) >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 1, 100, inclusive=True),
                  default=Decimal("50"),
                  prompt_on_new=True),
    "volatility_hours":
        ConfigVar(key="volatility_hours",
                  prompt="Enter amount of hourly candles that will be used to calculate the volatility >>> ",
                  type_str="int",
                  validator=lambda v: validate_decimal(v, 5, 600),
                  prompt_on_new=True,
                  default=100),
    "bb_upper_band":
        ConfigVar(key="bb_upper_band",
                  prompt="How many standard deviations above the mean should the upper price be >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 0, 6),
                  prompt_on_new=True,
                  default=3),
    "bb_lower_band":
        ConfigVar(key="bb_lower_band",
                  prompt="How many standard deviations below the mean should the lower price be >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 0, 6),
                  prompt_on_new=True,
                  default=1),
    "rsi_upper_threshold":
        ConfigVar(key="rsi_upper_threshold",
                  prompt="What RSI threshold should be used to cancel buy orders? (Enter 0 to indicate 100%) >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 50, 100, inclusive=True),
                  default=Decimal("70"),
                  prompt_on_new=False),
    "rsi_lower_threshold":
        ConfigVar(key="rsi_lower_threshold",
                  prompt="What RSI threshold should be used to cancel sell orders? (Enter 0 to indicate 100%) >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 0, 50, inclusive=True),
                  default=Decimal("70"),
                  prompt_on_new=False),
}
