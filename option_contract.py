from datetime import timedelta # Corrected import for timedelta
import numpy as np

class OptionContract:
    '''期权合约表示'''
    def __init__(self, symbol, strike, expiry, option_type, premium):
        self.symbol = symbol
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type  # 'call' or 'put'
        self.premium = premium

    def intrinsic_value(self, underlying_price):
        '''计算内在价值'''
        if self.option_type == 'call':
            return max(0, underlying_price - self.strike)
        else:
            return max(0, self.strike - underlying_price)

    def time_value(self, underlying_price):
        '''计算时间价值'''
        return self.premium - self.intrinsic_value(underlying_price)

    def __repr__(self):
        return f"{self.symbol} {self.expiry} {self.strike} {self.option_type} @ {self.premium}"
