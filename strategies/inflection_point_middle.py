import threading
import pandas as pd
from finta import TA
from datetime import datetime
from LZCTrader.strategy import Strategy
from brokers.broker import Broker
from LZCTrader.order import Order
from features import ForceFeatureTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class INFLECTION1(Strategy):
    """Example Strategy

    策略演示文件
    请根据此文件的规范格式，写出自定义策略
    """

    def __init__(
        self, instrument: str, exchange: str, parameters: dict, broker: Broker  # 这四个是必需参数
    ) -> None:
        # 必需：
        self.instrument = instrument  # 品种
        self.exchange = exchange  # 交易所
        self.parameters = parameters  # 策略参数
        self.broker = broker  # 功能接口

        # 自定义：
        self.trade_num = 10  # 交易手数
        self.trade_offset = 3  # 取买几卖几
        self.lock = threading.Lock()  # 线程锁

        # 参数

        # 模型设置


    def generate_features(self, data: pd.DataFrame):
        # 在此函数中，根据传入参数data，计算出你策略所需的MA，EMA等指标，非必需
        data['middle'] = (data['Open'] + data['Close']) / 2

        wper = 2 * self.parameters["period"] - 1
        abs_diff = (data['middle'] - data['middle'].shift(1)).abs()
        ema1 = abs_diff.ewm(span=self.parameters["period"], adjust=False).mean()
        ema2 = ema1.ewm(span=wper, adjust=False).mean()

        # ema5 = data['middle'].ewm(span=5, adjust=False).mean()
        # ema20 = data['middle'].ewm(span=20, adjust=False).mean()
        diff = (ema1 * self.parameters['m'])

        return diff

    def rngfilt(self, price: pd.Series, smoothrng: pd.Series) -> pd.Series:
        filt = pd.Series(index=price.index, dtype='float64')
        filt.iloc[0] = price.iloc[0]

        for i in range(1, len(price)):
            prev = filt.iloc[i - 1]
            r = smoothrng.iloc[i]
            x = price.iloc[i]

            if x > prev:
                filt.iloc[i] = prev if x - r < prev else x - r
            else:
                filt.iloc[i] = prev if x + r > prev else x + r

        return filt

    def generate_signal(self, dt: datetime):
        # 此为函数主体，根据指标进行计算，产生交易信号并下单，程序只会调用这一个函数进行不断循环。必需
        start = "2025-06-27 09:00:00"  # 2024-06-01 09:00:00
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(end)
        new_orders = []
        data = self.broker.get_candles(self.instrument, granularity="1min", start_time=start, end_time=end, cut_yesterday=False)  # 取行情数据函数示例
        # granularity：时间粒度，支持1s，5s，1min，1h等；
        # count：取k线的数目；
        # cut_yesterday：取的数据中，当同时包含今日数据和昨日数据时，是否去掉昨日数据。True表示去掉；

        data = data[::-1]  # 取到的数据中，按时间由近到远排序。再此翻转为由远到近，便于某些策略处理
        #更新最新价格数据



        if len(data) < 99:
            print("数据不足!")
            return None








        # signal
        data['middle'] = (data['Open'] + data['Close']) / 2

        diff_ema = self.generate_features(data)
        data['diff'] = diff_ema
        data['smooth'] = self.rngfilt(data['middle'], diff_ema)

        print(data.tail(5).to_string())


        signal = 0
        if data['smooth'].iloc[-1] > data['smooth'].iloc[-2]:
            signal = 1
        elif data['smooth'].iloc[-1] < data['smooth'].iloc[-2]:
            signal = -1


        # 仓位判断
        position_dicts = self.broker.get_positionInfo(self.instrument)
        position = {}
        if position_dicts is not None:
            for position_dict in position_dicts:
                if position_dict["direction"] == 2:
                    position["long_tdPosition"] = position_dict["tdPosition"]
                    position["long_ydPosition"] = position_dict["ydPosition"]
                    position["openPrice"] = position_dict["openPrice"]
                    position["positionProfit"] = position_dict["positionProfit"]
                elif position_dict["direction"] == 3:
                    position["short_tdPosition"] = position_dict["tdPosition"]
                    position["short_ydPosition"] = position_dict["ydPosition"]
                    position["openPrice"] = position_dict["openPrice"]
                    position["positionProfit"] = position_dict["positionProfit"]


        # print(f"{self.instrument} position", position_dict["long_tdPosition"], position_dict["long_ydPosition"], position_dict["short_tdPosition"], position_dict["short_ydPosition"]) # 仓位查询
        temp = self.broker.get_candles(self.instrument, granularity="1s", count=1)
        self.current_point = temp.iloc[0]['Close']  # 取最近一根秒级k线，作为当前价格
        print(f'{self.instrument}{position}')
        if not position:

            if signal == 1:
                print(f'做空{self.instrument}')
                self.broker.relog()  # 由于一段时间不登录，交易所可能会自动下线，所以每次下单前先登录
                duo_enter_point = self.current_point + self.trade_offset  # 下单价。为保证立刻成交，在此取买三、卖三报单，按照价格优先原则，会按当前价成交。取买几、卖几可自定义。
                new_order = Order(
                    instrument=self.instrument,
                    exchange=self.exchange,
                    direction=2,  # 2为买，3为卖
                    offset=1,  # 1为开仓，4为平今，5为平昨
                    price=duo_enter_point,  # 下单价
                    volume=self.trade_num,  # 下单手数
                    stopPrice=0,  # 未实现功能。设为0即可
                    orderPriceType=1  # 类型：限价单（现在限价单和市价单由报单价决定。以开多仓为例，报单价比当前价高，则立即成交，相当于市价单。报单价比当前价低，则需等价格跌到此价才成交，相当于现价单）
                )
                self.point = self.current_point
                new_orders.append(new_order)
                self.write_order(instrument=self.instrument, type=1, point=self.point, profit=0)  # 记录下单结果
            if signal == -1:
                print(f'做多{self.instrument}')
                self.broker.relog()
                kong_enter_point = self.current_point - self.trade_offset
                new_order = Order(
                    instrument=self.instrument,
                    exchange=self.exchange,
                    direction=3,
                    offset=1,
                    price=kong_enter_point,
                    volume=self.trade_num,
                    stopPrice=0,
                    orderPriceType=1
                )
                new_orders.append(new_order)
                self.point = self.current_point
                self.write_order(instrument=self.instrument, type=2, point=self.point, profit=0)
        elif position.get("long_tdPosition", 0) > 0:
            profit = position['positionProfit']/position['openPrice']
            print(f'{self.instrument}:{profit}')
            if signal == -1:
                print(f'平多仓{self.instrument},profit:{profit},singal:{signal}')
                self.broker.relog()
                kong_enter_point = self.current_point - self.trade_offset
                new_order = Order(
                    instrument=self.instrument,
                    exchange=self.exchange,
                    direction=3,
                    offset=4,
                    price=kong_enter_point,
                    volume=position["long_tdPosition"],
                    stopPrice=0,
                    orderPriceType=1
                )
                new_order1 = Order(
                    instrument=self.instrument,
                    exchange=self.exchange,
                    direction=3,  # 2为买，3为卖
                    offset=1,  # 1为开仓，4为平今，5为平昨
                    price=kong_enter_point,  # 下单价
                    volume=self.trade_num,  # 下单手数
                    stopPrice=0,  # 未实现功能。设为0即可
                    orderPriceType=1
                    # 类型：限价单（现在限价单和市价单由报单价决定。以开多仓为例，报单价比当前价高，则立即成交，相当于市价单。报单价比当前价低，则需等价格跌到此价才成交，相当于现价单）
                )
                new_orders.append(new_order)
                new_orders.append(new_order1)
                self.write_order(instrument=self.instrument, type=3, point=self.current_point, profit=profit)
                self.write_order(instrument=self.instrument, type=2, point=self.current_point, profit=profit)

        elif position.get("short_tdPosition", 0) > 0:
            profit = position['positionProfit'] / position['openPrice']
            print(f'{self.instrument}:{profit}')
            if signal == 1:
                print(f'平空仓{self.instrument}, profit:{profit}, singal:{signal}')
                self.broker.relog()
                long_enter_point = self.current_point + self.trade_offset
                new_order = Order(
                    instrument=self.instrument,
                    exchange=self.exchange,
                    direction=2,
                    offset=4,
                    price=long_enter_point,
                    volume=position["short_tdPosition"],
                    stopPrice=0,
                    orderPriceType=1
                )
                new_order1 = Order(
                    instrument=self.instrument,
                    exchange=self.exchange,
                    direction=2,  # 2为买，3为卖
                    offset=1,  # 1为开仓，4为平今，5为平昨
                    price=long_enter_point,  # 下单价
                    volume=self.trade_num,  # 下单手数
                    stopPrice=0,  # 未实现功能。设为0即可
                    orderPriceType=1
                    # 类型：限价单（现在限价单和市价单由报单价决定。以开多仓为例，报单价比当前价高，则立即成交，相当于市价单。报单价比当前价低，则需等价格跌到此价才成交，相当于现价单）
                )
                new_orders.append(new_order)
                new_orders.append(new_order1)

                self.write_order(instrument=self.instrument, type=4, point=self.current_point, profit=profit)
                self.write_order(instrument=self.instrument, type=1, point=self.current_point, profit=profit)
        return new_orders


    def write_order(self, instrument, type, point, profit):  # 记录下单结果函数，非必需
        now = datetime.now().strftime("%m-%d %H:%M:%S")
        if type == 1:  # 买开
            line = f"{now} 做多 {self.instrument} 入场价：{point} "
        elif type == 2:  # 买平
            line = f"{now} 做空 {self.instrument} 入场价：{point} "
        elif type == 3:  # 卖开
            line = f"{now} 平多仓 {self.instrument} 卖出价：{point}  盈利:{profit}\n"
        elif type == 4:  # 卖平
            line = f"{now} 平空仓 {self.instrument} 卖出价：{point}  盈利:{profit}\n"
        else:
            raise ValueError("Invalid type")

        with self.lock:
            path = f'./result/{instrument}.txt'
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)


