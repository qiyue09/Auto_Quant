import threading
import pandas as pd
from finta import TA
from datetime import datetime
from LZCTrader.strategy import Strategy
from brokers.broker import Broker
from LZCTrader.order import Order


class Example(Strategy):
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
        self.params = parameters  # 策略参数
        self.broker = broker  # 功能接口

        # 自定义：
        self.trade_num = 1  # 交易手数
        self.trade_offset = 3  # 取买几卖几
        self.lock = threading.Lock()  # 线程锁

    def min_generate_features(self, data: pd.DataFrame):
        # 在此函数中，根据传入参数data，计算出你策略所需的MA，EMA等指标，非必需
        ema_period = self.params['ema_period']  # 取参数，进行计算

        return data

    def generate_signal(self, dt: datetime):
        # 此为函数主体，根据指标进行计算，产生交易信号并下单，程序只会调用这一个函数进行不断循环。必需

        new_orders = []
        # start = "2025-06-25 09:00:00"  # 2024-06-01 09:00:00
        # end = "2025-06-26 14:50:50"  # 2024-06-01 15:00:00
        start = "2025-06-27 09:00:00"  # 2024-06-01 09:00:00
        end = "2025-07-03 10:57:46"
        print(end)
        data = self.broker.get_candles(self.instrument, granularity="1min", start_time=start, end_time=end) # 取行情数据函数示例
        path = f'{self.instrument}.csv'
        data = data[::-1]
        data.to_csv(path, index=True, index_label='datetime', encoding="utf-8-sig")
        print(f'{self.instrument}chenggong')

        print(data.index.min())
        print(data.index.max())

        print(data.tail())

        # granularity：时间粒度，支持1s，5s，1min，1h等；
        # count：取k线的数目；
        # cut_yesterday：取的数据中，当同时包含今日数据和昨日数据时，是否去掉昨日数据。True表示去掉；
          # 取到的数据中，按时间由近到远排序。再此翻转为由远到近，便于某些策略处理

        some_condition = True  # 由取到的data计算，得到某些condition，作为策略下单条件


        return new_orders

    def write_order(self, type, point):  # 记录下单结果函数，非必需
        now = datetime.now().strftime("%m-%d %H:%M:%S")
        if type == 1:  # 买开
            line = f"{now} {self.instrument}，买开，{point} \n"
        elif type == 2:  # 买平
            line = f"{now} {self.instrument}，买平，{point} \n"
        elif type == 3:  # 卖开
            line = f"{now} {self.instrument}，卖开，{point} \n"
        elif type == 4:  # 卖平
            line = f"{now} {self.instrument}，卖平，{point} \n"
        else:
            raise ValueError("Invalid type")

        with self.lock:
            with open("L:/Quantification/LeopardSeek/result/order_book.txt", "a", encoding="utf-8") as f:
                f.write(line)


