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

class potential_force(Strategy):
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
        self.trade_num = 10  # 交易手数
        self.trade_offset = 3  # 取买几卖几
        self.lock = threading.Lock()  # 线程锁

        # 参数
        self.retrain_interval=7  # 每 5 根K线训练一次
        self.window_size=100  # 用过去多少条数据训练模型
        self.transformer_n=10 #势函数的价格点
        self.transformer_sigma=1

        # 模型设置
        self.counter = 0
        self.model = RandomForestClassifier()
        self.transformer = ForceFeatureTransformer(n=self.transformer_n, sigma=self.transformer_sigma)
        self.data_buffer = []
        self.point = 0

    def min_generate_features(self, data: pd.DataFrame):
        # 在此函数中，根据传入参数data，计算出你策略所需的MA，EMA等指标，非必需
        ema_period = self.params['ema_period']  # 取参数，进行计算

        return data

    def generate_signal(self, dt: datetime):
        # 此为函数主体，根据指标进行计算，产生交易信号并下单，程序只会调用这一个函数进行不断循环。必需

        new_orders = []
        data = self.broker.get_candles(self.instrument, granularity="1min", count=103, cut_yesterday=False)  # 取行情数据函数示例
        # granularity：时间粒度，支持1s，5s，1min，1h等；
        # count：取k线的数目；
        # cut_yesterday：取的数据中，当同时包含今日数据和昨日数据时，是否去掉昨日数据。True表示去掉；
        data = data[::-1]  # 取到的数据中，按时间由近到远排序。再此翻转为由远到近，便于某些策略处理
        if len(data) < 101:
            print("数据不足!")
            return None

        df = pd.DataFrame(data[-100:])

        # 每隔一段时间重新训练模型
        if self.counter % self.retrain_interval == 0:
            features = self.transformer.fit_transform(df)
            df_feat = pd.concat([df.reset_index(drop=True), features], axis=1)
            df_feat['target'] = np.sign(df_feat['Close'].diff().shift(-1))  # 明日涨跌

            X = df_feat[['force']].iloc[:-1]
            y = df_feat['target'].iloc[:-1]
            self.model.fit(X, y)
        self.counter += 1
        # 用最新数据做预测
        recent_df = df.iloc[-(self.transformer_n + 1):]
        print(recent_df)
        force_df = self.transformer.transform(recent_df)

        # 当前最新一行的特征
        current_force_value = force_df.iloc[-1]['force']
        current_force_df = pd.DataFrame([[current_force_value]], columns=['force'])

        pred = self.model.predict(current_force_df)
        signal = pred[0]
        print(current_force_value)
        print(signal)

        some_condition = True  # 由取到的data计算，得到某些condition，作为策略下单条件
        position_dict = self.broker.get_position(self.instrument)

        print(f"{self.instrument} position", position_dict["long_tdPosition"], position_dict["long_ydPosition"], position_dict["short_tdPosition"], position_dict["short_ydPosition"]) # 仓位查询

        temp = self.broker.get_candles(self.instrument, granularity="1s", count=1)
        current_point = temp.iloc[0]['Close']  # 取最近一根秒级k线，作为当前价格

        if position_dict["long_tdPosition"] == 0 and position_dict["short_tdPosition"] == 0:
            print("开")
            if signal == 1:
                self.broker.relog()  # 由于一段时间不登录，交易所可能会自动下线，所以每次下单前先登录
                duo_enter_point = current_point + self.trade_offset  # 下单价。为保证立刻成交，在此取买三、卖三报单，按照价格优先原则，会按当前价成交。取买几、卖几可自定义。
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
                self.point = current_point
                new_orders.append(new_order)
                self.write_order(instrument=self.instrument, type=1, point=current_point, profit=0)  # 记录下单结果
            if signal == -1:
                self.broker.relog()
                kong_enter_point = current_point - self.trade_offset
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
                self.point = current_point
                self.write_order(instrument=self.instrument, type=2, point=current_point, profit=0)
        elif position_dict["long_tdPosition"] > 0 and signal == -1:
            print("平")
            self.broker.relog()
            kong_enter_point = current_point - self.trade_offset
            new_order = Order(
                instrument=self.instrument,
                exchange=self.exchange,
                direction=3,
                offset=4,
                price=kong_enter_point,
                volume=position_dict["long_tdPosition"],
                stopPrice=0,
                orderPriceType=1
            )
            profit = current_point - self.point
            new_order1 = Order(
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
            self.write_order(instrument=self.instrument, type=3, point=current_point, profit=profit)
        elif position_dict["short_tdPosition"] > 0 and signal == 1:
            print("平")

            self.broker.relog()
            long_enter_point = current_point + self.trade_offset
            new_order = Order(
                instrument=self.instrument,
                exchange=self.exchange,
                direction=2,
                offset=4,
                price=long_enter_point,
                volume=position_dict["short_tdPosition"],
                stopPrice=0,
                orderPriceType=1
            )
            profit = self.point - current_point
            new_order1 = Order(
                instrument=self.instrument,
                exchange=self.exchange,
                direction=2,
                offset=1,
                price=long_enter_point,
                volume=self.trade_num,
                stopPrice=0,
                orderPriceType=1
            )
            new_orders.append(new_order)
            self.write_order(instrument=self.instrument, type=4, point=current_point, profit=profit)
        print(new_orders)
        return new_orders


    def write_order(self, instrument, type, point, profit):  # 记录下单结果函数，非必需
        now = datetime.now().strftime("%m-%d %H:%M:%S")
        if type == 1:  # 买开
            line = f"{now} 做多 {self.instrument}\n 入场价：{point} \n"
        elif type == 2:  # 买平
            line = f"{now} 做空 {self.instrument}\n 入场价：{point} \n"
        elif type == 3:  # 卖开
            line = f"{now} 平多仓 {self.instrument}\n 卖出价：{point} \n 盈利:{profit}\n"
        elif type == 4:  # 卖平
            line = f"{now} 平空仓 {self.instrument}\n 卖出价：{point} \n 盈利:{profit}\n"
        else:
            raise ValueError("Invalid type")

        with self.lock:
            path = f'./result/{instrument}.txt'
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)


