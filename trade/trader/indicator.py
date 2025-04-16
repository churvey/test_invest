from backtrader import Indicator
import backtrader as bt
from trade.train.utils import *
import numpy as np

class ModelIndicator(Indicator):
    '''
    Defined by John Bollinger in the 80s. It measures volatility by defining
    upper and lower bands at distance x standard deviations

    Formula:
      - midband = SimpleMovingAverage(close, period)
      - topband = midband + devfactor * StandardDeviation(data, period)
      - botband = midband - devfactor * StandardDeviation(data, period)

    See:
      - http://en.wikipedia.org/wiki/Bollinger_Bands
    '''
    alias = ('ModelIndicator',)

    lines = ('next_open',"inc")
    params = (('period', 61),)

    plotinfo = dict(subplot=False)
    plotlines = dict(
        # next_open=dict(ls='--'),
        next_open=dict(_samecolor=False),
        # bot=dict(_samecolor=True),
    )
    
    models = get_models("reg")

    # def _plotlabel(self):
    #     plabels = [self.p.period, self.p.devfactor]
    #     plabels += [self.p.movav] * self.p.notdefault('movav')
    #     return plabels

    def __init__(self):
        super(ModelIndicator, self).__init__()
        self.addminperiod(self.params.period)
        # data = {
        #     "close" : np.array(self.data.close.array),
        #     "open" : np.array(self.data.open.array),
        #     "volume" : np.array(self.data.volume.array),
        #     "high" : np.array(self.data.high.array),
        #     "low" : np.array(self.data.low.array),
        #     # "change" : np.array(self.data.change.array),
        # }
        
        # data["change"] =  np.concatenate([[float("nan")], data["close"][1:] / data["close"][:-1]])
        # # print(data)
        # model = None
        # for m in self.models.values():
        #     model = m
        
        # self.inc = model.predict(data).reshape([-1])
        # self.next_open = data["close"] * np.exp(self.inc)
        
    def next(self):
        # self.lines.next_open[0] = 
        
        model = None
        for m in self.models.values():
            model = m
        
        data = {
            "close" : np.array(self.data.close.get(size = self.p.period)),
            "open" : np.array(self.data.open.get(size = self.p.period)),
            "volume" : np.array(self.data.volume.get(size = self.p.period)),
            "high" : np.array(self.data.high.get(size = self.p.period)),
            "low" : np.array(self.data.low.get(size = self.p.period)),
            # "change" : np.array(self.data.change.array),
        }
        # for i in range(len(data["close"])):
        #     print(f"{i} ==> {data['close'][i]}")
        # print(np.array(self.data.close.array[:61]))
        # print(self.data.close[0], )
        # print(data)
        data["change"] =  np.concatenate([[float("nan")], data["close"][1:] / data["close"][:-1]])
        inc = np.exp(model.predict(data).reshape([-1]))
        next_open = data["close"] * inc
        self.lines.next_open[0] = next_open[-1]
        self.lines.inc[0] = inc[-1]
    
        # self.lines.next_open = self.data.close * 1.0
        # self.lines.next_open.array = next_open.tolist()
        # self.lines.next_open =  self.data.close * np.exp(inc)
        # self.lines.next_open.array =  np.empty(100_000, dtype=np.float64)
        
        # line_array = self.lines.next_open.array
        # # 创建内存映射的 NumPy 数组
        # self.np_memmap = np.frombuffer(line_array, dtype=np.float64)
        # # 将外部数据写入内存映射
        # self.np_memmap[:] = np.random.rand(len(line_array))  # 示例填充
        

        
