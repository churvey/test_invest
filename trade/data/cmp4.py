from trade.data.loader import QlibDataloader, FtDataloader
import os
import numpy as np
import pandas as pd
import datetime
    
    
# def t_test_series(x):
#     n = len(x)
#     T = np.zeros(n)
#     for i in range(2, n - 2):  # 忽略首尾
#         x1, x2 = x[:i], x[i:]
#         mean1, mean2 = np.mean(x1), np.mean(x2)
#         std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
#         # pooled_std = np.sqrt((std1**2 / i) + (std2**2 / (n - i)))
#         n1 = len(x1) - 1
#         n2 = len(x2) - 1
#         sd = (std1**2 * n1  + std2**2 * n2) / (n1 + n2) * (1 / (n1 + 1) + 1 / (n2 +1))
#         T[i] = abs(mean1 - mean2) / np.sqrt(sd)
#     return T

def t_test_series(y, short=3, long = 2):
    n = len(y)
    T = np.full(n, float("inf"))
    x = np.arange(n)
    
    for i in range(long, n - long):  # 忽略首尾
        
        def v(left, right):
            l = max(0, i-left)
            r = min(len(x), i+right)
            y1, y2, y3 = y[l:i], y[i:r], y[l:r]
            x1, x2, x3 = x[l:i], x[i:r], x[l:r]
            # print(i, len(x), len(x1), len(x2))
            co1 = np.polyfit(x1, y1, deg = 1)
            y1_ = np.polyval(co1, x1)
            co2 = np.polyfit(x2, y2, deg = 1)
            y2_ = np.polyval(co2, x2)
            co3 = np.polyfit(x3, y3, deg = 1)
            y3_ = np.polyval(co3, x3)
            
            
            # print(co1, co2)
            res =  (np.sum((y1 - y1_) * (y1 - y1_)) + np.sum((y2 - y2_) * (y2 - y2_))) / np.sum((y3 -y3_) * (y3 - y3_))
            # v = np.sqrt(res) / abs(co1[0] - co2[0]) 
            # # if v < 0:
            # print(co1[0] - co2[0], v, res)
            # return v
            return res
        
        T[i] = max(v(short, long), v(long, short))
        
    return T



class StateMachine():
    def __init__(self):
        
        self.bars = [f"{p}min" for p in [1, 5, 30, 60]] + ["1d"]
        self.last_time = {p:None for p in self.bars}
        self.values = {p:{} for p in self.bars}
        self.states = {p:{} for p in self.bars}
        self.pending = {p:None for p in self.bars}
        self.extend_feature=["vma", "ma", "std", "z_bollinger", "z_pos__"]
        self.window_size = 20
        
    
    def update_state(self, bar):
        data = {k: np.array(v) for k,v in self.values[bar].items()}
        from .feature.feature import Feature
        data = Feature(data=data, features=self.extend_feature, rolling_window=[self.window_size])()
        # self.states
        for k, v in data.items():
            if k in self.states[bar]:
                self.states[bar][k] = np.append(self.states[bar][k], [v[-1]])
                
            else:
                self.states[bar][k] = np.array([v[-1]])
            
        if len(self.states[bar]["close"]) > self.window_size:
            self.states[bar] = {
                k: v[-self.window_size:] for k, v in self.states[bar].items()
            }
            if bar == "1d" and np.all(self.states[bar][f"z_pos_{self.window_size}"] >= 8 ):
                print("bar", bar,self.states[bar]["datetime"][-1], self.states[bar][f"z_pos_{self.window_size}"])
            # if self.states[bar]["datetime"] == datetime.strptime("2025-05-27 15:00:00", '%m-%d-%Y %H:%M:%S'):
                for k, v in self.states[bar].items():
                    print(k, "===>", v)
        
        
    def next(self, data):
        if isinstance(data["datetime"], str):         
            dt = datetime.strptime(data["datetime"], '%m-%d-%Y %H:%M:%S')
        else:
            dt = data["datetime"]
        if dt.second != 0 or dt.microsecond != 0:
            return   
        
        is_open = (dt.minute == 30 and dt.hour == 9)
        is_close = (dt.minute == 0 and dt.hour == 15)
        
        
        for _, bar in enumerate(self.bars):
            if self.pending[bar] is None:
                self.pending[bar] = data
            else:
                self.pending[bar]["high"] = max(self.pending[bar]["high"], data["high"])
                self.pending[bar]["low"] = min(self.pending[bar]["low"], data["low"])
                self.pending[bar]["volume"] += data["volume"]
            
            def is_last_point():
                if bar.endswith("min"):
                    if is_open:
                        return False
                    c = int(bar[:-3])
                    return dt.minute % c == 0
                elif bar.endswith("d"):
                    return is_close
                raise ValueError(f"not supported bar:{bar}")
                
            if is_last_point():
                self.pending[bar]["close"] = data["close"]
                self.pending[bar]["datetime"] = data["datetime"]
                for k, v in self.pending[bar].items():
                    if k in self.values[bar]:
                        self.values[bar][k].append(v)
                    else:
                        self.values[bar][k] = [v]
                
                if len(self.values[bar]["close"]) > self.window_size:
                    self.values[bar] = {
                        k: v[-self.window_size:] for k, v in self.values[bar].items()
                    }
                self.pending[bar] = None
                self.update_state(bar)

                
                
                    
                       
            
        

        


if __name__ == "__main__":    
    # f1 = FtDataloader("./tmp/2", [], extend_feature=["vma", "ma", "std", "z_bollinger"], rolling_window=[20], down_sample=[]).features
    down_sample = [None, "5m", "15m", "30m", "60m", "120m", "1d", "1w"]
    down_sample =  [None, "5min"]
    f2 = FtDataloader("./etf2", [], extend_feature=False).features
    print(f2)
    
    machine = StateMachine()
    for i, row in f2.iterrows():
        machine.next(row)
        if i % 1000 == 0:
            print(f"prcocessed {i} {row['datetime']}")
    
    
    # def f(f1):
    #     f1 = f1[f1["datetime"] >= "2025"]
        
    #     # f1 = f1[f1["datetime"] < "2025-02-25"]
    #     for c in ["ma_20", "bollinger_u_20",  "bollinger_d_20"]:
    #         f1[c] = f1[c] * f1["close"]
    #     return f1

    # f1 = f(f1)
    # f2 = f(f2)
    
   
    
    # print(f1[["datetime","ma_20"]].head(10))
    
    # print(f2[["datetime","ma_20"]].head(10))
    
    # print(f2)
    
    # std = f2["ma_20"].to_numpy()
    # datetime = f2["datetime"].to_numpy()
    
    # def cut(s, d):
    #     if  len(s) < 10:
    #         return []
    #     else:
    #         rs = []
    #         T = t_test_series(s)
    #         k = np.argmin(T)  
    #         d2 = np.datetime_as_string(d[k], unit="D")
    #         rs.append((d2, T[k]))
    #         rs.extend(
    #             cut(s[:k], d[:k])
    #         )
    #         rs.extend(
    #             cut(s[k+1:], d[k+1:])
    #         )
    #         return rs
    
    # print(cut(std, datetime))
   
    
    # for std in ["std_20_5min", "std_20"]:
    #     binned = pd.qcut(f1[std], q=4)
    #     counts = binned.value_counts(sort=False)
    #     print(counts)
      
      


    # T = t_test_series(y)
    # k = np.argmax(T)  # 切分点为 T 最大的位置