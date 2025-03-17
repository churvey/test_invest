import time
from futu import *

class RTDataTest(RTDataHandlerBase):
    
    def __init__(self, flush_count = 100):
        super(RTDataTest).__init__()
        self._flush_count = flush_count
        self._count = 0
        self._cache = []
    
    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super(RTDataTest, self).on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            print("RTDataTest: error, msg: %s" % data)
            return RET_ERROR, data
        
        self._cache.append(data)
        self._count += 1
        if self._count % 100 == 1:
            print(f"receive data: {data}")
        if self._count % self._flush_count == 0:
            tmp = pd.concat(self._cache)
            tmp.to_csv(f"./cache/{self._count}.csv")
            self._cache = []

        # print("RTDataTest ", data) # RTDataTest 自己的处理逻辑
        # print(data)
        # print(type(data))
        return RET_OK, data
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
handler = RTDataTest()
quote_ctx.set_handler(handler)  # 设置实时分时推送回调
ret, data = quote_ctx.subscribe(['SZ.300750'], [SubType.RT_DATA]) # 订阅分时类型，OpenD 开始持续收到服务器的推送 RT_DATA K_1M  TICKER
if ret != RET_OK:
    print('error:', data)
time.sleep(60)  # 设置脚本接收 OpenD 的推送持续时间为15秒
# quote_ctx.close()   # 关闭当条连接，OpenD 会在1分钟后自动取消相应股票相应类型的订阅    
 