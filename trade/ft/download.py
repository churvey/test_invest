from futu import *
import pandas as pd
import ray
import time


def get_selected():

    selected = """              
    588000    科创50ETF          
    159915    创业板ETF          
    513180    恒生科技指数ETF    
    510300    沪深300ETF         
    588200    科创芯片ETF        
    159338    中证A500ETF        
    512100    中证1000ETF        
    513050    中概互联网ETF      
    510050    XD上证50ETF        
    512880    证券ETF            
    518880    黄金ETF            
    512480    半导体ETF          
    159605    中概互联ETF        
    159995    芯片ETF            
    159941    纳指ETF            
    512690    酒ETF              
    562500    机器人ETF          
    159869    游戏ETF            
    512010    医药ETF            
    159928    消费ETF            
    563300    中证2000ETF        
    159922    中证500ETF         
    159819    人工智能ETF        
    512200    房地产ETF          
    510880    红利ETF            
    588800    科创100ETF华夏     
    588060    科创50ETF龙头      
    512710    军工龙头ETF        
    515880    通信ETF            
    512980    传媒ETF            
    510760    上证综指ETF        
    515220    煤炭ETF            
    512670    国防ETF            
    159857    光伏ETF            
    159611    电力ETF            
    516510    云计算ETF          
    159638    高端装备ETF        
    561980    半导体设备ETF      
    560080    中药ETF            
    513910    港股央企红利ETF    
    516950    基建ETF            
    515290    银行ETF天弘        
    516670    畜牧养殖ETF        
    560980    光伏30ETF          
    159637    新能源车龙头ETF    
    159622    创新药ETF沪港深    
    """


    selected = selected.split("\n")
    selected = [v.strip() for v in selected if v.strip()]
    print(selected)
    selected = {v.split()[0]: v.split()[1] for v in selected}


    selected2 = {}
    for k, v in selected.items():
        if k[0] == "1":
            pre = "SZ"
        else:
            pre = "SH"
        selected2[f"{pre}.{k}"] = v
    selected = selected2


    print(selected)

    stocks = pd.read_csv("./cache/stock_select.csv")

    codes = stocks["stock_code"].to_list()
    names = stocks["stock_name"].to_list()

    selected.update(
        dict(zip(codes, names))
    )
    
    return selected

@ray.remote
class Query:
    def __init__(self, ktype = KLType.K_1M, n_parallel=24):
        self.quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        self.ktype = ktype
        self.n_parallel = n_parallel
        self.least_cost = 0.5 * n_parallel
        
    def query(self, codes, start="2012-01-01", end="2025-02-25", max_count=1000, use_cache=True):
        # ktype = KLType.K_DAY
        ktype = self.ktype
        autype = AuType.QFQ
        all_codes = list(codes.keys())
        for i, code in enumerate(all_codes):
            begin = time.time()
            ktype_str = str(ktype).split(".")[-1]
            dirs =  f"data/{ktype_str}"
            os.makedirs(dirs, exist_ok=True)
            save_path = f"{dirs}/{code}-{codes[code]}.csv"
            if use_cache and os.path.exists(save_path):
                print(f"already cached will skip {save_path}")
                continue
            page_req_key = None
            total_data = []
            while True:
                if page_req_key is None:
                    ret, data, page_req_key = self.quote_ctx.request_history_kline(
                        code,
                        start=start,
                        end=end,
                        ktype=ktype,
                        autype=autype,
                        max_count=max_count,
                    )  # 每页5个，请求第一页
                else:
                    ret, data, page_req_key = self.quote_ctx.request_history_kline(
                        code,
                        start=start,
                        end=end,
                        ktype=ktype,
                        max_count=max_count,
                        autype=autype,
                        page_req_key=page_req_key,
                    )  # 每页5个，请求第一页
                if ret == RET_OK:
                    # print(data)
                    total_data.append(data)
                    print(f"got data {codes[code]} {len(total_data) * max_count}")
                else:
                    raise Exception(f"error:{data}")
                if page_req_key is None:
                    break
            data_save = pd.concat(total_data)
            data_save.to_csv(save_path)
            # all.append(data_save)
            
            finished_time = time.time()
            print(f"finish save {i} ==> {code} cost:{finished_time - begin}")
            if finished_time - begin < self.least_cost:
                sleep_time = self.least_cost - (finished_time - begin)
                print(f"will sleep {sleep_time}")
                time.sleep(sleep_time)


def query_data(codes, start="2012-01-01", end="2025-02-25", max_count=1000, use_cache=True):
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    # all = []
    # ktype = KLType.K_DAY
    ktype = KLType.K_1M
    autype = AuType.QFQ
    all_codes = list(codes.keys())
    for i, code in enumerate(all_codes):
        ktype_str = str(ktype).split(".")[-1]
        dirs =  f"data/{ktype_str}"
        os.makedirs(dirs, exist_ok=True)
        save_path = f"{dirs}/{code}-{codes[code]}.csv"
        if use_cache and os.path.exists(save_path):
            print(f"already cached will skip {save_path}")
            continue
        page_req_key = None
        total_data = []
        while True:
            if page_req_key is None:
                ret, data, page_req_key = quote_ctx.request_history_kline(
                    code,
                    start=start,
                    end=end,
                    ktype=ktype,
                    autype=autype,
                    max_count=max_count,
                )  # 每页5个，请求第一页
            else:
                ret, data, page_req_key = quote_ctx.request_history_kline(
                    code,
                    start=start,
                    end=end,
                    ktype=ktype,
                    max_count=max_count,
                    autype=autype,
                    page_req_key=page_req_key,
                )  # 每页5个，请求第一页
            if ret == RET_OK:
                # print(data)
                total_data.append(data)
                print(f"got data {codes[code]} {len(total_data) * max_count}")
            else:
                raise Exception(f"error:{data}")
            if page_req_key is None:
                break
        data_save = pd.concat(total_data)
        data_save.to_csv(save_path)
        # all.append(data_save)
        print(f"finish save {i} ==> {code}")
    # all = pd.concat(all)
    # all.to_csv(f"data/{ktype_str}.csv")
    quote_ctx.close()  # 结束后记得关闭当条连接，防止连接条数用尽



def query_parallel(codes, ktype = KLType.K_1M,  n_parallel = 24):
    keys = list(codes.keys())[:1000]
    querier = [Query.remote(ktype, n_parallel) for _ in range(n_parallel)]
    waiting = []
    for i in range(n_parallel):
        code_i = {
            key:codes[key] for key in keys[i::n_parallel]
        }
        waiting.append(querier[i].query.remote(code_i))
    ray.get(waiting)


if __name__ == "__main__":
    selected = get_selected()
    # query_data(selected)
    query_parallel(selected)