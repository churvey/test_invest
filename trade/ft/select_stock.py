from futu import *
import time
import pandas as pd
import os

def select_stock(path = "./cache/stock_select.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    all_list = []
    for market in [Market.SH]:
        acc_filter = AccumulateFilter ()
        acc_filter.days = 20
        acc_filter.filter_min = 1
        # acc_filter.filter_max = 100 * 20
        acc_filter.stock_field = StockField.TURNOVER_RATE
        acc_filter.is_no_filter = False
        acc_filter.sort = SortDir.DESCEND
        
        simple_filter = SimpleFilter ()
        simple_filter.filter_min = 100 * 1e8
        # simple_filter.filter_max = 100 * 20
        simple_filter.stock_field = StockField.MARKET_VAL
        simple_filter.is_no_filter = False
        # simple_filter.sort = SortDir.DESCEND
        nBegin = 0
        last_page = False
        ret_list = list()
        while not last_page:
            nBegin += len(ret_list)
            ret, ls = quote_ctx.get_stock_filter(market=market, filter_list=[acc_filter, simple_filter], begin=nBegin)  # 对香港市场的股票做简单、财务和指标筛选
            if ret == RET_OK:
                last_page, all_count, ret_list = ls
                print('all count = ', nBegin, "/",all_count)
                # for item in ret_list:
                #     # print(item.stock_code)  # 取股票代码
                #     # print(item.stock_name)  # 取股票名称
                #     # print(item[acc_filter])   # 取 simple_filter 对应的变量值
                #     # # print(item[financial_filter])   # 取 financial_filter 对应的变量值
                #     # print(item[simple_filter])  # 获取 custom_filter 的数值
                #     all_list.append(pd.DataFrame.from_dict({
                #         "stock_code":[item.stock_code],
                #         "stock_name":[item.stock_name],
                #         "turn_rate":[item[acc_filter] / 20],
                #         "market_val":[item[simple_filter] / 1e8],
                #     }))
                    
                ret_list = pd.DataFrame.from_dict({
                    "stock_code": [item.stock_code for item in ret_list],
                    "stock_name":[item.stock_name for item in ret_list],
                    "turnover_rate":[item[acc_filter] for item in ret_list],
                    "market_val":[item[simple_filter] / 1e8 for item in ret_list],
                })
            else:
                print('error: ', ls)
            # time.sleep(3)  # 加入时间间隔，避免触发限频
            all_list.append(ret_list)
    rs = pd.concat(all_list)   
    rs.to_csv(path)  
    quote_ctx.close() 
    return rs

if __name__ == "__main__":
    df = select_stock()
    print(df)
    print(len(df))


# # simple_filter = SimpleFilter()
# simple_filter.filter_min = 2
# simple_filter.filter_max = 1000
# simple_filter.stock_field = StockField.CUR_PRICE
# simple_filter.is_no_filter = False
# # simple_filter.sort = SortDir.ASCEND

# financial_filter = FinancialFilter()
# financial_filter.filter_min = 0.5
# financial_filter.filter_max = 50
# financial_filter.stock_field = StockField.CURRENT_RATIO
# financial_filter.is_no_filter = False
# financial_filter.sort = SortDir.ASCEND
# financial_filter.quarter = FinancialQuarter.ANNUAL

# custom_filter = CustomIndicatorFilter()
# custom_filter.ktype = KLType.K_DAY
# custom_filter.stock_field1 = StockField.MA10
# custom_filter.stock_field2 = StockField.MA60
# custom_filter.relative_position = RelativePosition.MORE
# custom_filter.is_no_filter = False


