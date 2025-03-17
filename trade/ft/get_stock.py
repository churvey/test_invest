from futu import *
import time
import pandas as pd
import os

def get_stocks(path = "./cache/stock.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    all_list = []
    for market in [Market.SH, Market.SZ]:
        # nBegin = 0
        # last_page = False
        # ret_list = list()
        
        # while not last_page:
            # nBegin += len(ret_list)
            # ret, ls = quote_ctx.get_stock_filter(market=market, filter_list=[], begin=nBegin)  # 对香港市场的股票做简单、财务和指标筛选
            # quote_ctx.get_stock_basicinfo(market, stock_type=ft.SecurityType.STOCK)
            
        ret_code, ret_data = quote_ctx.get_stock_basicinfo(market, stock_type=SecurityType.STOCK)
        if ret_code == 0:
            print("get_stock_basicinfo: market={}, count={}".format(market, len(ret_data)))
            print(f"ret_data {type(ret_data)}")
            # for ix, row in ret_data.iterrows():
            #     stock_codes.append(row['code'])
            all_list.append(ret_data)

            # if ret == RET_OK:
            #     last_page, all_count, ret_list = ls
            #     all_list.extend(ret_list)
            #     print('all count = ', all_count)
            #     for item in ret_list:
            #         print(item.stock_code,"\t", item.stock_name)  # 取股票代码, 取股票名称
            # else:
            #     print('error: ', ls)
            time.sleep(3)  # 加入时间间隔，避免触发限频
            
    print(all_list)
    rs = pd.concat(all_list)
    rs.to_csv(path)
            
    quote_ctx.close() 
    
def get_stock2(path = "./cache/stock2.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data = quote_ctx.get_plate_stock('SH.LIST0600')
    if ret == RET_OK:
        pass
        # print(data)
        # print(data['stock_name'][0])    # 取第一条的股票名称
        # print(data['stock_name'].values.tolist())   # 转为 list
    else:
        print('error:', data)
    quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽
    data.to_csv(path)
    return data

if __name__ == "__main__":
    df = get_stocks()
    print(df)
    df2 = get_stock2()
    print(df2)


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


