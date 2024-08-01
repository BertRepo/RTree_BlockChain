import string
import time
import random
import sys

from blockchain import BMTree, Transaction, MerkleTree, RTree
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd

# 加载字体
from matplotlib.font_manager import FontProperties

# 使用macOS自带的字体 PingFang SC
font_path = "/System/Library/Fonts/PingFang.ttc"
font_prop = FontProperties(fname=font_path)

matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()]
matplotlib.rcParams['font.family'] = 'sans-serif'

config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": [font_prop.get_name()],
}
matplotlib.rcParams.update(config)

filename = "./dataset/ahealthdata.csv"

# 随机生成哈希值
def random_string(length=10):
    """生成随机字符串作为交易哈希"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# TODO: 矩形边界设置有问题 感觉不对劲 另外也没有实现真的二维
def generate_transactions(num_transactions):
    """模拟生成交易"""
    transactions = []
    for i in range(num_transactions):
        tx_hash = random_string()
        attribute = random.randint(1, 10000)
        # 随机生成边界，假设为二维范围
        bounds = (random.randint(0, 100), random.randint(0, 100), random.randint(100, 200), random.randint(100, 200))
        transactions.append(Transaction(tx_hash, attribute, bounds))  # 添加bounds 二维的边界范围值
        print(tx_hash, attribute, bounds)
    return transactions

# 数据转换 针对数据集定制化
def transform_data(data):
    data['baby_birthday'] = pd.to_datetime(data['baby_birthday'], dayfirst=True)  # 将宝宝生日转换为日期时间格式
    data['baby_sex'] = data['baby_sex'].map({1: '男', 2: '女'})  # 将宝宝性别转换为男女
    return data

'''
    使用实验室的先心病儿童的一万条数据
    传入参数：交易数量、属性个数
'''
def get_dataset(num_transactions):
    data = pd.read_csv(filename)  # 读取CSV文件
    data = transform_data(data)  # 数据转换

    transactions = []
    for i in range(num_transactions):
        tx_hash = random_string()
        attribute = data.iloc[i].values  # 使用转换后的数据作为交易属性值
        # 假设我们将 'baby_birthday' 和 'baby_sex' 作为二维的边界属性
        bounds = (data.iloc[i]['baby_birthday'].timestamp(), 0 if data.iloc[i]['baby_sex'] == '男' else 1,
                  data.iloc[i]['baby_birthday'].timestamp(), 0 if data.iloc[i]['baby_sex'] == '男' else 1)
        transactions.append(Transaction(tx_hash, attribute, bounds))  # 将转换后的数据作为交易存储在区块中
        print(tx_hash, bounds)
    return transactions

def measure_insert_time_bm_tree(transactions, order):
    """ bmtree 插入计算器"""
    start_time = time.time()
    bm_tree = BMTree(order)
    for tx in transactions:
        bm_tree.insert(tx)
    end_time = time.time()
    return (end_time - start_time) * 1000

def measure_search_time_bm_tree(bm_tree, attributes_to_search):
    """ bm 查询计算器"""
    start_time = time.time()
    for attr in attributes_to_search:
        bm_tree.search(attr)
    end_time = time.time()
    return (end_time - start_time) * 1000

def measure_insert_time_mt(transactions):
    """列表插入计算器"""
    start_time = time.time()
    MerkleTree(transactions)
    end_time = time.time()
    return (end_time - start_time) * 1000

def measure_search_time_mt(tx_list, attributes_to_search):
    """列表查询计时器"""
    start_time = time.time()
    for attr in attributes_to_search:
        next((tx for tx in tx_list if tx.attribute == attr), None)
    end_time = time.time()
    return (end_time - start_time) * 1000  # 转换为毫秒


def measure_insert_time_r_tree(transactions):
    """R树插入计算器"""
    start_time = time.time()
    r_tree = RTree()
    for tx in transactions:
        r_tree.insert(tx)
    end_time = time.time()
    return (end_time - start_time) * 1000


def measure_search_time_r_tree(r_tree, bounds_to_search):
    """R树查询计算器"""
    start_time = time.time()
    for bounds in bounds_to_search:
        r_tree.search(bounds)
    end_time = time.time()
    return (end_time - start_time) * 1000  # 转换为毫秒


def plot_results(search_time_all_r_tree, search_time_all_mk_tree, search_time_all_gb_tree, num_transactions,
                 insert_time_results_bm_tree, insert_time_results_mt, search_time_results_bm_tree,
                 search_time_results_list, insert_time_results_global_index, search_time_results_global_index,
                 storage_size_results_bm_tree, storage_size_results_merkle_tree, storage_size_results_global_index,
                 is_block=False):
    """绘制对比图"""
    plt.figure(figsize=(12, 6))

    # # 绘制插入时间的比较
    # plt.figure(figsize=(10, 6))
    # plt.plot(num_transactions, insert_time_results_bm_tree, label='BM树创建时间', marker='o')
    # plt.plot(num_transactions, insert_time_results_mt, label='默克尔树创建时间', marker='x')
    # plt.plot(num_transactions, insert_time_results_global_index, label='HGMB+树创建时间', marker='v')
    # if is_block:
    #     plt.xlabel('区块数量')
    # else:
    #     plt.xlabel('交易数量')
    # plt.ylabel('时间(s)')
    # plt.title('创建时间比较')
    # plt.legend()
    # # 设置导出的文件名和格式为PDF
    # plt.savefig('my_plot1.pdf', format='pdf')
    # plt.show()

    # 绘制搜索时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, search_time_all_r_tree, label='R树查找时间', marker='o')
    plt.plot(num_transactions, search_time_all_mk_tree, label='默克尔树查找时间', marker='x')
    plt.plot(num_transactions, search_time_all_gb_tree, label='HGMB+树查找时间', marker='v')
    # plt.plot(num_transactions, [0.8, 0.9, 1.3, 1.4, 1.9], label='BM树查找时间', marker='o')
    # plt.plot(num_transactions, [1.5, 1.6, 1.9, 2.4, 2.8], label='默克尔树查找时间', marker='x')
    # plt.plot(num_transactions, [0.3, 0.3, 0.5, 0.5, 0.7], label='HGMB+树查找时间', marker='v')
    if is_block:
        plt.xlabel('区块数量')
    else:
        plt.xlabel('交易数量')
    plt.ylabel('时间(s)')
    plt.title('查询时间比较')
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('my_plot2.pdf', format='pdf')
    plt.show()

    # # 绘制存储容量的比较
    # plt.figure(figsize=(10, 6))
    # plt.plot(num_transactions, storage_size_results_bm_tree, label='BM树存储大小', marker='o')
    # plt.plot(num_transactions, storage_size_results_merkle_tree, label='默克尔树存储大小', marker='x')
    # plt.plot(num_transactions, storage_size_results_global_index, label='HGMB+树存储大小', marker='v')
    # if is_block:
    #     plt.xlabel('区块数量')
    # else:
    #     plt.xlabel('交易数量')
    # plt.ylabel('大小(bytes)')
    # plt.title('存储空间大小比较')
    # plt.legend()
    # # 设置导出的文件名和格式为PDF
    # plt.savefig('my_plot3.pdf', format='pdf')
    # plt.show()

    # # 绘制并发量的比较
    # plt.figure(figsize=(10, 5))
    # if is_block:
    #     block_height_list = [1, 10, 20, 30, 40, 50, 60]
    #     # plt.plot(block_height_list, search_time_all_r_tree, label='R树查找时间', marker='o')
    #     plt.plot(block_height_list, [0.9, 3.8, 6.4, 7.0, 7.2, 7.4, 7.3], label='默克尔树查找时间', marker='x')
    #     # plt.plot(block_height_list, search_time_all_gb_tree, label='HGMB+树查找时间', marker='v')
    #     plt.xlabel('并发数量')
    #     plt.ylabel('性能占用(GB)')
    #     plt.title('区块链中查询时间比较')
    #     plt.legend()
    #     # 设置导出的文件名和格式为PDF
    #     plt.savefig('my_plot4.pdf', format='pdf')
    #     plt.show()
    # else:
    #     num_transactions_list = [1, 10, 20, 30, 40, 50, 60]
    #     # plt.plot(num_transactions_list, search_time_all_r_tree, label='R树查找时间', marker='o')
    #     plt.plot(num_transactions_list, [1.0, 5.7, 7.1, 7.0, 7.2, 7.4, 7.3], label='Fabric', marker='x')
    #     plt.plot(num_transactions_list, [1.0, 2.1, 4.2, 6.2, 6.2, 6.3, 6.1], label='HGMB+树查找时间', marker='v')
    #     plt.xlabel('并发数量')
    #     plt.ylabel('性能占用(GB)')
    #     plt.title(' ')
    #     ax = plt.gca()
    #     y_major_locator = MultipleLocator(1)
    #     ax.yaxis.set_major_locator(y_major_locator)
    #     plt.grid(axis="y")
    #     plt.legend()
    #     # 设置导出的文件名和格式为PDF
    #     plt.savefig('my_plot4_1.pdf', format='pdf')
    #     plt.show()

    # # 绘制不同树高的查询比较
    # plt.figure(figsize=(10, 6))
    # tree_high = [3000, 4000, 5000, 6000, 7000]
    # plt.plot(tree_high, [0.003, 0.053, 0.234, 0.487, 0.712], label='BM树查找时间', marker='o')
    # plt.plot(tree_high, [0.0000052, 0.008, 0.051, 0.226, 0.476], label='默克尔树查找时间', marker='x')
    # plt.plot(tree_high, [0.0000052, 0.0000152, 0.0000252, 0.0000452, 0.0002352], label='HGMB+树查找时间', marker='v')
    # plt.xlabel('树高')
    # plt.ylabel('时间(s)')
    # plt.title('不同树高的查询比较')
    # plt.legend()
    # # 设置导出的文件名和格式为PDF
    # plt.savefig('my_plot5.pdf', format='pdf')
    # plt.show()
    # #  并发数量 | 内存占用 -- （4）100 200 300 400
    # plt.figure(figsize=(10, 6))
    # num_transactions_list = [1, 10, 20, 30, 40, 50, 60]
    # plt.plot(num_transactions_list, [0.8, 2.1, 3.9, 5.1, 6.8, 6.9, 7.4], label='HGMB+1')
    # plt.plot(num_transactions_list, [1.5, 4.1, 5.4, 6.1, 7.0, 7.3, 7.6], label='HGMB+2')
    # plt.plot(num_transactions_list, [2.4, 5.1, 6.3, 6.8, 7.4, 7.6, 7.8], label='HGMB+3')
    # plt.plot(num_transactions_list, [3.3, 5.6, 7.4, 7.5, 7.8, 7.9, 7.9], label='HGMB+4')
    # if is_block:
    #     plt.xlabel('并发数量')
    # else:
    #     plt.xlabel('并发数量')
    # plt.ylabel('性能占用(GB)')
    # plt.title(' ')
    # plt.legend(loc='upper center', ncol=4)
    # plt.grid(axis="y")
    # plt.legend()
    # # 设置导出的文件名和格式为PDF
    # plt.savefig('my_plot4.pdf', format='pdf')
    # plt.show()

    # #  key 值查询与 hash 值查询性能对比
    # plt.figure(figsize=(10, 6))
    # num_transactions_list = [1, 10, 20, 30, 40, 50, 60]
    # plt.plot(num_transactions_list, [0.03, 0.05, 0.09, 0.15, 0.23, 0.33, 0.45], label='根据hash查询')
    # plt.plot(num_transactions_list, [0.02, 0.04, 0.08, 0.13, 0.21, 0.31, 0.43], label='根据key查询')
    # if is_block:
    #     plt.xlabel('交易数/万条')
    # else:
    #     plt.xlabel('交易数/万条')
    # plt.ylabel('时间(s)')
    # plt.title(' ')
    # plt.grid(axis="y")
    # plt.legend()
    # # 设置导出的文件名和格式为PDF
    # plt.savefig('my_plot4_1.pdf', format='pdf')
    # plt.show()

    # # 条形图
    # plt.figure(figsize=(10, 6))
    # YBM = [1.7, 3.4, 5.1, 6.8, 8.5]
    # YMK = [1.4, 2.8, 3.2, 4.6, 6.0]
    # YHGMB = [2.1, 4.2, 6.3, 8.4, 10.5]
    # labels = ['100', '200', '300', '400', '500']
    # bar_width = 0.2
    # # 绘图
    # plt.bar(np.arange(5), YMK, label='Fabric', color='#96B0B7', alpha=0.8, width=bar_width)
    # plt.bar(np.arange(5) + bar_width, YBM, label='MerkleRBTree', color='#507D95', alpha=0.8, width=bar_width)
    # plt.bar(np.arange(5) + 0.4, YHGMB, label='本文方案', color='#FA9038', alpha=0.8, width=bar_width)
    # # 添加轴标签
    # plt.xlabel('区块数量')
    # plt.ylabel('大小(MB)')
    # # 添加标题
    # plt.title(' ')
    # # 添加刻度标签
    # plt.xticks(np.arange(5) + bar_width, labels)
    # # 显示图例
    # plt.legend()
    # plt.savefig('my_plot4_2.pdf', format='pdf')
    # # 显示图形
    # plt.show()

# def plot_results(num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt):
#     """绘制对比图"""
#     plt.figure(figsize=(12, 6))
#
#     # 绘制插入时间的比较
#     plt.figure(figsize=(10, 6))
#     plt.plot(num_transactions, insert_time_results_r_tree, label='R树创建时间', marker='o')
#     plt.plot(num_transactions, insert_time_results_mt, label='默克尔树创建时间', marker='x')
#     plt.xlabel('交易数量')
#     plt.ylabel('时间(ms)')
#     plt.title('创建时间比较')
#     plt.legend()
#     # 设置导出的文件名和格式为PDF
#     plt.savefig('insert_time_comparison.pdf', format='pdf')
#     plt.show()
#
#     # 绘制搜索时间的比较
#     plt.figure(figsize=(10, 6))
#     plt.plot(num_transactions, search_time_results_r_tree, label='R树查找时间', marker='o')
#     plt.plot(num_transactions, search_time_results_mt, label='默克尔树查找时间', marker='x')
#     plt.xlabel('交易数量')
#     plt.ylabel('时间(ms)')
#     plt.title('查询时间比较')
#     plt.legend()
#     # 设置导出的文件名和格式为PDF
#     plt.savefig('search_time_comparison.pdf', format='pdf')
#     plt.show()


def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # 添加到已见集合中
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


# if __name__ == '__main__':
#     get_dataset(9)