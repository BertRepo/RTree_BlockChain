import string
import time
import random
import sys
import json
import numpy as np
import pandas as pd

from blockchain import BMTree, Transaction, MerkleTree, RTree
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator


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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
        attributeArray = data.iloc[i].values  # 使用转换后的数据作为交易属性值
        attribute = convert_to_serializable(attributeArray)

        # 假设我们将 'baby_birthday' 和 'baby_sex' 作为二维的边界属性
        bounds = (data.iloc[i]['baby_birthday'].timestamp(), 0 if data.iloc[i]['baby_sex'] == '男' else 1,
                  data.iloc[i]['baby_birthday'].timestamp(), 0 if data.iloc[i]['baby_sex'] == '男' else 1)
        transactions.append(Transaction(tx_hash, attribute, bounds))  # 将转换后的数据作为交易存储在区块中
    return transactions

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None  # 或者其他合适的值
        return float(obj)
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return obj.isoformat()  # 转换为 ISO 格式的字符串
    elif isinstance(obj, np.ndarray):
        # 转换 ndarray 为 list，并递归转换其中的元素
        return [convert_to_serializable(i) for i in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None  # 或者其他合适的值
        return obj
    else:
        return obj

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
    # MerkleTree(transactions)
    merkle_tree = MerkleTree(4)
    for tx in transactions:
        merkle_tree.insert(tx)
    end_time = time.time()
    return (end_time - start_time) * 1000

def measure_search_time_mt(m_tree, trans, attributes_to_search):
    """列表查询计时器"""
    start_time = time.time()
    for attr in attributes_to_search:
        m_tree.search(trans, attr)
    # for attr in attributes_to_search:
    #     next((tx for tx in tx_list if tx.attribute[3] == attr.attribute[3] and tx.attribute[5] == attr.attribute[5]), None)
        # next((tx for tx in tx_list if tx.bounds == attr), None)
    end_time = time.time()
    return (end_time - start_time) * 1000  # 转换为毫秒


def measure_insert_time_r_tree(transactions):
    """R树插入计算器"""
    start_time = time.time()
    r_tree = RTree(4)
    for tx in transactions:
        r_tree.insert(tx)
    end_time = time.time()
    return (end_time - start_time) * 1000


def measure_search_time_r_tree(r_tree, tx_to_search):
    """R树查询计算器"""
    start_time = time.time()
    for tx in tx_to_search:
        r_tree.search(tx.bounds)
    end_time = time.time()
    return (end_time - start_time) * 1000  # 转换为毫秒


def plot_results(num_transactions,
        insert_time_results_rtree,
        insert_time_results_merkle_tree,
        search_time_results_rtree,
        search_time_results_merkle_tree,
        storage_size_results_rtree,
        storage_size_results_merkle_tree,
                 is_block=False):
    """绘制对比图"""

    # 绘制插入时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, insert_time_results_rtree, label='BM树创建时间', marker='o')
    plt.plot(num_transactions, insert_time_results_merkle_tree, label='默克尔树创建时间', marker='x')
    if is_block:
        plt.xlabel('区块数量')
    else:
        plt.xlabel('交易数量')
    plt.ylabel('时间(s)')
    plt.title('创建时间比较')
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('insert.pdf', format='pdf')
    plt.show()

    # 绘制存储容量的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, storage_size_results_rtree, label='BM树存储大小', marker='o')
    plt.plot(num_transactions, storage_size_results_merkle_tree, label='默克尔树存储大小', marker='x')
    # plt.plot(num_transactions, storage_size_results_global_index, label='HGMB+树存储大小', marker='v')
    if is_block:
        plt.xlabel('区块数量')
    else:
        plt.xlabel('交易数量')
    plt.ylabel('大小(bytes)')
    plt.title('存储空间大小比较')
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('storage.pdf', format='pdf')
    plt.show()

    # 搜索比较
    plt.figure(figsize=(10, 5))
    if is_block:
        # TODO： 没修改完
        block_height_list = [1, 10, 20, 30, 40, 50, 60]
        # plt.plot(block_height_list, search_time_all_r_tree, label='R树查找时间', marker='o')
        plt.plot(block_height_list, [0.9, 3.8, 6.4, 7.0, 7.2, 7.4, 7.3], label='默克尔树查找时间', marker='x')
        # plt.plot(block_height_list, search_time_all_gb_tree, label='HGMB+树查找时间', marker='v')
        plt.xlabel('区块数量')
        plt.ylabel('性能占用(GB)')
        plt.title('区块链中查询时间比较')
        plt.legend()
        # 设置导出的文件名和格式为PDF
        plt.savefig('searchByHeight.pdf', format='pdf')
        plt.show()
    else:
        plt.plot(num_transactions, search_time_results_rtree, label='R树查找时间', marker='o')
        plt.plot(num_transactions, search_time_results_merkle_tree, label='默克尔树查找时间', marker='x')
        plt.xlabel('交易数量')
        plt.ylabel('时间(s)')
        plt.title('区块链中查询时间比较')
        ax = plt.gca()
        y_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(axis="y")
        plt.legend()
        # 设置导出的文件名和格式为PDF
        plt.savefig('searchByTranNum.pdf', format='pdf')
        plt.show()

def plot_tree_results(num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt):
    """绘制对比图"""
    plt.figure(figsize=(12, 6))

    # 绘制插入时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, insert_time_results_r_tree, label='R树创建时间', marker='o')
    plt.plot(num_transactions, insert_time_results_mt, label='默克尔树创建时间', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(ms)')
    plt.title('创建时间比较')

    # 设置纵坐标为对数刻度
    plt.yscale('log')
    # 设置纵坐标刻度位置
    plt.yticks([0.5, 1, 10, 100, 1000, 5000], ['0.5', '1', '10', '100', '1000', '5000'])

    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('insert_time_comparison.pdf', format='pdf')
    plt.show()

    # 绘制搜索时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, search_time_results_r_tree, label='R树查找时间', marker='o')
    plt.plot(num_transactions, search_time_results_mt, label='默克尔树查找时间', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(ms)')
    plt.title('查询时间比较')

    # # 设置纵坐标为对数刻度
    # plt.yscale('log')
    # # 设置纵坐标刻度位置
    # plt.yticks([0.5, 1, 10, 100, 1000, 5000], ['0.5', '1', '10', '100', '1000', '5000'])

    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('search_time_comparison.pdf', format='pdf')
    plt.show()


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
