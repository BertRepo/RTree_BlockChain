import sys
import string
import time
import random
import json
import numpy as np
import pandas as pd
from datetime import timedelta

from blockchain import Transaction, MerkleTree, RTree


# 数据集路径
filename = "./dataset/ahealthdata.csv"


# 随机生成哈希值
def random_string(length=10):
    """生成随机字符串作为交易哈希"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def convert_to_serializable(obj):
    """
    格式转换
    """
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


def create_mapping(df, column):
    """创建类别映射字典 用于转换数据集中数据"""
    categories = df[column].unique()
    mapping = {category: idx for idx, category in enumerate(categories)}
    return mapping


def transform_data(data):
    """数据转换 目前采用的转换方式"""
    # 将 日期时间字段 列转换为日期时间格式
    data['baby_birthday'] = pd.to_datetime(data['baby_birthday'], dayfirst=True)
    data['screen_register_time'] = pd.to_datetime(data['screen_register_time'], dayfirst=True)
    data['report_add_date'] = pd.to_datetime(data['report_add_date'], dayfirst=True)
    data['dia_date'] = pd.to_datetime(data['dia_date'], dayfirst=True)

    # 将 'baby_sex' 列转换为'男'或'女'
    data['baby_sex'] = data['baby_sex'].map({1: '男', 2: '女', 3: '雄', 4: '雌'})
    return data


def generate_bounds(row, attributes):
    """基于交易数据生成边界"""
    bounds_min = []
    bounds_max = []
    for attr in attributes:
        if attr in ['baby_birthday', 'screen_register_time', 'report_add_date', 'dia_date']:
            timestamp = row[attr].timestamp()
            bounds_min.append(timestamp)
            bounds_max.append(timestamp)
        elif attr == 'baby_sex':
            if row[attr] == '男':
                sex_value = 1
            elif row[attr] == '女':
                sex_value = 2
            elif row[attr] == '雄':
                sex_value = 3
            elif row[attr] == '雌':
                sex_value = 4
            else:
                sex_value = 5
            bounds_min.append(sex_value)
            bounds_max.append(sex_value)
        else:
            bounds_min.append(row[attr])
            bounds_max.append(row[attr])
    bounds = np.append(bounds_min, bounds_max)
    return tuple(bounds)


'''
    使用实验室的先心病儿童的一万条数据
    传入参数：交易数量、属性个数
'''
def get_dataset(num_transactions, d=2):
    """生成数据集"""
    # 读取CSV文件
    data = pd.read_csv(filename)
    # 数据转换
    data = transform_data(data)
    # 获取所有属性列表
    attributes = ['baby_birthday', 'baby_sex', 'baby_weight', 'pregnant_week',
                  'upper_limbs_spo2', 'lower_limbs_spo2', 'born_type', 'baby_num',
                  'screen_register_time', 'dia_result', 'dia_date', 'report_add_date',
                  'dia_advice_type', 'patient_type']

    # # 随机选择d个属性
    # selected_attributes = random.sample(attributes, d)
    # 选择前d个属性
    selected_attributes = attributes[:d]
    transactions = []
    for i in range(num_transactions):
        tx_hash = random_string()
        attributeArray = data[selected_attributes].iloc[i].values  # 使用转换后的数据作为交易属性值
        attribute = convert_to_serializable(attributeArray)
        # 生成边界
        bounds = generate_bounds(data.iloc[i], selected_attributes)
        transactions.append(Transaction(tx_hash, attribute, bounds))  # 将转换后的数据作为交易存储在区块中

    return transactions


def generate_similar_data(data, num_transactions, attributes, d):
    """
    生成虚假交易
    """
    # 选择前 d 个属性
    selected_attributes = attributes[:d]
    transactions = []
    for i in range(num_transactions):
        tx_hash = random_string()  # 生成随机交易哈希
        # 随机从 data 中复制一条记录
        random_row = data.sample().iloc[0]
        # 创建一个字典用于存储修改后的属性值
        attribute_after = random_row.to_dict()  # 将随机行转为字典

        for attr in selected_attributes:
            if attr in ['baby_birthday', 'screen_register_time', 'report_add_date', 'dia_date']:
                # 处理时间戳类型的属性
                original_timestamp = pd.to_datetime(data[attr].iloc[i])
                new_timestamp = original_timestamp + pd.to_timedelta(np.random.randint(-1000, 1000), unit='s')
                attribute_after[attr] = new_timestamp.isoformat()  # 转换为 ISO 8601 格式的字符串
            elif attr == 'baby_sex':
                # 处理性别属性
                attribute_after[attr] = random.choice(['雄', '雌'])
            else:
                # 处理其他数值型属性
                attribute_after[attr] = data[attr].iloc[i] + np.random.randint(0, 20)

        # 仅保留 selected_attributes 中对应的键值对
        selected_attribute_values = [attribute_after[attr] for attr in selected_attributes]
        # 将修改后的数据转为可序列化的格式
        attribute = convert_to_serializable(selected_attribute_values)
        # 生成边界
        bounds = generate_bounds(random_row, selected_attributes)
        # 将交易加入列表
        transactions.append(Transaction(tx_hash, attribute, bounds))

    return transactions


def get_random_nonexistent_data(original_transactions, num_transactions, d):
    """
    生成虚假数据
    """
    data = pd.read_csv(filename)  # 读取原始数据CSV文件
    data = transform_data(data)  # 数据转换
    # cv_attribute = original_transactions[0].attribute
    # 获取所有属性列表
    cv_attribute = ['baby_birthday', 'baby_sex', 'baby_weight', 'pregnant_week',
                  'upper_limbs_spo2', 'lower_limbs_spo2', 'born_type', 'baby_num',
                  'screen_register_time', 'dia_result', 'dia_date', 'report_add_date',
                  'dia_advice_type', 'patient_type']
    # 生成与原始数据类似的虚拟数据
    random_transactions = generate_similar_data(data, num_transactions, cv_attribute, d)
    # 确保生成的数据与原始数据不重复
    existing_hashes = {tx.tx_hash for tx in original_transactions}
    filtered_transactions = [tx for tx in random_transactions if tx.tx_hash not in existing_hashes]
    # TODO: 检查一下过滤后的交易列表的长度
    return filtered_transactions

# TODO: 修改
def generate_history_data(num_transactions, d, num_history, n):
    """
    生成历史数据集
    """
    # 生成 num_transactions 条交易
    df = pd.read_csv(filename)
    data = df.sample(n=num_transactions)
    transactions = []
    # blockchain = []  # 用于存储区块链

    # 获取所有属性列表
    attributes = ['baby_birthday', 'baby_sex', 'baby_weight', 'pregnant_week',
                  'upper_limbs_spo2', 'lower_limbs_spo2', 'born_type', 'baby_num',
                  'screen_register_time', 'dia_result', 'dia_date', 'report_add_date',
                  'dia_advice_type', 'patient_type']
    selected_attributes = attributes[:d]

    for i in range(num_transactions):
        tx_hash = random_string()
        attributeArray = data[selected_attributes].iloc[i].values  # 使用转换后的数据作为交易属性值
        attribute = convert_to_serializable(attributeArray)
        # 生成边界
        bounds = generate_bounds(data.iloc[i], selected_attributes)
        transactions.append(Transaction(tx_hash, attribute, bounds))  # 将转换后的数据作为交易存储在区块中
    #     original_row = data.iloc[i]
    #     original_row['id'] = i  # 为每条数据添加 id 字段
    #     print(original_row)
    #     history_transactions = []
    #
    #     for j in range(num_history):
    #         tx_hash = random_string()  # 生成随机交易哈希
    #
    #         modified_row = original_row.copy()
    #
    #         for attr in selected_attributes:
    #             if attr in ['baby_birthday', 'screen_register_time', 'report_add_date', 'dia_date']:
    #                 # 处理时间戳类型的属性
    #                 new_timestamp = pd.to_datetime(modified_row[attr]) + pd.to_timedelta(np.random.randint(-1000, 1000),
    #                                                                                      unit='s')
    #                 modified_row[attr] = new_timestamp.isoformat()  # 转换为 ISO 8601 格式的字符串
    #             elif attr == 'baby_sex':
    #                 # 处理性别属性
    #                 modified_row[attr] = random.choice(['雄', '雌'])
    #             else:
    #                 # 处理其他数值型属性
    #                 modified_row[attr] = modified_row[attr] + np.random.randint(0, 20)
    #
    #         # 将修改后的数据转为可序列化的格式
    #         attribute = convert_to_serializable(modified_row[selected_attributes].tolist())
    #         print(modified_row)
    #         # 生成边界
    #         bounds = generate_bounds(modified_row, selected_attributes)
    #         # 将交易加入列表
    #         history_transactions.append(Transaction(tx_hash, attribute, bounds, modified_row['id']))
    #
    #     transactions.extend(history_transactions)
    #
    # # 打乱 transactions 列表
    # random.shuffle(transactions)

    return transactions

    # # 将交易插入区块链中
    # current_block = []
    # for tx in transactions:
    #     # 如果当前区块中的交易数量达到 n，则将当前区块添加到区块链，并重置当前区块
    #     if len(current_block) == n:
    #         blockchain.append(current_block)
    #         current_block = []
    #
    #     # 查找该交易的 id 是否已经在区块链中
    #     tx_ids = [t.id for block in blockchain for t in block]
    #     prev_hash = tx_ids.index(tx.id) if tx.id in tx_ids else -1
    #     tx.prev_hash = prev_hash  # 更新 prev_hash 字段
    #     current_block.append(tx)
    #
    # # 如果还有未插入区块链的交易，将其插入最后一个区块
    # if current_block:
    #     blockchain.append(current_block)
    #
    # return blockchain


def measure_insert_time_mt(transactions):
    """列表插入计时器"""
    start_time = time.time()
    merkle_tree = MerkleTree(4)
    for tx in transactions:
        merkle_tree.insert(tx)
    end_time = time.time()
    return (end_time - start_time) * 1000


def measure_search_time_mt(m_tree, trans, attributes_to_search, d):
    """列表查询计时器"""
    start_time = time.time()
    for attr in attributes_to_search:
        m_tree.search(trans, attr, d)
    end_time = time.time()
    return (end_time - start_time) * 1000  # 转换为毫秒


def measure_insert_time_r_tree(transactions):
    """R树插入计时器"""
    start_time = time.time()
    r_tree = RTree(4)
    for tx in transactions:
        r_tree.insert(tx)
    end_time = time.time()
    return (end_time - start_time) * 1000


def measure_search_time_r_tree(r_tree, tx_to_search):
    """R树查询计时器"""
    start_time = time.time()
    for tx in tx_to_search:
        r_tree.search(tx.bounds)
    end_time = time.time()
    return (end_time - start_time) * 1000  # 转换为毫秒