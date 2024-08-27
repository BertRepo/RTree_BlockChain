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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_mapping(df, column):
    """创建类别映射字典"""
    categories = df[column].unique()
    mapping = {category: idx for idx, category in enumerate(categories)}
    return mapping

def transform_data(data):
    """数据转换"""
    # 将 日期时间字段 列转换为日期时间格式
    data['baby_birthday'] = pd.to_datetime(data['baby_birthday'], dayfirst=True)
    data['screen_register_time'] = pd.to_datetime(data['screen_register_time'], dayfirst=True)
    data['report_add_date'] = pd.to_datetime(data['report_add_date'], dayfirst=True)
    data['dia_date'] = pd.to_datetime(data['dia_date'], dayfirst=True)

    # 将 'baby_sex' 列转换为'男'或'女'
    data['baby_sex'] = data['baby_sex'].map({1: '男', 2: '女', 3: '雄', 4: '雌'})

    return data

# def generate_bounds(row, d):
#     """生成边界的函数"""
#     # 假设 'baby_birthday' 转换为时间戳，'baby_sex' 转换为数字
#     birthday_timestamp = row['baby_birthday'].timestamp()
#     register_timestamp = row['screen_register_time'].timestamp()
#     report_timestamp = row['report_add_date'].timestamp()
#     dia_timestamp = row['dia_date'].timestamp()
#     sex_value = 1 if row['baby_sex'] == '男' else 2
#
#     # 生成边界（假设边界是基于时间戳和性别）
#     # 这里简单的边界生成示例，实际应用中可能会更加复杂
#     max_a = min_a = birthday_timestamp
#     max_b = min_b = sex_value
#     max_c = min_c = row['baby_weight']
#     max_d = min_d = row['pregnant_week']
#     max_e = min_e = row['upper_limbs_spo2']
#
#     # min_a = birthday_timestamp - np.random.randint(0, 10)
#     # max_a = birthday_timestamp + np.random.randint(0, 10)
#     # min_b = sex_value - np.random.randint(0, 1)
#     # max_b = sex_value + np.random.randint(0, 1)
#
#     return (min_a, min_b, max_a, max_b)

# '''
#     使用实验室的先心病儿童的一万条数据
#     传入参数：交易数量、属性个数
# '''
# def get_dataset(num_transactions, d=2):
#
#     # 读取CSV文件
#     data = pd.read_csv(filename)
#     # 数据转换
#     data = transform_data(data)
#
#     transactions = []
#     for i in range(num_transactions):
#         tx_hash = random_string()
#         attributeArray = data.iloc[i].values  # 使用转换后的数据作为交易属性值
#         attribute = convert_to_serializable(attributeArray)
#
#         # 生成边界
#         bounds = generate_bounds(data.iloc[i], d)
#
#         transactions.append(Transaction(tx_hash, attribute, bounds))  # 将转换后的数据作为交易存储在区块中
#     return transactions

def generate_bounds(row, attributes):
    """生成边界的函数"""
    bounds_min = []
    bounds_max = []
    for attr in attributes:
        if attr in ['baby_birthday', 'screen_register_time', 'report_add_date', 'dia_date']:
            timestamp = row[attr].timestamp()
            bounds_min.append(timestamp)
            bounds_max.append(timestamp)
            # bounds.append((timestamp, timestamp))  # (min, max)
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
            # bounds.append((sex_value, sex_value))  # (min, max)
        else:
            bounds_min.append(row[attr])
            bounds_max.append(row[attr])
            # bounds.append((row[attr], row[attr]))  # (min, max)
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

'''
    随机生成不存在的数据
'''
def generate_random_timestamp(start_date, end_date):
    """
    生成一个在 start_date 和 end_date 之间的随机时间戳。

    :param start_date: 开始日期 (datetime 对象)
    :param end_date: 结束日期 (datetime 对象)
    :return: 随机生成的日期 (datetime 对象)
    """
    # 确保开始日期早于结束日期
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")

    # 计算时间差
    delta = end_date - start_date

    # 生成一个随机的秒数
    random_seconds = random.uniform(0, delta.total_seconds())

    # 使用随机秒数生成新的日期时间
    random_date = start_date + timedelta(seconds=random_seconds)

    return random_date

# # 生成与原始数据类似的虚拟数据
# def generate_similar_data(data, num_transactions, attributes, d):
#     # 选择前 d 个属性
#     selected_attributes = attributes[:d]
#
#     transactions = []
#     for i in range(num_transactions):
#         tx_hash = random_string()  # 生成随机交易哈希
#         # 随机从 data 中复制一条记录
#         random_row = data.sample().iloc[0]
#         # 创建一个字典用于存储修改后的属性值
#         attribute_after = random_row.to_dict()  # 将随机行转为字典
#
#         for attr in selected_attributes:
#             if attr in ['baby_birthday', 'screen_register_time', 'report_add_date', 'dia_date']:
#                 # 处理时间戳类型的属性
#                 original_timestamp = pd.to_datetime(data[attr].iloc[i]).timestamp()  # 转换为时间戳
#                 attribute_after[attr] = original_timestamp + np.random.randint(-1000, 1000)
#             elif attr == 'baby_sex':
#                 # 处理性别属性
#                 attribute_after[attr] = random.choice(['雄', '雌'])
#             else:
#                 # 处理其他数值型属性
#                 attribute_after[attr] = data[attr].iloc[i] + np.random.randint(0, 20)
#
#         # 仅保留 selected_attributes 中对应的键值对
#         selected_attribute_values = [attribute_after[attr] for attr in selected_attributes]
#
#         # 将修改后的数据转为可序列化的格式
#         attribute = convert_to_serializable(selected_attribute_values)
#
#         # 生成边界
#         bounds = generate_bounds(random_row, selected_attributes)
#
#         # 将交易加入列表
#         transactions.append(Transaction(tx_hash, attribute, bounds))
#
#     return transactions
def generate_similar_data(data, num_transactions, attributes, d):
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


# 使用函数生成不存在的数据
def get_random_nonexistent_data(original_transactions, num_transactions, d):
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

def measure_insert_time_mt(transactions):
    """列表插入计算器"""
    start_time = time.time()
    # MerkleTree(transactions)
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