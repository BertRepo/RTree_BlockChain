import random

from blockchain import MerkleTree, RTree
from global_method import get_dataset, measure_insert_time_r_tree, measure_insert_time_mt, measure_search_time_r_tree, measure_search_time_mt


# 针对两种树结构本身 的 创建 查询的测试
def in_memory_test():
    num_transactions = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]    # 交易量设置
    # num_transactions = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # 交易量设置
    num_search_queries = 20  # 随机抽取50个

    # 初始化测试结果存储列表
    insert_time_results_r_tree = []
    insert_time_results_mt = []
    search_time_results_r_tree = []
    search_time_results_mt = []

    for n in num_transactions:
        print(f"当前交易数量：{n}")
        # 生成交易
        transactions = get_dataset(n)
        # 随机采样50个数据进行查找
        tx_to_search = random.sample([tx for tx in transactions], num_search_queries)

        # 测量插入时间
        insert_time_r_tree = measure_insert_time_r_tree(transactions)  # R树插入时间
        insert_time_mt = measure_insert_time_mt(transactions)  # Merkle树插入时间

        # 构建树结构
        # merkle_tree = MerkleTree(transactions)
        merkle_tree = MerkleTree()
        r_tree = RTree()
        for tx in transactions:
            merkle_tree.insert(tx)
            r_tree.insert(tx)

        print(tx_to_search)
        # 测量查找时间
        # merkle_transactions = merkle_tree.transactions
        search_time_list = measure_search_time_mt(transactions, tx_to_search)
        search_time_r_tree = measure_search_time_r_tree(r_tree, tx_to_search)


        # 记录结果
        insert_time_results_r_tree.append(insert_time_r_tree)
        insert_time_results_mt.append(insert_time_mt)
        search_time_results_r_tree.append(search_time_r_tree)
        search_time_results_mt.append(search_time_list)

        print("R树构建",insert_time_results_r_tree)
        print("M树构建",insert_time_results_mt)
        print("R树查询",search_time_results_r_tree)
        print("M树查询",search_time_results_mt)

    return num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt