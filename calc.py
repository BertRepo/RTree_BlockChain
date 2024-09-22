import time
import json
import random
import pickle
import couchdb

# 导入自定义包
from blockchain import Transaction, Blockchain, Block, RTree, MerkleTree, MerkleTreeBlock
from data import get_dataset, get_random_nonexistent_data, generate_history_data, measure_insert_time_r_tree, measure_insert_time_mt, measure_search_time_r_tree, measure_search_time_mt

# 判断两个边界是否相交
def intersects(mbr1, mbr2):
    return all(mbr1[i] <= mbr2[i + len(mbr1) // 2] and mbr2[i] <= mbr1[i + len(mbr1) // 2] for i in range(len(mbr1) // 2))

# 用于溯源 从区块链查询所有历史数据 ———— 适用于前两种方案
def get_history_by_id(t, chain, res):
    """
    从区块链中溯源t的历史交易记录
    :param t: 需要溯源的交易
    :param chain: 区块链
    :param res: 用于存储结果集
    :return: 符合条件的记录列表
    """
    # 检查 t 是否有 attribute 和 pre_index，并且 pre_index 大于 -1
    if t.attribute and t.pre_index > -1:
        block = chain[t.pre_index]  # 获取包含前一个交易的区块
        # 遍历 tx_batch 找到属性值中id相同的交易
        for tx in block.transactions:
            if tx.attribute[-1] == t.attribute[-1] and tx.pre_index != t.pre_index:  # 匹配倒数第一个元素 即id
                res.append(tx)  # 当前交易加入结果集
                get_history_by_id(tx, chain, res)  # 递归查找历史交易并加入结果集
    print('打印', len(res))
    return res

# 用于溯源 从历史数据库查询所有历史数据 ———— 适用于后两种方案
def query_transactions(db, match_attributes):
    """
    从fabric_db中查询与match_attributes匹配的交易记录
    :param db: 数据库连接
    :param match_attributes: 用于匹配的属性列表（不包括最后一个元素）
    :return: 符合条件的记录列表
    """
    result = []
    for doc_id in db:
        doc = db[doc_id]
        if 'transactions' in doc:
            for transaction in doc['transactions']:
                # 比较前n-1个元素是否相等
                if transaction['attribute'][:-1] == match_attributes[:-1]:
                    result.append(transaction)
    return result

def save_experiment_results(
    d_list,
    n_list,
    insert_time_rtree,
    insert_time_rtree_mbr,
    insert_time_merkle_tree,
    storage_size_rtree,
    storage_size_rtree_mbr,
    storage_size_merkle_tree,
    search_time_rtree,
    search_time_rtree_mbr,
    search_time_merkle_tree,
    search_no_time_rtree,
    search_no_time_rtree_mbr,
    search_no_time_merkle_tree,
    file_path="experiment_results.json"):
    # 构建一个字典存储所有结果
    results = {
        "d_list": d_list,
        "n_list": n_list,
        "insert_time_rtree": insert_time_rtree,
        "insert_time_rtree_mbr": insert_time_rtree_mbr,
        "insert_time_merkle_tree": insert_time_merkle_tree,
        "storage_size_rtree": storage_size_rtree,
        "storage_size_rtree_mbr": storage_size_rtree_mbr,
        "storage_size_merkle_tree": storage_size_merkle_tree,
        "search_time_rtree": search_time_rtree,
        "search_time_rtree_mbr": search_time_rtree_mbr,
        "search_time_merkle_tree": search_time_merkle_tree,
        "search_no_time_rtree": search_no_time_rtree,
        "search_no_time_rtree_mbr": search_no_time_rtree_mbr,
        "search_no_time_merkle_tree": search_no_time_merkle_tree
    }

    # 将字典保存为 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"实验结果已保存到 {file_path}")

# 保存溯源实验结果
def save_history_results(
    d_list,
    n_list,
    insert_time_rtree,
    insert_time_rtree_mbr,
    insert_time_fabric,
    insert_time_fabric_sort,
    storage_size_rtree,
    storage_size_rtree_mbr,
    storage_size_fabric,
    storage_size_fabric_sort,
    search_time_rtree,
    search_time_rtree_mbr,
    search_time_fabric,
    search_time_fabric_sort,
    file_path="experiment_results.json"
):
    # 构建一个字典存储所有结果
    results = {
        "d_list": d_list,
        "n_list": n_list,
        "insert_time_rtree": insert_time_rtree,
        "insert_time_rtree_mbr": insert_time_rtree_mbr,
        "insert_time_fabric": insert_time_fabric,
        "insert_time_fabric_sort": insert_time_fabric_sort,
        "storage_size_rtree": storage_size_rtree,
        "storage_size_rtree_mbr": storage_size_rtree_mbr,
        "storage_size_fabric": storage_size_fabric,
        "storage_size_fabric_sort": storage_size_fabric_sort,
        "search_time_rtree": search_time_rtree,
        "search_time_rtree_mbr": search_time_rtree_mbr,
        "search_time_fabric": search_time_fabric,
        "search_time_fabric_sort": search_time_fabric_sort
    }

    # 将字典保存为 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"实验结果已保存到 {file_path}")


'''
x轴不同交易量 三张图分别对应不同属性个数
'''
def calc_every_n():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')

    # 固定交易量
    num_transactions = 10000
    # 节点内最大交易量
    order = 1
    # 区块内最大交易量
    n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    # 属性数量
    d_list = [2, 6, 10]

    # 构建时间
    insert_time_rtree = {}
    insert_time_rtree_mbr = {}
    insert_time_merkle_tree = {}
    # 存在搜索
    search_time_rtree = {}
    search_time_rtree_mbr = {}
    search_time_merkle_tree = {}
    # 不存在搜索
    search_no_time_rtree = {}
    search_no_time_rtree_mbr = {}
    search_no_time_merkle_tree = {}
    # 存储
    storage_size_rtree = {}
    storage_size_rtree_mbr = {}
    storage_size_merkle_tree = {}

    for d in d_list:
        # 构建时间
        insert_time_results_rtree = []
        insert_time_results_rtree_mbr = []
        insert_time_results_merkle_tree = []
        # 存在搜索
        search_time_results_rtree = []
        search_time_results_rtree_mbr = []
        search_time_results_merkle_tree = []
        # 不存在搜索
        search_no_time_results_rtree = []
        search_no_time_results_rtree_mbr = []
        search_no_time_results_merkle_tree = []
        # 存储
        storage_size_results_rtree = []
        storage_size_results_rtree_mbr = []
        storage_size_results_merkle_tree = []

        for n in n_list:
            print(f"当前属性数量：{d}，当前区块内交易数量：{n}")

            rtree_db_name = 'blockchain_rtree_{}_{}'.format(d, n)
            rtree_mbr_db_name = 'blockchain_rtree_mbr_{}_{}'.format(d, n)
            mt_db_name = 'blockchain_mt_{}_{}'.format(d, n)

            # 函数用于检查并删除数据库，如果存在的话
            def delete_and_create_db(db_name):
                if db_name in server:
                    server.delete(db_name)
                return server.create(db_name)

            # 删除并创建新数据库
            rtree_db = delete_and_create_db(rtree_db_name)
            rtree_mbr_db = delete_and_create_db(rtree_mbr_db_name)
            mt_db = delete_and_create_db(mt_db_name)
            # 创建区块链对象
            blockchain_r = Blockchain(rtree_db)
            blockchain_mbr = Blockchain(rtree_mbr_db)
            blockchain_mt = Blockchain(mt_db)

            transactions = get_dataset(num_transactions, d)

            '''
            RTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                rtree = RTree(order)
                for tx in tx_batch:
                    rtree.insert(tx)

                # 计算 R 树的根哈希
                merkle_root_rt = rtree.calculate_merkle_root()

                # 创建新的区块
                new_block = Block(merkle_root_rt, rtree, tx_batch, time.time(), "previous_hash_here", n)
                blockchain_r.add_block(new_block)

            # 计算 R 树构建时间
            insert_time_with_rtree = time.time() - start_time

            # 计算存储大小
            total_rtree_storage_size = 0
            for block in blockchain_r.chain:
                serialized_block = pickle.dumps(block)
                total_rtree_storage_size += len(serialized_block)
            serialized_trans = pickle.dumps(transactions)
            # 减去多的一份交易
            total_rtree_storage_size -= len(serialized_trans)

            '''
            mbr_RTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                rtree_mbr = RTree(order)
                for tx in tx_batch:
                    rtree_mbr.insert(tx)

                # 计算 R 树的根哈希
                merkle_root_rt_mbr = rtree_mbr.calculate_merkle_root()

                # 创建新的区块
                new_block = Block(merkle_root_rt_mbr, rtree_mbr, tx_batch, time.time(), "previous_hash_here", n, rtree_mbr.root.bounds)
                blockchain_mbr.add_block(new_block)

            # 计算 R 树构建时间
            insert_time_with_rtree_mbr = time.time() - start_time

            # 计算存储大小
            total_rtree_storage_size_mbr = 0
            for block in blockchain_mbr.chain:
                serialized_block_mbr = pickle.dumps(block)
                total_rtree_storage_size_mbr += len(serialized_block_mbr)
            serialized_trans_mbr = pickle.dumps(transactions)
            # 减去多的一份交易
            total_rtree_storage_size_mbr -= len(serialized_trans_mbr)

            '''
            MerkleTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                merkle_tree = MerkleTree(order)
                for tx in tx_batch:
                    merkle_tree.insert(Transaction(tx.tx_hash, tx.attribute))

                # 计算 Merkle Tree 的根哈希
                merkle_root_mt = merkle_tree.get_root_hash()

                # 创建新的区块
                new_block = MerkleTreeBlock(merkle_root_mt, merkle_tree, tx_batch, time.time(), "previous_hash_here", n)
                # new_block.extra_data = [tx.to_dict() for tx in tx_batch]  # 在 extra_data 字段中保存交易列表
                blockchain_mt.add_block(new_block)

                # 计算 Merkle Tree 构建时间
            insert_time_with_merkle_tree = time.time() - start_time

            # 计算存储大小
            total_merkle_tree_storage_size = 0
            for block in blockchain_mt.chain:
                serialized_block_mt = pickle.dumps(block)
                total_merkle_tree_storage_size += len(serialized_block_mt)
            # print('开始', total_merkle_tree_storage_size)
            # # 减去多余的边界
            # firstBlock = blockchain_mt.getBlock(0)
            # firstTransactionBatch = firstBlock.getTransaction()
            # firstTransactionBounds = firstTransactionBatch[0].bounds
            # serialized_bounds_mt = pickle.dumps(firstTransactionBounds)
            # # 减去交易中多的边界
            # total_merkle_tree_storage_size -= len(serialized_bounds_mt) * num_transactions
            # print('最后',total_merkle_tree_storage_size)

            """-----------对存在的查询条件---------"""
            attributes_to_search = random.sample([tx for tx in transactions], min(50, num_transactions))
            ''' RTree 搜索 '''
            start_time = time.time()
            for tx in attributes_to_search:
                for block in blockchain_r.chain:
                    block.tree.search(tx.bounds)
            search_time_with_rtree = time.time() - start_time
            ''' mbr_RTree 搜索 '''
            start_time = time.time()
            for tx_mbr in attributes_to_search:
                for block in blockchain_mbr.chain:
                    if intersects(tx_mbr.bounds, block.extra_data):
                        block.tree.search(tx_mbr.bounds)
            search_time_with_rtree_mbr = time.time() - start_time
            ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
            start_time = time.time()
            for attr in attributes_to_search:
                for block in blockchain_mt.chain:
                    block.tree.search(transactions, attr, d)
            search_time_with_list = time.time() - start_time

            """-----------对不存在的查询条件---------"""
            # 随机生成假数据
            no_attributes_to_search = get_random_nonexistent_data(transactions, min(50, num_transactions), d)
            ''' RTree 搜索 '''
            start_time = time.time()
            for tx in no_attributes_to_search:
                for block in blockchain_r.chain:
                    block.tree.search(tx.bounds)
            search_no_time_with_rtree = time.time() - start_time
            ''' mbr_RTree 搜索 '''
            start_time = time.time()
            for tx_mbr in no_attributes_to_search:
                for block in blockchain_mbr.chain:
                    if intersects(tx_mbr.bounds, block.extra_data):
                        block.tree.search(tx_mbr.bounds)
            search_no_time_with_rtree_mbr = time.time() - start_time
            ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
            start_time = time.time()
            for attr in no_attributes_to_search:
                for block in blockchain_mt.chain:
                    block.tree.search(transactions, attr, d)
            search_no_time_with_list = time.time() - start_time

            # 保存和打印结果
            insert_time_results_rtree.append(insert_time_with_rtree)
            insert_time_results_rtree_mbr.append(insert_time_with_rtree_mbr)
            insert_time_results_merkle_tree.append(insert_time_with_merkle_tree)
            storage_size_results_rtree.append(total_rtree_storage_size)
            storage_size_results_rtree_mbr.append(total_rtree_storage_size_mbr)
            storage_size_results_merkle_tree.append(total_merkle_tree_storage_size)
            search_time_results_rtree.append(search_time_with_rtree)
            search_time_results_rtree_mbr.append(search_time_with_rtree_mbr)
            search_time_results_merkle_tree.append(search_time_with_list)
            search_no_time_results_rtree.append(search_no_time_with_rtree)
            search_no_time_results_rtree_mbr.append(search_no_time_with_rtree_mbr)
            search_no_time_results_merkle_tree.append(search_no_time_with_list)

        # 构建时间
        insert_time_rtree[d] = insert_time_results_rtree
        insert_time_rtree_mbr[d] = insert_time_results_rtree_mbr
        insert_time_merkle_tree[d] = insert_time_results_merkle_tree
        # 存在搜索
        search_time_rtree[d] = search_time_results_rtree
        search_time_rtree_mbr[d] = search_time_results_rtree_mbr
        search_time_merkle_tree[d] = search_time_results_merkle_tree
        # 不存在搜索
        search_no_time_rtree[d] = search_no_time_results_rtree
        search_no_time_rtree_mbr[d] = search_no_time_results_rtree_mbr
        search_no_time_merkle_tree[d] = search_no_time_results_merkle_tree
        # 存储
        storage_size_rtree[d] = storage_size_results_rtree
        storage_size_rtree_mbr[d] = storage_size_results_rtree_mbr
        storage_size_merkle_tree[d] = storage_size_results_merkle_tree

    d_list_str = [str(d) for d in d_list]

    # 在实验结束后调用函数保存结果
    save_experiment_results(
        d_list_str,
        n_list,
        insert_time_rtree,
        insert_time_rtree_mbr,
        insert_time_merkle_tree,
        storage_size_rtree,
        storage_size_rtree_mbr,
        storage_size_merkle_tree,
        search_time_rtree,
        search_time_rtree_mbr,
        search_time_merkle_tree,
        search_no_time_rtree,
        search_no_time_rtree_mbr,
        search_no_time_merkle_tree,
        'output/every_n_results.json'
    )

    return d_list, \
        n_list, \
        insert_time_rtree, \
        insert_time_rtree_mbr, \
        insert_time_merkle_tree, \
        storage_size_rtree, \
        storage_size_rtree_mbr, \
        storage_size_merkle_tree, \
        search_time_rtree, \
        search_time_rtree_mbr, \
        search_time_merkle_tree, \
        search_no_time_rtree, \
        search_no_time_rtree_mbr, \
        search_no_time_merkle_tree


'''
x轴不同属性个数 三张图分别对应不同交易个数
'''
def calc_every_d():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')

    # 固定交易量
    num_transactions = 10000
    # 节点内最大交易量
    order = 1
    # 区块内最大交易量
    n_list = [5, 25, 45]
    # 属性数量
    d_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 构建时间
    insert_time_rtree = {}
    insert_time_rtree_mbr = {}
    insert_time_merkle_tree = {}
    # 存在搜索
    search_time_rtree = {}
    search_time_rtree_mbr = {}
    search_time_merkle_tree = {}
    # 不存在搜索
    search_no_time_rtree = {}
    search_no_time_rtree_mbr = {}
    search_no_time_merkle_tree = {}
    # 存储
    storage_size_rtree = {}
    storage_size_rtree_mbr = {}
    storage_size_merkle_tree = {}

    for n in n_list:
        # 构建时间
        insert_time_results_rtree = []
        insert_time_results_rtree_mbr = []
        insert_time_results_merkle_tree = []
        # 存在搜索
        search_time_results_rtree = []
        search_time_results_rtree_mbr = []
        search_time_results_merkle_tree = []
        # 不存在搜索
        search_no_time_results_rtree = []
        search_no_time_results_rtree_mbr = []
        search_no_time_results_merkle_tree = []
        # 存储
        storage_size_results_rtree = []
        storage_size_results_rtree_mbr = []
        storage_size_results_merkle_tree = []

        for d in d_list:
            print(f"当前区块内交易数量：{n}，当前属性数量：{d}")

            rtree_db_name = 'blockchain_rtree_{}_{}'.format(n, d)
            rtree_mbr_db_name = 'blockchain_rtree_mbr_{}_{}'.format(n, d)
            mt_db_name = 'blockchain_mt_{}_{}'.format(n, d)

            # 函数用于检查并删除数据库，如果存在的话
            def delete_and_create_db(db_name):
                if db_name in server:
                    server.delete(db_name)
                return server.create(db_name)

            # 删除并创建新数据库
            rtree_db = delete_and_create_db(rtree_db_name)
            rtree_mbr_db = delete_and_create_db(rtree_mbr_db_name)
            mt_db = delete_and_create_db(mt_db_name)
            # 创建区块链对象
            blockchain_r = Blockchain(rtree_db)
            blockchain_mbr = Blockchain(rtree_mbr_db)
            blockchain_mt = Blockchain(mt_db)

            transactions = get_dataset(num_transactions, d)

            '''
            RTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                rtree = RTree(order)
                for tx in tx_batch:
                    rtree.insert(tx)

                # 计算 R 树的根哈希
                merkle_root_rt = rtree.calculate_merkle_root()

                # 创建新的区块
                new_block = Block(merkle_root_rt, rtree, tx_batch, time.time(), "previous_hash_here", n)
                blockchain_r.add_block(new_block)

            # 计算 R 树构建时间
            insert_time_with_rtree = time.time() - start_time

            # 计算存储大小
            total_rtree_storage_size = 0
            for block in blockchain_r.chain:
                serialized_block = pickle.dumps(block)
                total_rtree_storage_size += len(serialized_block)
            serialized_trans = pickle.dumps(transactions)
            # 减去多的一份交易
            total_rtree_storage_size -= len(serialized_trans)

            '''
            mbr_RTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                rtree_mbr = RTree(order)
                for tx in tx_batch:
                    rtree_mbr.insert(tx)

                # 计算 R 树的根哈希
                merkle_root_rt_mbr = rtree_mbr.calculate_merkle_root()

                # 创建新的区块
                new_block = Block(merkle_root_rt_mbr, rtree_mbr, tx_batch, time.time(), "previous_hash_here", n,
                                  rtree_mbr.root.bounds)
                blockchain_mbr.add_block(new_block)

            # 计算 R 树构建时间
            insert_time_with_rtree_mbr = time.time() - start_time

            # 计算存储大小
            total_rtree_storage_size_mbr = 0
            for block in blockchain_mbr.chain:
                serialized_block_mbr = pickle.dumps(block)
                total_rtree_storage_size_mbr += len(serialized_block_mbr)
            serialized_trans_mbr = pickle.dumps(transactions)
            # 减去多的一份交易
            total_rtree_storage_size_mbr -= len(serialized_trans_mbr)

            '''
            MerkleTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                merkle_tree = MerkleTree(order)
                for tx in tx_batch:
                    merkle_tree.insert(Transaction(tx.tx_hash, tx.attribute))

                # 计算 Merkle Tree 的根哈希
                merkle_root_mt = merkle_tree.get_root_hash()

                # 创建新的区块
                new_block = MerkleTreeBlock(merkle_root_mt, merkle_tree, tx_batch, time.time(), "previous_hash_here", n)
                blockchain_mt.add_block(new_block)

                # 计算 Merkle Tree 构建时间
            insert_time_with_merkle_tree = time.time() - start_time

            # 计算存储大小
            total_merkle_tree_storage_size = 0
            for block in blockchain_mt.chain:
                serialized_block_mt = pickle.dumps(block)
                total_merkle_tree_storage_size += len(serialized_block_mt)
            # # 减去多余的边界
            # firstBlock = blockchain_mt.getBlock(0)
            # firstTransactionBatch = firstBlock.getTransaction()
            # firstTransactionBounds = firstTransactionBatch[0].bounds
            # serialized_bounds_mt = pickle.dumps(firstTransactionBounds)
            # # 减去交易中多的边界
            # total_merkle_tree_storage_size -= len(serialized_bounds_mt) * num_transactions

            """-----------对存在的查询条件---------"""
            attributes_to_search = random.sample([tx for tx in transactions], min(50, num_transactions))
            ''' RTree 搜索 '''
            start_time = time.time()
            for tx in attributes_to_search:
                for block in blockchain_r.chain:
                    block.tree.search(tx.bounds)
            search_time_with_rtree = time.time() - start_time
            ''' mbr_RTree 搜索 '''
            start_time = time.time()
            for tx_mbr in attributes_to_search:
                for block in blockchain_mbr.chain:
                    if intersects(tx_mbr.bounds, block.extra_data):
                        block.tree.search(tx_mbr.bounds)
            search_time_with_rtree_mbr = time.time() - start_time
            ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
            start_time = time.time()
            for attr in attributes_to_search:
                for block in blockchain_mt.chain:
                    block.tree.search(transactions, attr, d)
            search_time_with_list = time.time() - start_time

            """-----------对不存在的查询条件---------"""
            # 随机生成假数据
            no_attributes_to_search = get_random_nonexistent_data(transactions, min(50, num_transactions), d)
            ''' RTree 搜索 '''
            start_time = time.time()
            for tx in no_attributes_to_search:
                for block in blockchain_r.chain:
                    block.tree.search(tx.bounds)
            search_no_time_with_rtree = time.time() - start_time
            ''' mbr_RTree 搜索 '''
            start_time = time.time()
            for tx_mbr in no_attributes_to_search:
                for block in blockchain_mbr.chain:
                    if intersects(tx_mbr.bounds, block.extra_data):
                        block.tree.search(tx_mbr.bounds)
            search_no_time_with_rtree_mbr = time.time() - start_time
            ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
            start_time = time.time()
            for attr in no_attributes_to_search:
                for block in blockchain_mt.chain:
                    block.tree.search(transactions, attr, d)
            search_no_time_with_list = time.time() - start_time

            # 保存和打印结果
            insert_time_results_rtree.append(insert_time_with_rtree)
            insert_time_results_rtree_mbr.append(insert_time_with_rtree_mbr)
            insert_time_results_merkle_tree.append(insert_time_with_merkle_tree)
            storage_size_results_rtree.append(total_rtree_storage_size)
            storage_size_results_rtree_mbr.append(total_rtree_storage_size_mbr)
            storage_size_results_merkle_tree.append(total_merkle_tree_storage_size)
            search_time_results_rtree.append(search_time_with_rtree)
            search_time_results_rtree_mbr.append(search_time_with_rtree_mbr)
            search_time_results_merkle_tree.append(search_time_with_list)
            search_no_time_results_rtree.append(search_no_time_with_rtree)
            search_no_time_results_rtree_mbr.append(search_no_time_with_rtree_mbr)
            search_no_time_results_merkle_tree.append(search_no_time_with_list)

        # 构建时间
        insert_time_rtree[n] = insert_time_results_rtree
        insert_time_rtree_mbr[n] = insert_time_results_rtree_mbr
        insert_time_merkle_tree[n] = insert_time_results_merkle_tree
        # 存在搜索
        search_time_rtree[n] = search_time_results_rtree
        search_time_rtree_mbr[n] = search_time_results_rtree_mbr
        search_time_merkle_tree[n] = search_time_results_merkle_tree
        # 不存在搜索
        search_no_time_rtree[n] = search_no_time_results_rtree
        search_no_time_rtree_mbr[n] = search_no_time_results_rtree_mbr
        search_no_time_merkle_tree[n] = search_no_time_results_merkle_tree
        # 存储
        storage_size_rtree[n] = storage_size_results_rtree
        storage_size_rtree_mbr[n] = storage_size_results_rtree_mbr
        storage_size_merkle_tree[n] = storage_size_results_merkle_tree

    n_list_str = [str(n) for n in n_list]
    # 在实验结束后调用函数保存结果
    save_experiment_results(
        d_list,
        n_list_str,
        insert_time_rtree,
        insert_time_rtree_mbr,
        insert_time_merkle_tree,
        storage_size_rtree,
        storage_size_rtree_mbr,
        storage_size_merkle_tree,
        search_time_rtree,
        search_time_rtree_mbr,
        search_time_merkle_tree,
        search_no_time_rtree,
        search_no_time_rtree_mbr,
        search_no_time_merkle_tree,
        'output/every_d_results.json'
    )

    return d_list, \
           n_list, \
           insert_time_rtree, \
           insert_time_rtree_mbr, \
           insert_time_merkle_tree, \
           storage_size_rtree, \
           storage_size_rtree_mbr, \
           storage_size_merkle_tree, \
           search_time_rtree, \
           search_time_rtree_mbr, \
           search_time_merkle_tree, \
           search_no_time_rtree, \
           search_no_time_rtree_mbr, \
           search_no_time_merkle_tree


'''
x轴不同总交易量 三张图分别对应不同查询个数
'''
def calc_every_t():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')
    # 数据库名称
    rtree_db_name = 'blockchain_rtree'
    rtree_mbr_db_name = 'blockchain_rtree_mbr'
    mt_db_name = 'blockchain_mt'

    # 函数用于检查并删除数据库，如果存在的话
    def delete_and_create_db(db_name):
        if db_name in server:
            server.delete(db_name)
        return server.create(db_name)

    # 删除并创建新数据库
    rtree_db = delete_and_create_db(rtree_db_name)
    rtree_mbr_db = delete_and_create_db(rtree_mbr_db_name)
    mt_db = delete_and_create_db(mt_db_name)

    # 创建区块链对象
    blockchain = Blockchain(rtree_db)
    blockchain_mbr = Blockchain(rtree_mbr_db)
    blockchain_mt = Blockchain(mt_db)

    # 不同交易量
    num_transactions_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # 节点内最大交易量
    order = 1
    # 区块内最大交易量 5 10 15
    n = 10
    # 属性数量 2 4 8
    d = 2

    # 构建时间
    insert_time_results_rtree = []
    insert_time_results_rtree_mbr = []
    insert_time_results_merkle_tree = []
    # 存在搜索
    search_time_results_rtree = []
    search_time_results_rtree_mbr = []
    search_time_results_merkle_tree = []
    # 不存在搜索
    search_no_time_results_rtree = []
    search_no_time_results_rtree_mbr = []
    search_no_time_results_merkle_tree = []
    # 存储
    storage_size_results_rtree = []
    storage_size_results_rtree_mbr = []
    storage_size_results_merkle_tree = []

    """-----------不同交易量-------固定查询条件个数---------"""
    for num_transactions in num_transactions_list:
        print(f"当前交易数量：{num_transactions}，当前属性数量：{d}")
        transactions = get_dataset(num_transactions, d)

        '''
        RTree 构建 存储
        '''
        start_time = time.time()
        for i in range(0, num_transactions, n):
            tx_batch = transactions[i:i + n]
            rtree = RTree(order)
            for tx in tx_batch:
                rtree.insert(tx)

            # 计算 R 树的根哈希
            merkle_root_rt = rtree.calculate_merkle_root()

            # 创建新的区块
            new_block = Block(merkle_root_rt, rtree, tx_batch, time.time(), "previous_hash_here", n)
            blockchain.add_block(new_block)

        # 计算 R 树构建时间
        insert_time_with_rtree = time.time() - start_time

        # 计算存储大小
        total_rtree_storage_size = 0
        for block in blockchain.chain:
            serialized_block = pickle.dumps(block)
            total_rtree_storage_size += len(serialized_block)
        serialized_trans = pickle.dumps(transactions)
        # 减去多的一份交易
        total_rtree_storage_size -= len(serialized_trans)

        '''
        mbr_RTree 构建 存储
        '''
        start_time = time.time()
        for i in range(0, num_transactions, n):
            tx_batch = transactions[i:i + n]
            rtree_mbr = RTree(order)
            for tx in tx_batch:
                rtree_mbr.insert(tx)

            # 计算 R 树的根哈希
            merkle_root_rt_mbr = rtree_mbr.calculate_merkle_root()

            # 创建新的区块
            new_block = Block(merkle_root_rt_mbr, rtree_mbr, tx_batch, time.time(), "previous_hash_here", n,
                              rtree_mbr.root.bounds)
            blockchain_mbr.add_block(new_block)

        # 计算 R 树构建时间
        insert_time_with_rtree_mbr = time.time() - start_time

        # 计算存储大小
        total_rtree_storage_size_mbr = 0
        for block in blockchain_mbr.chain:
            serialized_block_mbr = pickle.dumps(block)
            total_rtree_storage_size_mbr += len(serialized_block_mbr)
        serialized_trans_mbr = pickle.dumps(transactions)
        # 减去多的一份交易
        total_rtree_storage_size_mbr -= len(serialized_trans_mbr)

        '''
        MerkleTree 构建 存储
        '''
        start_time = time.time()
        for i in range(0, num_transactions, n):
            tx_batch = transactions[i:i + n]
            merkle_tree = MerkleTree(order)
            for tx in tx_batch:
                merkle_tree.insert(Transaction(tx.tx_hash, tx.attribute))

            # 计算 Merkle Tree 的根哈希
            merkle_root_mt = merkle_tree.get_root_hash()

            # 创建新的区块
            new_block = MerkleTreeBlock(merkle_root_mt, merkle_tree, tx_batch, time.time(), "previous_hash_here", n)
            # new_block.extra_data = [tx.to_dict() for tx in tx_batch]  # 在 extra_data 字段中保存交易列表
            blockchain_mt.add_block(new_block)

            # 计算 Merkle Tree 构建时间
        insert_time_with_merkle_tree = time.time() - start_time

        # 计算存储大小
        total_merkle_tree_storage_size = 0
        for block in blockchain_mt.chain:
            serialized_block_mt = pickle.dumps(block)
            total_merkle_tree_storage_size += len(serialized_block_mt)
        # 减去多余的边界
        firstBlock = blockchain_mt.getBlock(0)
        firstTransactionBatch = firstBlock.getTransaction()
        firstTransactionBounds = firstTransactionBatch[0].bounds
        serialized_bounds_mt = pickle.dumps(firstTransactionBounds)
        # 减去交易中多的边界
        total_merkle_tree_storage_size -= len(serialized_bounds_mt) * num_transactions

        """-----------对存在的查询条件---------"""
        attributes_to_search = random.sample([tx for tx in transactions], min(50, num_transactions))
        ''' RTree 搜索 '''
        start_time = time.time()
        for tx in attributes_to_search:
            for block in blockchain.chain:
                block.tree.search(tx.bounds)
        search_time_with_rtree = time.time() - start_time
        ''' mbr_RTree 搜索 '''
        start_time = time.time()
        for tx_mbr in attributes_to_search:
            for block in blockchain_mbr.chain:
                if intersects(tx_mbr.bounds, block.extra_data):
                    block.tree.search(tx_mbr.bounds)
        search_time_with_rtree_mbr = time.time() - start_time
        ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
        start_time = time.time()
        for attr in attributes_to_search:
            for block in blockchain_mt.chain:
                block.tree.search(transactions, attr, d)
        search_time_with_list = time.time() - start_time

        """-----------对不存在的查询条件---------"""
        # 随机生成假数据
        no_attributes_to_search = get_random_nonexistent_data(transactions, min(50, num_transactions), d)
        ''' RTree 搜索 '''
        start_time = time.time()
        for tx in no_attributes_to_search:
            for block in blockchain.chain:
                block.tree.search(tx.bounds)
        search_no_time_with_rtree = time.time() - start_time
        ''' mbr_RTree 搜索 '''
        start_time = time.time()
        for tx_mbr in no_attributes_to_search:
            for block in blockchain_mbr.chain:
                if intersects(tx_mbr.bounds, block.extra_data):
                    block.tree.search(tx_mbr.bounds)
        search_no_time_with_rtree_mbr = time.time() - start_time
        ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
        start_time = time.time()
        for attr in no_attributes_to_search:
            for block in blockchain_mt.chain:
                block.tree.search(transactions, attr, d)
        search_no_time_with_list = time.time() - start_time

        # 保存和打印结果
        insert_time_results_rtree.append(insert_time_with_rtree)
        insert_time_results_rtree_mbr.append(insert_time_with_rtree_mbr)
        insert_time_results_merkle_tree.append(insert_time_with_merkle_tree)
        storage_size_results_rtree.append(total_rtree_storage_size)
        storage_size_results_rtree_mbr.append(total_rtree_storage_size_mbr)
        storage_size_results_merkle_tree.append(total_merkle_tree_storage_size)
        search_time_results_rtree.append(search_time_with_rtree)
        search_time_results_rtree_mbr.append(search_time_with_rtree_mbr)
        search_time_results_merkle_tree.append(search_time_with_list)
        search_no_time_results_rtree.append(search_no_time_with_rtree)
        search_no_time_results_rtree_mbr.append(search_no_time_with_rtree_mbr)
        search_no_time_results_merkle_tree.append(search_no_time_with_list)

    return num_transactions_list, \
           insert_time_results_rtree, \
           insert_time_results_rtree_mbr, \
           insert_time_results_merkle_tree, \
           storage_size_results_rtree, \
           storage_size_results_rtree_mbr, \
           storage_size_results_merkle_tree, \
           search_time_results_rtree, \
           search_time_results_rtree_mbr, \
           search_time_results_merkle_tree, \
           search_no_time_results_rtree, \
           search_no_time_results_rtree_mbr, \
           search_no_time_results_merkle_tree


'''
针对两种树结构本身的 构建 查询 的测试
'''
def calc_tree_self():
    num_transactions = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # 交易量设置
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
        merkle_tree = MerkleTree(4)
        r_tree = RTree(4)
        for tx in transactions:
            merkle_tree.insert(tx)
            r_tree.insert(tx)

        # 测量查找时间
        # merkle_transactions = merkle_tree.transactions
        search_time_list = measure_search_time_mt(merkle_tree, transactions, tx_to_search, 2)
        search_time_r_tree = measure_search_time_r_tree(r_tree, tx_to_search)


        # 记录结果
        insert_time_results_r_tree.append(insert_time_r_tree)
        insert_time_results_mt.append(insert_time_mt)
        search_time_results_r_tree.append(search_time_r_tree/num_search_queries)
        search_time_results_mt.append(search_time_list/num_search_queries)

        print("R树构建",insert_time_results_r_tree)
        print("M树构建",insert_time_results_mt)
        print("R树查询",search_time_results_r_tree)
        print("M树查询",search_time_results_mt)

    return num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt


'''
针对几种溯源方案的 构建 溯源 的测试
'''
def calc_trace_every_n():
    # 先用R树查询到最新的节点和相应的原始数据
    # 然后根据原始数据中的区块号和id定位到前一条数据
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')

    # 固定交易量 随机选择
    num_transactions = 1000
    # 每一条对应生成10条历史数据
    num_history = 10
    # 节点内最大交易量
    order = 1
    # 区块内最大交易量
    n_list = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    # 属性数量
    d_list = [2, 6, 10]

    # 构建时间
    insert_time_rtree = {}
    insert_time_rtree_mbr = {}
    insert_time_fabric = {}
    insert_time_fabric_sort = {}
    # 存在搜索
    search_time_rtree = {}
    search_time_rtree_mbr = {}
    search_time_fabric = {}
    search_time_fabric_sort = {}
    # 存储
    storage_size_rtree = {}
    storage_size_rtree_mbr = {}
    storage_size_fabric = {}
    storage_size_fabric_sort = {}

    for d in d_list:
        # 构建时间
        insert_time_results_rtree = []
        insert_time_results_rtree_mbr = []
        insert_time_results_fabric = []
        insert_time_results_fabric_sort = []
        # 存在搜索
        search_time_results_rtree = []
        search_time_results_rtree_mbr = []
        search_time_results_fabric = []
        search_time_results_fabric_sort = []
        # 存储
        storage_size_results_rtree = []
        storage_size_results_rtree_mbr = []
        storage_size_results_fabric = []
        storage_size_results_fabric_sort = []

        for n in n_list:
            print(f"当前属性数量：{d}，当前区块内交易数量：{n}")

            rtree_db_name = 'blockchain_history_rtree_{}_{}'.format(d, n)
            rtree_mbr_db_name = 'blockchain_history_rtree_mbr_{}_{}'.format(d, n)
            fabric_db_name = 'blockchain_history_mt_{}_{}'.format(d, n)
            fabric_sort_db_name = 'blockchain_history_mt_{}_{}'.format(d, n)

            # 函数用于检查并删除数据库，如果存在的话
            def delete_and_create_db(db_name):
                if db_name in server:
                    server.delete(db_name)
                return server.create(db_name)

            # 删除并创建新数据库
            rtree_db = delete_and_create_db(rtree_db_name)
            rtree_mbr_db = delete_and_create_db(rtree_mbr_db_name)
            fabric_db = delete_and_create_db(fabric_db_name)
            fabric_sort_db = delete_and_create_db(fabric_sort_db_name)
            # 创建区块链对象
            blockchain_r = Blockchain(rtree_db)
            blockchain_mbr = Blockchain(rtree_mbr_db)
            blockchain_fabric = Blockchain(fabric_db)
            blockchain_fabric_sort = Blockchain(fabric_sort_db)

            # 获取数据
            transactions = generate_history_data(num_transactions, d, num_history)

            # 历史数据构建索引状态表
            mapping_r = {}
            mapping_mbr = {}

            '''
            RTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                rtree = RTree(order)
                current_block_index = blockchain_r.length()  # 当前区块的索引

                for tx in tx_batch:
                    tx_id = tx.attribute[-1]  # 假设 tx_id 是交易的唯一标识符
                    if tx_id in mapping_r:
                        # 如果交易的历史版本已经存在，则将历史区块索引添加到当前交易
                        history_block_index = mapping_r[tx_id]
                        tx.pre_index = history_block_index
                    else:
                        # 如果交易是第一次出现，添加 -1 作为占位符
                        tx.pre_index = -1

                    # 更新 mapping_r，将当前交易的 ID 映射到当前区块索引
                    mapping_r[tx_id] = current_block_index
                    rtree.insert(tx)

                # 计算 R 树的根哈希
                merkle_root_rt = rtree.calculate_merkle_root()

                # 创建新的区块
                new_block = Block(merkle_root_rt, rtree, tx_batch, time.time(), "previous_hash_here", n)
                blockchain_r.add_block(new_block)

            # 计算 R 树构建时间
            insert_time_with_rtree = time.time() - start_time

            # 计算存储大小
            total_rtree_storage_size = 0
            for block in blockchain_r.chain:
                serialized_block = pickle.dumps(block)
                total_rtree_storage_size += len(serialized_block)
            # TODO
            # serialized_trans = pickle.dumps(transactions)
            # # 减去多的一份交易
            # total_rtree_storage_size -= len(serialized_trans)

            '''
            mbr_RTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                rtree_mbr = RTree(order)
                current_block_index = blockchain_mbr.length()  # 当前区块的索引

                for tx in tx_batch:
                    tx_id = tx.attribute[-1]  # 假设 tx_id 是交易的唯一标识符
                    if tx_id in mapping_mbr:
                        # 如果交易的历史版本已经存在，则将历史区块索引添加到当前交易
                        history_block_index = mapping_mbr[tx_id]
                        tx.pre_index = history_block_index
                    else:
                        # 如果交易是第一次出现，添加 -1 作为占位符
                        tx.pre_index = -1

                    # 更新 mapping_mbr，将当前交易的 ID 映射到当前区块索引
                    mapping_mbr[tx_id] = current_block_index
                    rtree_mbr.insert(tx)

                # 计算 R 树的根哈希
                merkle_root_rt_mbr = rtree_mbr.calculate_merkle_root()
                # 创建新的区块，并将 MBR（最小边界矩形）存储在区块中
                new_block = Block(merkle_root_rt_mbr, rtree_mbr, tx_batch, time.time(), "previous_hash_here", n,
                                  rtree_mbr.root.bounds)
                blockchain_mbr.add_block(new_block)

            # 计算 R 树构建时间
            insert_time_with_rtree_mbr = time.time() - start_time

            # 计算存储大小
            total_rtree_storage_size_mbr = 0
            for block in blockchain_mbr.chain:
                serialized_block_mbr = pickle.dumps(block)
                total_rtree_storage_size_mbr += len(serialized_block_mbr)
            # TODO
            # serialized_trans_mbr = pickle.dumps(transactions)
            # # 减去多的一份交易
            # total_rtree_storage_size_mbr -= len(serialized_trans_mbr)

            '''
            fabric----MerkleTree 构建 存储
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                merkle_tree = MerkleTree(order)
                for tx in tx_batch:
                    merkle_tree.insert(Transaction(tx.tx_hash, tx.attribute))

                # 计算 Merkle Tree 的根哈希
                merkle_root_mt = merkle_tree.get_root_hash()

                # 创建新的区块
                new_block = MerkleTreeBlock(merkle_root_mt, merkle_tree, tx_batch, time.time(), "previous_hash_here", n)
                blockchain_fabric.add_block(new_block)

                # 计算 Merkle Tree 构建时间
            insert_time_with_fabric = time.time() - start_time

            # 计算存储大小
            total_fabric_storage_size = 0
            for block in blockchain_fabric.chain:
                serialized_block_mt = pickle.dumps(block)
                total_fabric_storage_size += len(serialized_block_mt)
            # # 减去多余的边界
            # firstBlock = blockchain_fabric.getBlock(0)
            # firstTransactionBatch = firstBlock.getTransaction()
            # firstTransactionBounds = firstTransactionBatch[0].bounds
            # serialized_bounds_mt = pickle.dumps(firstTransactionBounds)
            # # 减去交易中多的边界
            # total_fabric_storage_size -= len(serialized_bounds_mt) * num_transactions

            '''
            fabric_sort----MerkleTree构建 块体内时间戳排序
            '''
            start_time = time.time()
            for i in range(0, num_transactions, n):
                tx_batch = transactions[i:i + n]
                merkle_sort_tree = MerkleTree(order)

                # 在批次中计算 bounds[0] 的最小值和最大值
                bounds_first_values = [tx.bounds[0] for tx in tx_batch]  # 只针对 tx_batch 进行操作
                min_value = min(bounds_first_values)
                max_value = max(bounds_first_values)
                dateRange = [min_value, max_value]

                for tx in tx_batch:
                    # 插入交易的哈希和属性到 Merkle Tree 中
                    merkle_sort_tree.insert(Transaction(tx.tx_hash, tx.attribute))

                # 计算 Merkle Tree 的根哈希
                merkle_root_mt = merkle_sort_tree.get_root_hash()

                # 创建新的区块
                new_block = MerkleTreeBlock(merkle_root_mt, merkle_sort_tree, tx_batch, time.time(),
                                            "previous_hash_here", n, dateRange)

                # 将新的区块添加到区块链中
                blockchain_fabric_sort.add_block(new_block)

            # 计算 Merkle Tree 构建时间
            insert_time_with_fabric_sort = time.time() - start_time

            # 计算存储大小
            total_fabric_sort_storage_size = 0
            for block in blockchain_fabric_sort.chain:
                serialized_block_fabric_sort = pickle.dumps(block)
                total_fabric_sort_storage_size += len(serialized_block_fabric_sort)
            # # 减去多余的边界
            # firstBlock_fabric_sort = blockchain_fabric_sort.getBlock(0)
            # firstTransactionBatch_fabric_sort = firstBlock_fabric_sort.getTransaction()
            # firstTransactionBounds_fabric_sort = firstTransactionBatch_fabric_sort[0].bounds
            # serialized_bounds_fabric_sort = pickle.dumps(firstTransactionBounds_fabric_sort)
            # # 减去交易中多的边界
            # total_fabric_sort_storage_size -= len(serialized_bounds_fabric_sort) * num_transactions

            """-----------对存在的查询条件 进行查找和溯源验证---------"""
            attributes_to_search = random.sample([tx for tx in transactions], min(50, num_transactions))
            ''' RTree 搜索 '''
            history_r = []
            r_res = []
            start_time = time.time()
            for tx in attributes_to_search:
                for block in blockchain_r.chain:
                    history_r.extend(block.tree.search(tx.bounds))
            for r in history_r:
                res = []
                res.append(r)
                r_res.append(get_history_by_id(r, blockchain_r.chain, res))
            search_time_with_rtree = time.time() - start_time
            # if len(history_r) > 0:
            #     search_time_with_rtree = search_time_with_rtree / len(history_r) * min(50, num_transactions)
            # else:
            #     search_time_with_rtree = search_time_with_rtree * min(50, num_transactions)
            print('RTree溯源结果：', r_res)

            ''' Mbr_RTree 搜索 '''
            history_mbr = []
            mbr_res = []
            start_time = time.time()
            for tx_mbr in attributes_to_search:
                for block in blockchain_mbr.chain:
                    if intersects(tx_mbr.bounds, block.extra_data):
                        history_mbr.extend(block.tree.search(tx_mbr.bounds))
            for r in history_mbr:
                res_tmp = []
                res_tmp.append(r)
                mbr_res.append(get_history_by_id(r, blockchain_mbr.chain, res_tmp))
            search_time_with_rtree_mbr = time.time() - start_time
            # if len(history_mbr) > 0:
            #     search_time_with_rtree_mbr = search_time_with_rtree_mbr / len(history_mbr) * min(50, num_transactions)
            # else:
            #     search_time_with_rtree_mbr = search_time_with_rtree_mbr * min(50, num_transactions)
            print('MRTree溯源结果：', mbr_res)

            ''' Fabric MerkleTree 搜索 查数据库 再查账本 '''
            # TODO：值查询 范围查询 两种  目前是先写一种简单的 值查询 然后去验证
            history_fabric = []
            start_time = time.time()
            for tx_fabric in attributes_to_search:
                # 查询历史数据库并返回匹配的数据
                matching_transactions = query_transactions(fabric_db, tx_fabric.attribute)
                for tran_fabric in matching_transactions:
                    # 遍历区块链 找到对应区块中的数据 哈希验证
                    for block in blockchain_fabric.chain:
                        # FIXME：备用方案
                        # history_fabric.extend(block.tree.search(transactions, tran_fabric, d))
                        # 优先方案 利用哈希值 遍历区块链 并在区块内无序查找
                        history_fabric.extend(block.tree.searchByHash(transactions, tran_fabric))
            search_time_with_fabric = time.time() - start_time
            print('Fabric-MTree溯源结果：', history_fabric)

            ''' 数据库 查询 MerkleTree内排序 验证 溯源 '''
            history_fabric_sort = []
            start_time = time.time()
            for tx_fabric_sort in attributes_to_search:
                # 查询历史数据库并返回匹配的数据
                matching_transactions = query_transactions(fabric_sort_db, tx_fabric_sort.attribute)

                for tran_fabric_sort in matching_transactions:
                    # 遍历区块链 找到对应区块中的数据 哈希验证
                    for block in blockchain_fabric_sort.chain:
                        # 利用时间戳 遍历区块链 快速定位一些区块 然后在区块内无序查找
                        if block.extra_data[0] <= tran_fabric_sort["bounds"][0] <= block.extra_data[1]:
                            # 区块内查找 这里直接使用哈希值判等 TODO：应该继续使用时间戳在块内的默克尔树中使用二分查找 然后比较哈希值
                            for bt in block.transactions:
                                if bt.tx_hash == tran_fabric_sort["tx_hash"]:
                                    # 添加到历史记录
                                    history_fabric_sort.append(bt)
                                    # 跳出当前块的循环，直接进入下一个 matching_transactions 循环
                                    break
                            else:
                                # 如果没有匹配的交易，继续下一次外层 block 循环
                                continue
                            # 找到匹配交易并已经 append 之后，跳出 block 循环
                            break
            search_time_with_fabric_sort = time.time() - start_time
            print('Fabric-sort-MTree溯源结果：', history_fabric_sort)


            # 保存和打印结果
            insert_time_results_rtree.append(insert_time_with_rtree)
            insert_time_results_rtree_mbr.append(insert_time_with_rtree_mbr)
            insert_time_results_fabric.append(insert_time_with_fabric)
            insert_time_results_fabric_sort.append(insert_time_with_fabric_sort)
            storage_size_results_rtree.append(total_rtree_storage_size)
            storage_size_results_rtree_mbr.append(total_rtree_storage_size_mbr)
            storage_size_results_fabric.append(total_fabric_storage_size)
            storage_size_results_fabric_sort.append(total_fabric_sort_storage_size)
            search_time_results_rtree.append(search_time_with_rtree)
            search_time_results_rtree_mbr.append(search_time_with_rtree_mbr)
            search_time_results_fabric.append(search_time_with_fabric)
            search_time_results_fabric_sort.append(search_time_with_fabric_sort)

        # 构建时间
        insert_time_rtree[d] = insert_time_results_rtree
        insert_time_rtree_mbr[d] = insert_time_results_rtree_mbr
        insert_time_fabric[d] = insert_time_results_fabric
        insert_time_fabric_sort[d] = insert_time_results_fabric_sort
        # 存在搜索
        search_time_rtree[d] = search_time_results_rtree
        search_time_rtree_mbr[d] = search_time_results_rtree_mbr
        search_time_fabric[d] = search_time_results_fabric
        search_time_fabric_sort[d] = search_time_results_fabric_sort
        # 存储
        storage_size_rtree[d] = storage_size_results_rtree
        storage_size_rtree_mbr[d] = storage_size_results_rtree_mbr
        storage_size_fabric[d] = storage_size_results_fabric
        storage_size_fabric_sort[d] = storage_size_results_fabric_sort

    d_list_str = [str(d) for d in d_list]

    # 在实验结束后调用函数保存结果
    save_history_results(
        d_list_str,
        n_list,
        insert_time_rtree,
        insert_time_rtree_mbr,
        insert_time_fabric,
        insert_time_fabric_sort,
        storage_size_rtree,
        storage_size_rtree_mbr,
        storage_size_fabric,
        storage_size_fabric_sort,
        search_time_rtree,
        search_time_rtree_mbr,
        search_time_fabric,
        search_time_fabric_sort,
        'output/history_every_n_results.json'
    )

    return d_list, \
           n_list, \
           insert_time_rtree, \
           insert_time_rtree_mbr, \
           insert_time_fabric, \
           insert_time_fabric_sort,\
           storage_size_rtree, \
           storage_size_rtree_mbr, \
           storage_size_fabric, \
           storage_size_fabric_sort,\
           search_time_rtree, \
           search_time_rtree_mbr, \
           search_time_fabric, \
           search_time_fabric_sort