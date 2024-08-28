import time
import json
import random
import pickle
import couchdb

# 导入自定义包
from blockchain import Transaction, Blockchain, Block, RTree, MerkleTree, MerkleTreeBlock
from data import get_dataset, get_random_nonexistent_data, measure_insert_time_r_tree, measure_insert_time_mt, measure_search_time_r_tree, measure_search_time_mt


def intersects(mbr1, mbr2):
    return all(mbr1[i] <= mbr2[i + len(mbr1) // 2] and mbr2[i] <= mbr1[i + len(mbr1) // 2] for i in range(len(mbr1) // 2))

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
    file_path="experiment_results.json"
):
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

    # 在实验结束后调用函数保存结果
    save_experiment_results(
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

    # 在实验结束后调用函数保存结果
    save_experiment_results(
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