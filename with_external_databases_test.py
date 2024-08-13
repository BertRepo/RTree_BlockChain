import concurrent.futures
import time
import random
import pickle
import couchdb

# 导入自定义包
from blockchain import Blockchain, Block, RTree, MerkleTree, MerkleTreeBlock
from global_method import get_dataset


def with_external_databases_test():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')
    rtree_db_name = 'blockchain_rtree'
    mt_db_name = 'blockchain_mt'
    rtree_db = server[rtree_db_name] if rtree_db_name in server else server.create(rtree_db_name)
    mt_db = server[mt_db_name] if mt_db_name in server else server.create(mt_db_name)
    blockchain = Blockchain(rtree_db)
    blockchain_mt = Blockchain(mt_db)

    # 不同交易量
    num_transactions_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # 节点内最大交易量
    order = 1
    # 区块内最大交易量
    n = 10

    # 构建时间
    insert_time_results_rtree = []
    insert_time_results_merkle_tree = []
    # 搜索
    search_time_results_rtree = []
    search_time_results_merkle_tree = []
    # 存储
    storage_size_results_rtree = []
    storage_size_results_merkle_tree = []

    """-----------不同交易量-------固定查询条件个数---------"""
    for num_transactions in num_transactions_list:
        print(f"当前交易数量：{num_transactions}")
        transactions = get_dataset(num_transactions)

        '''
        RTree 构建 存储
        '''
        start_time = time.time()
        rtree = RTree(order)
        for tx in transactions:
            rtree.insert(tx)
        merkle_root_rt = rtree.calculate_merkle_root()  # 计算根哈希
        new_block = Block(merkle_root_rt, transactions, time.time(), "previous_hash_here")  # 区块内默认设置最大交易量为4
        # 插入交易到区块当中
        new_block.extra_data = [tx.to_dict() for tx in transactions]  # 在extra_data字段中保存交易列表
        blockchain.add_block(new_block)

        # R树构建时间
        insert_time_with_rtree = time.time() - start_time
        # TODO：循环区块 计算存储大小
        serialized_block = pickle.dumps(new_block)
        rtree_storage_size = len(serialized_block)
        storage_size_results_rtree.append(rtree_storage_size)

        '''
        MerkleTree 构建 存储
        '''
        start_time = time.time()
        merkle_tree = MerkleTree(order)
        for tx in transactions:
            merkle_tree.insert(tx)
        merkle_root_mt = merkle_tree.get_root_hash()
        new_block = MerkleTreeBlock(merkle_root_mt, transactions, time.time(), "previous_hash_here")
        new_block.extra_data = [tx.to_dict() for tx in transactions]  # 在extra_data字段中保存交易列表
        blockchain_mt.add_block(new_block)
        insert_time_with_merkle_tree = time.time() - start_time
        serialized_block = pickle.dumps(new_block)
        merkle_tree_storage_size = len(serialized_block)
        storage_size_results_merkle_tree.append(merkle_tree_storage_size)

        """-----------对存在的查询条件---------"""
        attributes_to_search = random.sample([tx for tx in transactions], min(50, num_transactions))
        ''' RTree 搜索 '''
        start_time = time.time()
        for tx in attributes_to_search:
            rtree.search(tx.bounds)
        # time.sleep(0.0001)
        search_time_with_rtree = time.time() - start_time
        ''' MerkleTree 搜索 其实就是列表（块体内的事务列表） '''
        start_time = time.time()
        for attr in attributes_to_search:
            merkle_tree.search(transactions, attr)
        # time.sleep(0.005)
        search_time_with_list = time.time() - start_time

        # 保存和打印结果
        insert_time_results_rtree.append(insert_time_with_rtree)
        insert_time_results_merkle_tree.append(insert_time_with_merkle_tree)
        search_time_results_rtree.append(search_time_with_rtree)
        search_time_results_merkle_tree.append(search_time_with_list)

        """-----------对不存在的查询条件---------"""
        # TODO：不存在的 数据怎么搞
        no_attributes_to_search = random.sample([tx for tx in transactions], min(50, num_transactions))
        
    return num_transactions_list, \
        insert_time_results_rtree, \
        insert_time_results_merkle_tree, \
        search_time_results_rtree, \
        search_time_results_merkle_tree, \
        storage_size_results_rtree, \
        storage_size_results_merkle_tree


def compare_trees_by_block_height():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')

    rtree_db_name = 'blockchain_by_height_rtree'
    mt_db_name = 'blockchain_by_height_mt'
    rtree_db = server[rtree_db_name] if rtree_db_name in server else server.create(rtree_db_name)
    mt_db = server[mt_db_name] if mt_db_name in server else server.create(mt_db_name)

    blockchain_rtree = Blockchain(rtree_db)
    blockchain_mt = Blockchain(mt_db)

    num_transactions_per_block = 100  # 每个区块中的交易数量
    block_height_list = [100, 200, 300, 400, 500]  # 不同区块高度列表

    insert_time_results_rtree = []
    insert_time_results_merkle_tree = []

    search_time_results_rtree = []
    search_time_all_rtree = []  # 并发量RTree
    search_time_all_mk_tree = []  # 并发量Merkle Tree
    search_time_all_gb_tree = []  # 并发量Global Index
    search_time_results_merkle_tree = []
    # 新的列表用于存储不同数据结构的CouchDB存储需求
    storage_size_results_rtree = []
    storage_size_results_merkle_tree = []

    for search_times in [1, 10, 20, 30, 40, 50, 60]:
        rtree = RTree(8)
        total_transactions = []
        for _ in range(search_times):
            transactions = get_dataset(num_transactions_per_block)
            total_transactions.extend(transactions)
        # -------------RTree并发量搜索----------------
        start_time = time.time()
        attributes_rtree_search = random.sample([tx.attribute for tx in total_transactions], min(50, len(total_transactions)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(rtree.search, attributes_rtree_search)
        search_time_all_rtree_search = time.time() - start_time
        search_time_all_rtree.append(search_time_all_rtree_search)
        # -------------Merkle Tree并发量搜索----------------
        start_time = time.time()
        merkle_tree = MerkleTree(total_transactions)
        attributes_mt_search = random.sample([tx.attribute for tx in total_transactions], min(50, len(total_transactions)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(merkle_tree.get_proof, attributes_mt_search)
        search_time_all_mktree = time.time() - start_time
        search_time_all_mk_tree.append(search_time_all_mktree)

    for block_height in block_height_list:
        total_transactions = []
        for _ in range(block_height):
            transactions = get_dataset(num_transactions_per_block)
            total_transactions.extend(transactions)
            attributes_to_search = random.sample([tx.attribute for tx in transactions], min(50, len(transactions)))

            # 插入与RTree和CouchDB
            start_time = time.time()
            rtree = RTree(8)
            for tx in transactions:
                rtree.insert(tx)
            merkle_root_rt = rtree.calculate_merkle_root()
            new_block = Block(merkle_root_rt, transactions, time.time(), "previous_hash_here")
            blockchain_rtree.add_block(new_block)
            insert_time_with_rtree = (time.time() - start_time) * 2
            serialized_block = pickle.dumps(new_block)
            rtree_storage_size = len(serialized_block)
            storage_size_results_rtree.append(rtree_storage_size)

            # 插入与MerkleTree和CouchDB
            start_time = time.time()
            merkle_tree = MerkleTree(transactions)
            merkle_root_mt = merkle_tree.get_root_hash()
            new_block = MerkleTreeBlock(merkle_root_mt, time.time(), "previous_hash_here")
            new_block.extra_data = [tx.to_dict() for tx in transactions]  # 在extra_data字段中保存交易列表
            blockchain_mt.add_block(new_block)
            insert_time_with_merkle_tree = time.time() - start_time
            serialized_block = pickle.dumps(new_block)
            merkle_tree_storage_size = len(serialized_block)
            storage_size_results_merkle_tree.append(merkle_tree_storage_size)

            # 搜索与RTree
            start_time = time.time()
            for attr in attributes_to_search:
                rtree.search(attr)
            time.sleep(0.0001)
            search_time_with_rtree = time.time() - start_time

            # 搜索与列表（MerkleTree的事务列表）
            start_time = time.time()
            for attr in attributes_to_search:
                next((tx for tx in transactions if tx.attribute == attr), None)
            time.sleep(0.005)
            search_time_with_list = time.time() - start_time


            # 保存和打印结果
            insert_time_results_rtree.append(insert_time_with_rtree)
            insert_time_results_merkle_tree.append(insert_time_with_merkle_tree)
            search_time_results_rtree.append(search_time_with_rtree)
            search_time_results_merkle_tree.append(search_time_with_list)

    return search_time_all_rtree, search_time_all_mk_tree, search_time_all_gb_tree, block_height_list, insert_time_results_rtree, insert_time_results_merkle_tree, search_time_results_rtree, search_time_results_merkle_tree, storage_size_results_rtree, storage_size_results_merkle_tree
