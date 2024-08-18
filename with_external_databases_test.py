import concurrent.futures
import time
import random
import pickle
import couchdb

# 导入自定义包
from blockchain import Transaction, Blockchain, Block, RTree, MerkleTree, MerkleTreeBlock
from global_method import get_dataset, get_random_nonexistent_data


def intersects(mbr1, mbr2):
    return all(mbr1[i] <= mbr2[i + len(mbr1) // 2] and mbr2[i] <= mbr1[i + len(mbr1) // 2] for i in range(len(mbr1) // 2))

def with_external_databases_test():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')
    rtree_db_name = 'blockchain_rtree'
    rtree_mbr_db_name = 'blockchain_rtree_mbr'
    mt_db_name = 'blockchain_mt'
    rtree_db = server[rtree_db_name] if rtree_db_name in server else server.create(rtree_db_name)
    rtree_mbr_db = server[rtree_mbr_db_name] if rtree_mbr_db_name in server else server.create(rtree_mbr_db_name)
    mt_db = server[mt_db_name] if mt_db_name in server else server.create(mt_db_name)
    blockchain = Blockchain(rtree_db)
    blockchain_mbr = Blockchain(rtree_mbr_db)
    blockchain_mt = Blockchain(mt_db)

    # 不同交易量
    num_transactions_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # 节点内最大交易量
    order = 1
    # 区块内最大交易量
    n = 10
    # 属性数量
    d = 2 # 4 8

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
        print(f"当前交易数量：{num_transactions}")
        transactions = get_dataset(num_transactions)

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
            # TODO： 可以作为区块头 存放根节点mbr
            # new_block.extra_data = [tx.to_dict() for tx in tx_batch]  # 在 extra_data 字段中保存交易列表
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
            new_block = Block(merkle_root_rt_mbr, rtree_mbr, tx_batch, time.time(), "previous_hash_here", n, rtree_mbr.root.bounds)
            # TODO： 可以作为区块头 存放根节点mbr
            # new_block.extra_data = [tx.to_dict() for tx in tx_batch]  # 在 extra_data 字段中保存交易列表
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
                block.tree.search(transactions, attr)
        search_time_with_list = time.time() - start_time

        """-----------对不存在的查询条件---------"""
        # 随机生成假数据
        no_attributes_to_search = get_random_nonexistent_data(transactions, min(50, num_transactions))
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
                block.tree.search(transactions, attr)
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

        # print(search_time_results_rtree)
        # print(search_time_results_merkle_tree)
        
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

    return search_time_all_rtree, search_time_all_mk_tree, block_height_list, insert_time_results_rtree, insert_time_results_merkle_tree, search_time_results_rtree, search_time_results_merkle_tree, storage_size_results_rtree, storage_size_results_merkle_tree
