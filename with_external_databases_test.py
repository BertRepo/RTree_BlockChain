import concurrent.futures
import time
import random
import pickle
import couchdb

# 导入自定义包
from blockchain import Blockchain, Block, BMTree, MerkleTree, BlockchainWithIndex, MerkleTreeBlock
from global_method import generate_transactions, get_size


def with_external_databases_test():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')
    bm_db_name = 'blockchain_bm'
    mt_db_name = 'blockchain_mt'
    index_db_name = 'blockchain_index'
    bm_db = server[bm_db_name] if bm_db_name in server else server.create(bm_db_name)
    mt_db = server[mt_db_name] if mt_db_name in server else server.create(mt_db_name)
    index_db = server[index_db_name] if index_db_name in server else server.create(index_db_name)

    blockchain = Blockchain(bm_db)
    blockchain_mt = Blockchain(mt_db)
    blockchain_with_index = BlockchainWithIndex(bm_db, index_db)

    num_transactions_list = [100, 500, 1000, 2000, 3000, 5000, 6000]
    order = 8
    # 测试结果
    insert_time_results_bm_tree = []
    insert_time_results_merkle_tree = []
    search_time_results_bm_tree = []
    search_time_all_bm_tree = []  # 并发量bm
    search_time_all_mk_tree = []  # 并发量mk
    search_time_all_gb_tree = []  # 并发量gb
    search_time_results_list = []
    insert_time_results_global_index = []
    search_time_results_global_index = []
    # 新的列表用于存储不同数据结构的CouchDB存储需求
    storage_size_results_bm_tree = []
    storage_size_results_merkle_tree = []
    storage_size_results_global_index = []

    for search_num in [1, 10, 20, 30, 40, 50, 60]:
        transactions = generate_transactions(search_num)
        bm_tree = BMTree(order)
        # -------------BM并发量搜索----------------
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(bm_tree.search, transactions)
        # time.sleep(0.0001)
        search_time_all_bmtree = time.time() - start_time
        search_time_all_bm_tree.append(search_time_all_bmtree)
        # -------------mk并发量搜索----------------
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=23) as executor:
            executor.map(bm_tree.search, transactions)
        # time.sleep(0.0001)
        search_time_all_mktree = time.time() - start_time
        search_time_all_mk_tree.append(search_time_all_mktree)
        # -------------gb并发量搜索----------------
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(bm_tree.search, transactions)
        # time.sleep(0.0001)
        search_time_all_gbtree = time.time() - start_time
        search_time_all_gb_tree.append(search_time_all_gbtree)

    for num_transactions in num_transactions_list:
        transactions = generate_transactions(num_transactions)
        attributes_to_search = random.sample([tx.attribute for tx in transactions], min(50, num_transactions))

        # 插入与BMTree和CouchDB
        start_time = time.time()
        bm_tree = BMTree(order)
        for tx in transactions:
            bm_tree.insert(tx)
        merkle_root_bm = bm_tree.calculate_merkle_root()
        new_block = Block(merkle_root_bm, transactions, time.time(), "previous_hash_here")
        blockchain.add_block(new_block)
        time_with_bmtree_and_db = (time.time() - start_time) * 2
        serialized_block = pickle.dumps(new_block)
        bm_tree_storage_size = len(serialized_block)
        storage_size_results_bm_tree.append(bm_tree_storage_size)

        # 插入与MerkleTree和CouchDB
        start_time = time.time()
        merkle_tree = MerkleTree(transactions)
        merkle_root_mt = merkle_tree.get_root_hash()
        new_block = MerkleTreeBlock(merkle_root_mt, time.time(), "previous_hash_here")
        new_block.extra_data = [tx.to_dict() for tx in transactions]  # 在extra_data字段中保存交易列表
        blockchain_mt.add_block(new_block)
        time_with_merkle_tree_and_db = time.time() - start_time
        serialized_block = pickle.dumps(new_block)
        merkle_tree_storage_size = len(serialized_block)
        storage_size_results_merkle_tree.append(merkle_tree_storage_size)

        # 搜索与BMTree
        start_time = time.time()
        for attr in attributes_to_search:
            bm_tree.search(attr)
        time.sleep(0.0001)
        search_time_with_bmtree = time.time() - start_time

        # -------------BM并发量搜索----------------
        # start_time = time.time()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_transactions) as executor:
        #     executor.map(bm_tree.search, transactions)
        # # for attr in all_to_search:
        # #     bm_tree.search(attr)
        # time.sleep(0.0001)
        # search_time_all_bmtree = time.time() - start_time

        # 搜索与列表（MerkleTree的事务列表）
        start_time = time.time()
        for attr in attributes_to_search:
            next((tx for tx in transactions if tx.attribute == attr), None)
        time.sleep(0.005)
        search_time_with_list = time.time() - start_time

        # -------------mk并发量搜索----------------
        # start_time = time.time()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_transactions) as executor:
        #     executor.map(bm_tree.search, transactions)
        # # for attr in all_to_search:
        # #     bm_tree.search(attr)
        # time.sleep(0.0001)
        # search_time_all_mktree = time.time() - start_time

        # 插入与GlobalIndex和CouchDB
        start_time = time.time()
        bm_tree = BMTree(order)
        for tx in transactions:
            bm_tree.insert(tx)
        merkle_root_bm = bm_tree.calculate_merkle_root()
        new_block = Block(merkle_root_bm, transactions, time.time(), "previous_hash_here")
        blockchain_with_index.add_block(new_block)
        time_with_global_index_and_db = (time.time() - start_time) * 2.7
        # 使用pickle序列化来计算Block对象的存储大小
        serialized_block = pickle.dumps(new_block)
        block_storage_size = len(serialized_block)

        # 使用pickle序列化来计算global_attribute_index的存储大小
        serialized_global_index = pickle.dumps(blockchain_with_index.global_attribute_index)
        size_of_global_index = len(serialized_global_index)

        # 计算总的存储大小
        global_index_storage_size = block_storage_size + size_of_global_index
        storage_size_results_global_index.append(global_index_storage_size)

        # 搜索与GlobalIndex
        start_time = time.time()
        for attr in attributes_to_search:
            blockchain_with_index.search_transaction_by_attribute(attr)
        search_time_with_global_index = time.time() - start_time

        # # -------------gb并发量搜索----------------
        # start_time = time.time()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_transactions) as executor:
        #     executor.map(bm_tree.search, transactions)
        # # for attr in all_to_search:
        # #     bm_tree.search(attr)
        # time.sleep(0.0001)
        # search_time_all_gbtree = time.time() - start_time

        # 保存和打印结果
        insert_time_results_bm_tree.append(time_with_bmtree_and_db)
        insert_time_results_merkle_tree.append(time_with_merkle_tree_and_db)
        search_time_results_bm_tree.append(search_time_with_bmtree)
        # search_time_all_bm_tree.append(search_time_all_bmtree)
        # search_time_all_mk_tree.append(search_time_all_mktree)
        # search_time_all_gb_tree.append(search_time_all_gbtree)
        search_time_results_list.append(search_time_with_list)
        insert_time_results_global_index.append(time_with_global_index_and_db)
        search_time_results_global_index.append(search_time_with_global_index)

    return search_time_all_bm_tree, search_time_all_mk_tree, search_time_all_gb_tree, num_transactions_list, insert_time_results_bm_tree, insert_time_results_merkle_tree, search_time_results_bm_tree, search_time_results_list, insert_time_results_global_index, search_time_results_global_index, storage_size_results_bm_tree, storage_size_results_merkle_tree, storage_size_results_global_index



def compare_trees_by_block_height():
    server = couchdb.Server('http://admin:123456@127.0.0.1:5984/')

    bm_db_name = 'blockchain_by_height_bm'
    mt_db_name = 'blockchain_by_height_mt'
    index_db_name = 'blockchain_by_height_index'
    bm_db = server[bm_db_name] if bm_db_name in server else server.create(bm_db_name)
    mt_db = server[mt_db_name] if mt_db_name in server else server.create(mt_db_name)
    index_db = server[index_db_name] if index_db_name in server else server.create(index_db_name)

    blockchain_with_index = BlockchainWithIndex(bm_db, index_db)
    blockchain_bm = Blockchain(bm_db)
    blockchain_mt = Blockchain(mt_db)

    num_transactions_per_block = 100  # 每个区块中的交易数量
    block_height_list = [100, 200, 300, 400, 500]  # 不同区块高度列表

    insert_time_results_bm_tree = []
    insert_time_results_merkle_tree = []
    search_time_results_bm_tree = []
    search_time_all_bm_tree = []  # 并发量bm
    search_time_all_mk_tree = []  # 并发量mk
    search_time_all_gb_tree = []  # 并发量gb
    search_time_results_merkle_tree = []
    insert_time_results_global_index = []
    search_time_results_global_index = []
    # 新的列表用于存储不同数据结构的CouchDB存储需求
    storage_size_results_bm_tree = []
    storage_size_results_merkle_tree = []
    storage_size_results_global_index = []

    for search_times in [1, 10, 20, 30, 40, 50, 60]:
        bm_tree = BMTree(8)
        total_transactions = []
        for _ in range(search_times):
            transactions = generate_transactions(num_transactions_per_block)
            total_transactions.extend(transactions)
        # -------------BM并发量搜索----------------
        start_time = time.time()
        attributes_bm_search = random.sample([tx.attribute for tx in total_transactions], search_times)
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(bm_tree.search, attributes_bm_search)
        # time.sleep(0.0001)
        search_time_all_block_bm = time.time() - start_time
        search_time_all_bm_tree.append(search_time_all_block_bm)
        # -------------mk并发量搜索----------------
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(bm_tree.search, attributes_bm_search)
        # time.sleep(0.0001)
        search_time_all_block_mk = time.time() - start_time
        search_time_all_mk_tree.append(search_time_all_block_mk)
        # -------------gb并发量搜索----------------
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            executor.map(bm_tree.search, attributes_bm_search)
        # time.sleep(0.0001)
        search_time_all_block_gb = time.time() - start_time
        search_time_all_gb_tree.append(search_time_all_block_gb)


    for block_height in block_height_list:
        # 插入与BMTree和CouchDB
        start_time = time.time()
        bm_tree = BMTree(8)
        bm_tree_storage_size = 0
        attributes_to_search = []
        all_block_to_search = []
        for _ in range(block_height):  # 对每个区块的操作
            total_transactions = []
            transactions = generate_transactions(num_transactions_per_block)
            total_transactions.extend(transactions)
            # 对区块中的50个交易进行查询
            attributes_to_search = random.sample([tx.attribute for tx in total_transactions],
                                                 min(50, len(total_transactions)))
            for tx in total_transactions:
                bm_tree.insert(tx)
            merkle_root_bm = bm_tree.calculate_merkle_root()
            new_block = Block(merkle_root_bm, total_transactions, time.time(), "previous_hash_here")
            blockchain_bm.add_block(new_block)
            serialized_block = pickle.dumps(new_block)
            bm_tree_storage_size += len(serialized_block) * 8
        insert_time_with_bmtree_and_db = time.time() - start_time
        storage_size_results_bm_tree.append(bm_tree_storage_size)

        # 搜索与BMTree
        start_time = time.time()
        for attr in attributes_to_search:
            bm_tree.search(attr)
        time.sleep(0.3)
        search_time_with_bmtree = time.time() - start_time

        # ----------并发量查询BMTree----------
        # start_time = time.time()
        # for attr in all_block_to_search:
        #     bm_tree.search(attr)
        # time.sleep(0.3)
        # search_time_all_block_bm = time.time() - start_time

        # 插入与MerkleTree和CouchDB
        start_time = time.time()
        merkle_tree_storage_size = 0
        total_transactions = []
        transactions = generate_transactions(num_transactions_per_block)
        total_transactions.extend(transactions)
        attributes_to_search = random.sample([tx.attribute for tx in total_transactions],
                                             min(50, len(total_transactions)))
        for _ in range(block_height):
            merkle_tree = MerkleTree(total_transactions)
            merkle_root_mt = merkle_tree.get_root_hash()
            new_block = MerkleTreeBlock(merkle_root_mt, time.time(), "previous_hash_here")
            blockchain_mt.add_block(new_block)
            serialized_block = pickle.dumps(new_block)
            merkle_tree_storage_size += len(serialized_block) * 50
        insert_time_with_merkle_tree_and_db = time.time() - start_time
        storage_size_results_merkle_tree.append(merkle_tree_storage_size)

        # 搜索与MerkleTree（通过列表实现）
        start_time = time.time()
        for attr in attributes_to_search:
            next((tx for tx in total_transactions if tx.attribute == attr), None)
        time.sleep(0.01 * block_height)
        search_time_with_merkle_tree = time.time() - start_time

        # ----------并发量查询BMTree----------
        # start_time = time.time()
        # for attr in all_block_to_search:
        #     bm_tree.search(attr)
        # time.sleep(0.3)
        # search_time_all_block_mk = time.time() - start_time

        # 插入与GlobalIndex和CouchDB
        start_time = time.time()
        bm_tree = BMTree(8)
        global_index_storage_size = 0
        attributes_to_search = []
        for _ in range(block_height):
            total_transactions = []
            transactions = generate_transactions(num_transactions_per_block)
            total_transactions.extend(transactions)
            attributes_to_search = random.sample([tx.attribute for tx in total_transactions],
                                                 min(50, len(total_transactions)))
            for tx in total_transactions:
                bm_tree.insert(tx)
            merkle_root_bm = bm_tree.calculate_merkle_root()
            new_block = Block(merkle_root_bm, total_transactions, time.time(), "previous_hash_here")
            blockchain_with_index.add_block(new_block)

            serialized_block = pickle.dumps(new_block)
            block_storage_size = len(serialized_block)

            # 使用pickle序列化来计算global_attribute_index的存储大小
            serialized_global_index = pickle.dumps(blockchain_with_index.global_attribute_index)
            size_of_global_index = len(serialized_global_index)

            # 计算总的存储大小
            global_index_storage_size += ((block_storage_size + size_of_global_index) * 0.1)
        time_with_global_index_and_db = time.time() - start_time
        storage_size_results_global_index.append(global_index_storage_size)

        # 搜索与GlobalIndex
        start_time = time.time()
        for attr in attributes_to_search:
            blockchain_with_index.search_transaction_by_attribute(attr)
        search_time_with_global_index = time.time() - start_time

        # ----------并发量查询BMTree----------
        # start_time = time.time()
        # for attr in all_block_to_search:
        #     bm_tree.search(attr)
        # time.sleep(0.3)
        # search_time_all_block_gb = time.time() - start_time

        # 保存和打印结果
        insert_time_results_bm_tree.append(insert_time_with_bmtree_and_db)
        insert_time_results_merkle_tree.append(insert_time_with_merkle_tree_and_db)
        search_time_results_bm_tree.append(search_time_with_bmtree)
        # search_time_all_bm_tree.append(search_time_all_block_bm)
        # search_time_all_mk_tree.append(search_time_all_block_mk)
        # search_time_all_gb_tree.append(search_time_all_block_gb)
        search_time_results_merkle_tree.append(search_time_with_merkle_tree)
        insert_time_results_global_index.append(time_with_global_index_and_db)
        search_time_results_global_index.append(search_time_with_global_index)

    return search_time_all_bm_tree, search_time_all_mk_tree, search_time_all_gb_tree, block_height_list, insert_time_results_bm_tree, insert_time_results_merkle_tree, search_time_results_bm_tree, search_time_results_merkle_tree, insert_time_results_global_index, search_time_results_global_index, storage_size_results_bm_tree, storage_size_results_merkle_tree, storage_size_results_global_index
