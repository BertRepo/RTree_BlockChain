from global_method import plot_tree_results, plot_results
from in_memory_test import in_memory_test
from with_external_databases_test import with_external_databases_test, compare_trees_by_block_height

# 测试RTree和MerkleTree在插入和查找操作上的性能
def plot_memory_test():
    """
    内存中测试并绘制
    包括交易数量、RTree插入时间、Merkle Tree插入时间、RTree查找时间和Merkle Tree查找时间
    """
    num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt = in_memory_test()
    # 绘制
    plot_tree_results(
        num_transactions,
        insert_time_results_r_tree,
        insert_time_results_mt,
        search_time_results_r_tree,
        search_time_results_mt
    )

# 不同交易量下测试
def plot_database_tran_test():
    """连接数据库测试, 交易量变"""

    search_time_all_rtree, \
    search_time_all_mk_tree, \
    num_transactions_list, \
    insert_time_results_rtree, \
    insert_time_results_merkle_tree, \
    search_time_results_rtree, \
    search_time_results_list, \
    storage_size_results_rtree, \
    storage_size_results_merkle_tree  = with_external_databases_test()

    # 绘制
    plot_results(
        search_time_all_rtree,
        search_time_all_mk_tree,
        num_transactions_list,
        insert_time_results_rtree,
        insert_time_results_merkle_tree,
        search_time_results_rtree,
        search_time_results_list,
        storage_size_results_rtree,
        storage_size_results_merkle_tree
    )


# 不同区块高度
def plot_database_height_test():
    search_time_all_r_tree, search_time_all_mk_tree, search_time_all_gb_tree, block_height_list, insert_time_results_r_tree, insert_time_results_merkle_tree, search_time_results_r_tree, search_time_results_merkle_tree, insert_time_results_global_index, search_time_results_global_index, storage_size_results_r_tree, storage_size_results_merkle_tree, storage_size_results_global_index = compare_trees_by_block_height()

    # 绘制
    plot_results(
        search_time_all_r_tree,
        search_time_all_mk_tree,
        search_time_all_gb_tree,
        block_height_list,
        insert_time_results_r_tree,
        insert_time_results_merkle_tree,
        search_time_results_r_tree,
        search_time_results_merkle_tree,
        insert_time_results_global_index,
        search_time_results_global_index,
        storage_size_results_r_tree,
        storage_size_results_merkle_tree,
        storage_size_results_global_index,
        is_block=True
    )


if __name__ == '__main__':
    # plot_memory_test()
    plot_database_tran_test()
    # plot_database_height_test()
