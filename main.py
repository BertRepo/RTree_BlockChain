from plot import plot_tree_self, plot_every_n, plot_every_d, plot_every_t
from calc import calc_tree_self, calc_every_n, calc_every_d, calc_every_t


def run_tree_self():
    """
    测试RTree和MerkleTree在插入和查找操作上的性能
    包括交易数量、RTree插入时间、Merkle Tree插入时间、RTree查找时间和Merkle Tree查找时间
    """
    num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt = calc_tree_self()
    # 绘制
    plot_tree_self(
        num_transactions,
        insert_time_results_r_tree,
        insert_time_results_mt,
        search_time_results_r_tree,
        search_time_results_mt
    )


def run_every_n():
    """不同区块交易量下测试"""
    d_list, \
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
    search_no_time_merkle_tree  = calc_every_n()

    # 绘制
    plot_every_n(
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
        search_no_time_merkle_tree
    )


def run_every_d():
    """不同属性个数下测试"""
    d_list, \
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
    search_no_time_merkle_tree  = calc_every_d()

    # 绘制
    plot_every_d(
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
        search_no_time_merkle_tree
    )


def run_every_t():
    """不同交易量下测试"""
    num_transactions, \
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
    search_no_time_results_merkle_tree  = calc_every_t()

    # 绘制
    plot_every_t(
        num_transactions,
        insert_time_results_rtree,
        insert_time_results_rtree_mbr,
        insert_time_results_merkle_tree,
        storage_size_results_rtree,
        storage_size_results_rtree_mbr,
        storage_size_results_merkle_tree,
        search_time_results_rtree,
        search_time_results_rtree_mbr,
        search_time_results_merkle_tree,
        search_no_time_results_rtree,
        search_no_time_results_rtree_mbr,
        search_no_time_results_merkle_tree
    )


if __name__ == '__main__':
    # 比较树本身
    # run_tree_self()

    # 不同区块交易量下
    run_every_n()

    # 不同属性个数下
    # run_every_d()

    # 不同总体交易量下
    # run_every_t()
