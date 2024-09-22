import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, LogFormatter

# 加载字体
from matplotlib.font_manager import FontProperties

# 使用macOS自带的字体 PingFang SC
font_path = "/System/Library/Fonts/PingFang.ttc"
font_prop = FontProperties(fname=font_path)

rcParams['font.sans-serif'] = [font_prop.get_name()]
rcParams['font.family'] = 'sans-serif'

config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": [font_prop.get_name()],
}
rcParams.update(config)

# rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 ['SimHei'] 使用黑体
# rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_every_n(
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
        search_no_time_merkle_tree):
    """绘制不同区块交易量下对比图"""

    # 绘制构建时间的比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, insert_time_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, insert_time_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, insert_time_merkle_tree[d], label='MH-tree', marker='x')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].legend()
    fig.suptitle('不同区块交易量下的构建时耗结果图', fontsize=16)
    plt.savefig('insert_times_comparison_n.pdf', format='pdf')
    plt.show()

    # 绘制存储容量的比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, storage_size_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, storage_size_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, storage_size_merkle_tree[d], label='MH-Tree', marker='x')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('大小(bytes)')
        axs[i].legend()
    fig.suptitle('不同区块交易量下的存储开销结果图', fontsize=16)
    plt.savefig('storage_size_comparison_n.pdf', format='pdf')
    plt.show()

    # 存在条件查询时间比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, search_time_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, search_time_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, search_time_merkle_tree[d], label='MH-Tree', marker='x')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        axs[i].yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    fig.suptitle('不同区块交易量下的存在条件查询时耗结果图', fontsize=16)
    plt.savefig('search_time_comparison_n.pdf', format='pdf')
    plt.show()

    # 不存在条件查询时间比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, search_no_time_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, search_no_time_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, search_no_time_merkle_tree[d], label='MH-Tree', marker='x')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        axs[i].yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    fig.suptitle('不同区块交易量下的不存在条件查询时耗结果图', fontsize=16)
    plt.savefig('search_no_time_comparison_n.pdf', format='pdf')
    plt.show()


def plot_every_d(
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
    search_no_time_merkle_tree):
    """绘制不同属性个数下对比图"""

    # 绘制构建时间的比较
    fig, axs = plt.subplots(1, len(n_list), figsize=(18, 6))
    for i, n in enumerate(n_list):
        axs[i].plot(d_list, insert_time_rtree[n], label='MR-Tree', marker='o')
        axs[i].plot(d_list, insert_time_rtree_mbr[n], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(d_list, insert_time_merkle_tree[n], label='MH-Tree', marker='x')
        axs[i].set_title(f'区块内交易量={n}')
        axs[i].set_xlabel('属性数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].legend()
    fig.suptitle('不同属性数量下的构建时耗结果图', fontsize=16)
    plt.savefig('insert_times_comparison_d.pdf', format='pdf')
    plt.show()

    # 绘制存储容量的比较
    fig, axs = plt.subplots(1, len(n_list), figsize=(18, 6))
    for i, n in enumerate(n_list):
        axs[i].plot(d_list, storage_size_rtree[n], label='MR-Tree', marker='o')
        axs[i].plot(d_list, storage_size_rtree_mbr[n], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(d_list, storage_size_merkle_tree[n], label='MH-Tree', marker='x')
        axs[i].set_title(f'区块内交易量={n}')
        axs[i].set_xlabel('属性数量(个)')
        axs[i].set_ylabel('大小(bytes)')
        axs[i].legend()
    fig.suptitle('不同属性数量下的存储开销结果图', fontsize=16)
    plt.savefig('storage_size_comparison_d.pdf', format='pdf')
    plt.show()

    # 存在条件下搜索时间的比较
    fig, axs = plt.subplots(1, len(n_list), figsize=(18, 6))
    for i, n in enumerate(n_list):
        axs[i].plot(d_list, search_time_rtree[n], label='MR-Tree', marker='o')
        axs[i].plot(d_list, search_time_rtree_mbr[n], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(d_list, search_time_merkle_tree[n], label='MH-Tree', marker='x')
        axs[i].set_title(f'区块内交易量={n}')
        axs[i].set_xlabel('属性数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].set_yscale('log')
        axs[i].legend()
    fig.suptitle('不同属性数量下的存在条件查询时耗结果图', fontsize=16)
    plt.savefig('search_time_comparison_d.pdf', format='pdf')
    plt.show()

    # 不存在条件下搜索时间的比较
    fig, axs = plt.subplots(1, len(n_list), figsize=(18, 6))
    for i, n in enumerate(n_list):
        axs[i].plot(d_list, search_no_time_rtree[n], label='MR-Tree', marker='o')
        axs[i].plot(d_list, search_no_time_rtree_mbr[n], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(d_list, search_no_time_merkle_tree[n], label='MH-Tree', marker='x')
        axs[i].set_title(f'区块内交易量={n}')
        axs[i].set_xlabel('属性数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].set_yscale('log')
        axs[i].legend()
    fig.suptitle('不同属性数量下的不存在条件查询时耗结果图', fontsize=16)
    plt.savefig('search_no_time_comparison_d.pdf', format='pdf')
    plt.show()


def plot_every_t(num_transactions,
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
        search_no_time_results_merkle_tree):
    """绘制不同总体交易量下对比图"""

    # 绘制插入时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, insert_time_results_rtree, label='R树', marker='o')
    plt.plot(num_transactions, insert_time_results_rtree_mbr, label='R树-header-with-mbr', marker='v')
    plt.plot(num_transactions, insert_time_results_merkle_tree, label='默克尔树', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(s)')
    plt.title('创建时间比较')
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('insert.pdf', format='pdf')
    plt.show()

    # 绘制存储容量的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, storage_size_results_rtree, label='R树', marker='o')
    plt.plot(num_transactions, storage_size_results_rtree_mbr, label='R树-header-with-mbr', marker='v')
    plt.plot(num_transactions, storage_size_results_merkle_tree, label='默克尔树', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('大小(bytes)')
    plt.title('存储空间大小比较')
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('storage.pdf', format='pdf')
    plt.show()

    # 存在条件 搜索比较
    plt.figure(figsize=(10, 5))
    plt.plot(num_transactions, search_time_results_rtree, label='R树', marker='o')
    plt.plot(num_transactions, search_time_results_rtree_mbr, label='R树-header-with-mbr', marker='v')
    plt.plot(num_transactions, search_time_results_merkle_tree, label='默克尔树', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(s)')
    plt.title('区块链中存在条件查询时间比较')
    ax = plt.gca()
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    # 设置纵坐标显示两位小数
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 设置纵坐标为对数刻度
    ax.set_yscale('log')
    # 自定义对数刻度的刻度
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('searchByTranNum.pdf', format='pdf')
    plt.show()

    # 存在条件 搜索比较
    plt.figure(figsize=(10, 5))
    plt.plot(num_transactions, search_no_time_results_rtree, label='R树', marker='o')
    plt.plot(num_transactions, search_no_time_results_rtree_mbr, label='R树-header-with-mbr', marker='v')
    plt.plot(num_transactions, search_no_time_results_merkle_tree, label='默克尔树', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(s)')
    plt.title('区块链中不存在条件查询时间比较')
    ax = plt.gca()
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    # 设置纵坐标显示两位小数
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 设置纵坐标为对数刻度
    ax.set_yscale('log')
    # 自定义对数刻度的刻度
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('searchNoByTranNum.pdf', format='pdf')
    plt.show()


def plot_tree_self(num_transactions, insert_time_results_r_tree, insert_time_results_mt, search_time_results_r_tree, search_time_results_mt):
    """绘制对比图"""
    # 绘制插入时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, insert_time_results_r_tree, label='R树创建时间', marker='o')
    plt.plot(num_transactions, insert_time_results_mt, label='默克尔树创建时间', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(ms)')
    plt.title('创建时间比较')

    # 设置纵坐标为对数刻度
    plt.yscale('log')
    # 设置纵坐标刻度位置
    plt.yticks([0.5, 1, 10, 100, 1000, 5000], ['0.5', '1', '10', '100', '1000', '5000'])

    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('insert_time_comparison.pdf', format='pdf')
    plt.show()

    # 绘制搜索时间的比较
    plt.figure(figsize=(10, 6))
    plt.plot(num_transactions, search_time_results_r_tree, label='R树查找时间', marker='o')
    plt.plot(num_transactions, search_time_results_mt, label='默克尔树查找时间', marker='x')
    plt.xlabel('交易数量')
    plt.ylabel('时间(ms)')
    plt.title('查询时间比较')

    # # 设置纵坐标为对数刻度
    # plt.yscale('log')
    # # 设置纵坐标刻度位置
    # plt.yticks([0.5, 1, 10, 100, 1000, 5000], ['0.5', '1', '10', '100', '1000', '5000'])

    plt.legend()
    # 设置导出的文件名和格式为PDF
    plt.savefig('search_time_comparison.pdf', format='pdf')
    plt.show()


def plot_trace_every_n(
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
        search_time_fabric_sort):
    """绘制 溯源 不同区块交易量下对比图"""

    # 绘制构建时间的比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, insert_time_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, insert_time_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, insert_time_fabric[d], label='Fabric', marker='x')
        axs[i].plot(n_list, insert_time_fabric_sort[d], label='$\mathrm{Fabric}^{\mathrm{+}}$', marker='s')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].legend()
    fig.suptitle('不同区块交易量下的区块链构建时耗结果图', fontsize=16)
    plt.savefig('history_insert_times_comparison_n.pdf', format='pdf')
    plt.show()

    # 绘制存储容量的比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, storage_size_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, storage_size_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, storage_size_fabric[d], label='Fabric', marker='x')
        axs[i].plot(n_list, storage_size_fabric_sort[d], label='$\mathrm{Fabric}^{\mathrm{+}}$', marker='s')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('大小(bytes)')
        axs[i].legend()
    fig.suptitle('不同区块交易量下的区块链存储开销结果图', fontsize=16)
    plt.savefig('history_storage_size_comparison_n.pdf', format='pdf')
    plt.show()

    # 存在条件查询时间比较
    fig, axs = plt.subplots(1, len(d_list), figsize=(18, 6))
    for i, d in enumerate(d_list):
        axs[i].plot(n_list, search_time_rtree[d], label='MR-Tree', marker='o')
        axs[i].plot(n_list, search_time_rtree_mbr[d], label='$\mathrm{MR-Tree}^{\mathrm{+}}$', marker='v')
        axs[i].plot(n_list, search_time_fabric[d], label='Fabric', marker='x')
        axs[i].plot(n_list, search_time_fabric_sort[d], label='$\mathrm{Fabric}^{\mathrm{+}}$', marker='s')
        axs[i].set_title(f'属性数量={d}')
        axs[i].set_xlabel('区块内交易数量(个)')
        axs[i].set_ylabel('时间(s)')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        axs[i].yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    fig.suptitle('不同区块交易量下的溯源时耗结果图', fontsize=16)
    plt.savefig('history_search_time_comparison_n.pdf', format='pdf')
    plt.show()


def load_experiment_results(file_path):
    """读取保存的实验结果 JSON 文件"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def plot_from_saved_data(file_path, plot_type='n'):
    """从保存的数据文件中读取并绘制图表"""
    results = load_experiment_results(file_path)

    if plot_type == 'n':
        plot_every_n(
            results['d_list'],
            results['n_list'],
            results['insert_time_rtree'],
            results['insert_time_rtree_mbr'],
            results['insert_time_merkle_tree'],
            results['storage_size_rtree'],
            results['storage_size_rtree_mbr'],
            results['storage_size_merkle_tree'],
            results['search_time_rtree'],
            results['search_time_rtree_mbr'],
            results['search_time_merkle_tree'],
            results['search_no_time_rtree'],
            results['search_no_time_rtree_mbr'],
            results['search_no_time_merkle_tree']
        )
    elif plot_type == 'd':
        plot_every_d(
            results['d_list'],
            results['n_list'],
            results['insert_time_rtree'],
            results['insert_time_rtree_mbr'],
            results['insert_time_merkle_tree'],
            results['storage_size_rtree'],
            results['storage_size_rtree_mbr'],
            results['storage_size_merkle_tree'],
            results['search_time_rtree'],
            results['search_time_rtree_mbr'],
            results['search_time_merkle_tree'],
            results['search_no_time_rtree'],
            results['search_no_time_rtree_mbr'],
            results['search_no_time_merkle_tree']
        )
    elif plot_type == "trace":
        plot_trace_every_n(
            results['d_list'],
            results['n_list'],
            results['insert_time_rtree'],
            results['insert_time_rtree_mbr'],
            results['insert_time_fabric'],
            results['insert_time_fabric_sort'],
            results['storage_size_rtree'],
            results['storage_size_rtree_mbr'],
            results['storage_size_fabric'],
            results['storage_size_fabric_sort'],
            results['search_time_rtree'],
            results['search_time_rtree_mbr'],
            results['search_time_fabric'],
            results['search_time_fabric_sort']
        )
    else:
        print("Invalid plot type. Please use 'n' or 'd'.")


# # 调用函数绘制图表
# plot_from_saved_data('output/every_n_results.json', plot_type='n')
# plot_from_saved_data('output/every_d_results.json', plot_type='d')
# plot_from_saved_data('output/history_every_n_results.json', plot_type='trace')