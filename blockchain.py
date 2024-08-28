import hashlib
import json
import numpy as np
from datetime import datetime

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)  # Convert numpy int64 to standard Python int
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, RTree):
        return obj.to_dict()
    elif isinstance(obj, MerkleTree):
        return obj.to_dict()
    else:
        return obj

# 定义 RTree 的自定义序列化函数
def serialize_rtree(obj):
    if isinstance(obj, RTree):
        return obj.to_dict()
    elif isinstance(obj, RTreeNode):
        return obj.to_dict()
    elif isinstance(obj, Transaction):
        return obj.to_dict()
    # elif isinstance(obj, pd.Timestamp):
    #     return obj.isoformat()  # 将时间戳转换为 ISO 8601 格式字符串
    # elif pd.isna(obj):  # 检查是否为 NaT 或其他缺失值
    #     return None  # 将 NaT 或 NaN 转换为 None
    # # 处理其他类型的 RTree 对象的序列化
    raise TypeError(f"Type {type(obj)} not serializable")

def serialize_merkle_tree(obj):
    if isinstance(obj, MerkleTree):
        return obj.to_dict()
    elif isinstance(obj, MerkleNode):
        return obj.to_dict()
    elif isinstance(obj, Transaction):
        return obj.to_dict()
    # elif isinstance(obj, pd.Timestamp):
    #     return obj.isoformat()  # 将时间戳转换为 ISO 8601 格式字符串
    # elif pd.isna(obj):  # 检查是否为 NaT 或其他缺失值
    #     return None  # 将 NaT 或 NaN 转换为 None
    raise TypeError(f"Type {type(obj)} not serializable")

'''
    R树和默克尔树共用部分 交易类 和 区块链类
'''
# 交易类  R树和默克尔树对应的区块共用这个类 TODO: 后面考虑分开实现 因为其中包含边界值
class Transaction:
    def __init__(self, tx_hash, attribute, bounds=None):
        self.tx_hash = tx_hash
        self.attribute = attribute  # 用作索引的属性
        self.bounds = bounds  # 用于 R 树插入的边界

    # 计算交易的哈希值
    def calculate_hash(self):
        return hashlib.sha256(json.dumps(self.tx_hash, sort_keys=True).encode()).hexdigest()

    # 将交易转换为字典格式
    def to_dict(self):
        return {
            'tx_hash': self.tx_hash,
            'attribute': self.attribute,
            'bounds': self.bounds
        }

    # 从字典格式创建交易实例
    @classmethod
    def from_dict(cls, data):
        return cls(
            tx_hash=data['tx_hash'],
            attribute=data['attribute'],
            bounds=data['bounds']
        )

# 区块链类
class Blockchain:
    def __init__(self, db, max_transactions=8):
        self.chain = []
        self.db = db
        self.max_transactions = max_transactions
        self.current_transactions = []
        # self._load_chain_from_db()

    def _load_chain_from_db(self):
        for id in self.db:
            doc = self.db[id]
            block_type = doc.get('type', None)

            transactions = [Transaction.from_dict(tx_dict) for tx_dict in doc['transactions']]

            # TODO: 这个判断 屎山代码 后面有空最好修改下 没有必要判断每个区块类型（同区块类型都是放在同一个数据库中的）
            if block_type == 'RTreeBlock':
                
                block = Block(r_tree_root=doc['r_tree_root'], tree=doc['tree'], transactions=transactions,
                              timestamp=doc['timestamp'], prev_hash=doc['prev_hash'],
                              max_transactions=self.max_transactions)
            elif block_type == 'MerkleTreeBlock':
                
                block = MerkleTreeBlock(merkle_root=doc['merkle_root'], tree=doc['tree'], transactions=transactions,
                                        timestamp=doc['timestamp'], prev_hash=doc['prev_hash'],
                                        max_transactions=self.max_transactions)
            else:
                print(f"Warning: Unknown block type for block id {id}. Skipping.")
                continue

            self.chain.append(block)

    def add_transaction(self, transaction):
        self.current_transactions.append(transaction)
        if len(self.current_transactions) >= self.max_transactions:
            self._create_new_block()

    def _create_new_block(self):
        prev_block = self.chain[-1] if self.chain else None
        prev_hash = prev_block.calculate_hash() if prev_block else None
        r_tree = RTree()
        for tx in self.current_transactions:
            r_tree.insert(tx)
        r_tree_root = r_tree.calculate_merkle_root()
        new_block = Block(r_tree_root=r_tree_root, transactions=self.current_transactions,
                          timestamp=self._get_current_timestamp(), prev_hash=prev_hash,
                          max_transactions=self.max_transactions)
        self.add_block(new_block)
        self.current_transactions = []

        # 如果需要创建MerkleTree区块，可以在这里创建并添加
        # merkle_tree = MerkleTree(self.current_transactions)
        # merkle_block = MerkleTreeBlock(merkle_root=merkle_tree.get_root_hash(), transactions=self.current_transactions,
        #                                timestamp=self._get_current_timestamp(), prev_hash=prev_hash,
        #                                max_transactions=self.max_transactions)
        # self.add_block(merkle_block)
        # self.current_transactions = []

    def add_block(self, block):
        self.chain.append(block)
        transactions_dict = convert_ndarray_to_list([tx.to_dict() for tx in block.transactions])
        if isinstance(block, Block):
            self.db.save({
                'type': 'RTreeBlock',
                'r_tree_root': block.r_tree_root,
                'timestamp': block.timestamp,
                'prev_hash': block.prev_hash,
                'tree': json.dumps(block.tree, default=serialize_rtree) if isinstance(block.tree, RTree) else block.tree,
                'transactions': transactions_dict,
                'extra_data': block.extra_data
            })
        elif isinstance(block, MerkleTreeBlock):
            self.db.save({
                'type': 'MerkleTreeBlock',
                'merkle_root': block.merkle_root,
                'timestamp': block.timestamp,
                'prev_hash': block.prev_hash,
                'tree': json.dumps(block.tree, default=serialize_merkle_tree) if isinstance(block.tree, MerkleTree) else block.tree,
                'transactions': transactions_dict,
                'extra_data': block.extra_data
            })
        else:
            print("Unsupported block type")


    def validate_block(self, prev_block, new_block):
        return new_block.prev_hash == prev_block.calculate_hash()

    # TODO: 肯定要修改的
    def search_transaction_by_attribute(self, attribute):
        for block in self.chain:
            matching_transactions = [tx for tx in block.transactions if tx.attribute == attribute]
            if matching_transactions:
                return matching_transactions
        return None

    def _get_current_timestamp(self):
        return datetime.now().timestamp()


'''
    R树部分
'''

# R 树节点类
class RTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []  # 叶子节点存放数据，非叶子节点存放子节点
        self.bounds = None  # 存储节点的边界

    def update_bounds(self):
        if self.entries:
            min_bounds = [min(entry[1][i] for entry in self.entries) for i in range(len(self.entries[0][1]) // 2)]
            max_bounds = [max(entry[1][i + len(self.entries[0][1]) // 2] for entry in self.entries) for i in range(len(self.entries[0][1]) // 2)]
            self.bounds = tuple(min_bounds + max_bounds)
        else:
            self.bounds = None  # 没有条目时设置边界为空

    def to_dict(self):
        return {
            'is_leaf': self.is_leaf,
            'entries': [(entry[0], entry[1]) for entry in self.entries],
            'bounds': self.bounds
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(is_leaf=data['is_leaf'])
        node.entries = [(entry[0], entry[1]) for entry in data['entries']]
        node.bounds = data['bounds']
        return node


# R 树类
class RTree:
    def __init__(self, max_entries=4):
        self.root = RTreeNode()
        self.max_entries = max_entries # 每个节点中允许的最大条目数

    def insert(self, tx):
        entry = (tx, tx.bounds)  # 将交易和边界组成元组
        node = self._choose_leaf(self.root, entry)  # 从根节点开始，递归地选择最合适的叶节点
        node.entries.append(entry)  # 插入新条目到选中的叶节点
        node.update_bounds()  # 更新叶节点的边界

        if len(node.entries) > self.max_entries:  # 节点中的条目数超过了节点的最大容量，就对该节点进行分裂处理
            self._split_node(node)

        self._adjust_tree(node)

    def search(self, bounds):
        results = []
        self._search(self.root, bounds, results)
        # print(f"R树查询结果: {results}")
        return results

    def calculate_merkle_root(self):
        # 假设有一个从 RTree 计算 Merkle 根的方法
        return hashlib.sha256(json.dumps("dummy_root").encode()).hexdigest()

    def _choose_leaf(self, node, entry):
        if node.is_leaf:
            return node
        else:
            best_child = min(node.entries, key=lambda child: self._calc_enlargement(child[0].bounds, entry[1]))
            return self._choose_leaf(best_child[0], entry)

    # # 二分分裂策略
    # def _split_node(self, node):
    #     mid = len(node.entries) // 2
    #     new_node = RTreeNode(is_leaf=node.is_leaf)
    #     node.entries, new_node.entries = node.entries[:mid], node.entries[mid:]
    #     node.update_bounds()
    #     new_node.update_bounds()
    #
    #     if node == self.root:
    #         new_root = RTreeNode(is_leaf=False)
    #         new_root.entries.append((self.root, self.root.bounds))
    #         new_root.entries.append((new_node, new_node.bounds))
    #         new_root.update_bounds()
    #         self.root = new_root
    #     else:
    #         parent = self._find_parent(self.root, node)
    #         parent.entries.append((new_node, new_node.bounds))
    #         parent.update_bounds()
    #         if len(parent.entries) > self.max_entries:
    #             self._split_node(parent)

    # 排序分裂策略
    def _split_node(self, node):
        mid = len(node.entries) // 2
        new_node = RTreeNode(is_leaf=node.is_leaf)

        sorted_entries = sorted(node.entries, key=lambda entry: entry[1])
        node.entries, new_node.entries = sorted_entries[:mid], sorted_entries[mid:]

        node.update_bounds()
        new_node.update_bounds()

        if node == self.root:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries.append((self.root, self.root.bounds))
            new_root.entries.append((new_node, new_node.bounds))
            new_root.update_bounds()
            self.root = new_root
        else:
            parent = self._find_parent(self.root, node)
            parent.entries.append((new_node, new_node.bounds))
            parent.update_bounds()
            if len(parent.entries) > self.max_entries:
                self._split_node(parent)

    def _find_parent(self, current_node, target_node):
        if current_node.is_leaf or current_node == target_node:
            return None
        for child, _ in current_node.entries:
            if child == target_node:
                return current_node
            parent = self._find_parent(child, target_node)
            if parent:
                return parent
        return None

    def _calc_enlargement(self, mbr1, mbr2):
        combined_mbr = self._combine_mbr(mbr1, mbr2)
        return self._calc_area(combined_mbr) - self._calc_area(mbr1)

    def _calc_area(self, mbr):
        return (mbr[len(mbr) // 2] - mbr[0]) * (mbr[len(mbr) // 2 + 1] - mbr[1])

    def _combine_mbr(self, mbr1, mbr2):
        combined_mbr = []
        for i in range(len(mbr1) // 2):
            combined_mbr.append(min(mbr1[i], mbr2[i]))
            combined_mbr.append(max(mbr1[i + len(mbr1) // 2], mbr2[i + len(mbr1) // 2]))
        return tuple(combined_mbr)

    def _search(self, node, bounds, results):
        if node.bounds and not self._intersects(node.bounds, bounds):
            return

        for entry in node.entries:
            if self._intersects(entry[1], bounds):
                if node.is_leaf:
                    results.append(entry[0])
                else:
                    self._search(entry[0], bounds, results)

    def _intersects(self, mbr1, mbr2):
        return all(mbr1[i] <= mbr2[i + len(mbr1) // 2] and mbr2[i] <= mbr1[i + len(mbr1) // 2] for i in range(len(mbr1) // 2))

    def _adjust_tree(self, node):
        while node != self.root:
            parent = self._find_parent(self.root, node)
            parent.update_bounds()
            node = parent

    def to_dict(self):
        return {
            'root': self.root.to_dict(),
            'max_entries': self.max_entries
        }

    @classmethod
    def from_dict(cls, data):
        tree = cls(max_entries=data['max_entries'])
        tree.root = RTreeNode.from_dict(data['root'])
        return tree

# R树区块链中的一个区块，包含RTree的根哈希、交易、时间戳和前一个区块的哈希
class Block:
    def __init__(self, r_tree_root, r_tree, transactions, timestamp, prev_hash, max_transactions=8, extra_data=None):
        # 块头
        self.r_tree_root = r_tree_root
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.extra_data = extra_data # 可以用来存储mbr

        # 块体
        self.tree = r_tree
        self.transactions = transactions  # 只存储交易或交易的哈希
        self.max_transactions = max_transactions  # 最大交易数量


    # 计算区块的哈希值
    def calculate_hash(self):
        block_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    # 将区块转换为字典格式
    def to_dict(self):
        return {
            'r_tree_root': self.r_tree_root,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp,
            'prev_hash': self.prev_hash,
            'extra_data': self.extra_data
        }

    # 从字典格式创建区块实例
    @classmethod
    def from_dict(cls, data):
        transactions = [Transaction.from_dict(tx) for tx in data['transactions']]
        return cls(
            r_tree_root=data['r_tree_root'],
            transactions=transactions,
            timestamp=data['timestamp'],
            prev_hash=data['prev_hash'],
        )

'''
    默克尔树部分
'''
# # 默克尔树中的一个节点，包含哈希值及其左右子节点
# class MerkleNode:
#     def __init__(self, hash_value, left=None, right=None):
#         self.hash_value = hash_value
#         self.left = left
#         self.right = right

# class MerkleNode:
#     def __init__(self, hash_value=None, left=None, right=None):
#         self.hash_value = hash_value
#         self.left = left
#         self.right = right
#         self.entries = []  # 存储条目（交易）的列表
#
#     def update_hash(self):
#         left_hash = self.left.hash_value if self.left else ''
#         right_hash = self.right.hash_value if self.right else ''
#         self.hash_value = hashlib.sha256((left_hash + right_hash).encode()).hexdigest()
#
#     def is_leaf(self):
#         return not self.left and not self.right

# # 默克尔树类
# class MerkleTree:
#     def __init__(self, transactions):
#         self.transactions = transactions
#         self.root = self.build_merkle_tree([tx.calculate_hash() for tx in self.transactions])
#
#     def build_merkle_tree(self, hash_list):
#         queue = deque(hash_list)
#         while len(queue) > 1:
#             new_queue = deque()
#             while queue:
#                 left_hash = queue.popleft()
#                 right_hash = queue.popleft() if queue else left_hash
#                 parent_hash = hashlib.sha256((left_hash + right_hash).encode()).hexdigest()
#                 new_queue.append(parent_hash)
#             queue = new_queue
#
#         return MerkleNode(queue[0]) if queue else None
#
#     def get_root_hash(self):
#         return self.root.hash_value if self.root else None
#
#     def verify_transaction(self, transaction):
#         return self._verify_transaction(self.root, transaction.calculate_hash())
#
#     def _verify_transaction(self, node, tx_hash):
#         if node is None:
#             return False
#         if node.hash_value == tx_hash:
#             return True
#         return self._verify_transaction(node.left, tx_hash) or self._verify_transaction(node.right, tx_hash)

# class MerkleTree:
#     def __init__(self, max_entries=4):
#         self.root = None
#         self.max_entries = max_entries  # 每个节点中允许的最大条目数
#
#     def insert(self, transaction):
#         new_node = MerkleNode(hash_value=transaction.calculate_hash())
#         if self.root is None:
#             self.root = new_node
#         else:
#             self._insert(new_node)
#         self._adjust_tree()
#
#     def _insert(self, new_node):
#         stack = [self.root]
#         while stack:
#             node = stack.pop()
#             if node.is_leaf():
#                 if len(node.entries) < self.max_entries:
#                     node.entries.append(new_node)
#                     node.update_hash()
#                     return
#                 else:
#                     new_parent = MerkleNode(left=node, right=new_node)
#                     new_parent.update_hash()
#                     self.root = new_parent
#                     return
#             else:
#                 if node.right is None or (node.right.is_leaf() and len(node.right.entries) < self.max_entries):
#                     stack.append(node.right)
#                 else:
#                     stack.append(node.left)
#                 if node.right is None:
#                     node.right = new_node
#                 elif node.left is None:
#                     node.left = new_node
#
#     def _adjust_tree(self):
#         if not self.root:
#             return
#
#         stack = [self.root]
#         while stack:
#             node = stack.pop()
#             if node.left or node.right:
#                 if node.left:
#                     stack.append(node.left)
#                 if node.right:
#                     stack.append(node.right)
#                 node.update_hash()
#
#     def get_root_hash(self):
#         return self.root.hash_value if self.root else None
#
#     def verify_transaction(self, transaction):
#         return self._verify_transaction(self.root, transaction.calculate_hash())
#
#     def _verify_transaction(self, node, tx_hash):
#         if node is None:
#             return False
#         if node.hash_value == tx_hash:
#             return True
#         return self._verify_transaction(node.left, tx_hash) or self._verify_transaction(node.right, tx_hash)
class MerkleNode:
    def __init__(self, hash_value=None, left=None, right=None):
        self.hash_value = hash_value
        self.left = left
        self.right = right
        self.entries = []  # 存储条目（交易）的列表

    def update_hash(self):
        left_hash = self.left.hash_value if self.left else ''
        right_hash = self.right.hash_value if self.right else ''
        self.hash_value = hashlib.sha256((left_hash + right_hash).encode()).hexdigest()

    def is_leaf(self):
        return not self.left and not self.right

    def to_dict(self):
        return {
            'hash_value': self.hash_value,
            'left': self.left.to_dict() if self.left else None,
            'right': self.right.to_dict() if self.right else None,
            'entries': [entry.to_dict() for entry in self.entries] if self.entries else []
        }

    @classmethod
    def from_dict(cls, data):
        left_node = MerkleNode.from_dict(data['left']) if data['left'] else None
        right_node = MerkleNode.from_dict(data['right']) if data['right'] else None
        node = cls(hash_value=data['hash_value'], left=left_node, right=right_node, entries=data['entries'])
        return node

class MerkleTree:
    def __init__(self, max_entries=4):
        self.root = None
        self.max_entries = max_entries  # 每个节点中允许的最大条目数

    def insert(self, transaction):
        new_node = MerkleNode(hash_value=transaction.calculate_hash())
        if self.root is None:
            self.root = new_node
        else:
            stack = [self.root]
            while stack:
                node = stack.pop()
                if node.is_leaf():
                    if len(node.entries) < self.max_entries:
                        node.entries.append(new_node)
                        node.update_hash()
                        break
                    else:
                        # 创建新的父节点
                        new_parent = MerkleNode(left=node, right=new_node)
                        new_parent.update_hash()
                        self.root = new_parent
                        break
                else:
                    if node.right is None or (node.right.is_leaf() and len(node.right.entries) < self.max_entries):
                        if node.right is None:
                            node.right = new_node
                            node.update_hash()
                            break
                        else:
                            stack.append(node.right)
                    if node.left is None or (node.left.is_leaf() and len(node.left.entries) < self.max_entries):
                        if node.left is None:
                            node.left = new_node
                            node.update_hash()
                            break
                        else:
                            stack.append(node.left)

    def search(self, trans, attr, d):
        results = []
        self._search(trans, attr, results, d)
        # print(f"M树查询结果: {results}")
        return results

    def get_root_hash(self):
        return self.root.hash_value if self.root else None

    def verify_transaction(self, transaction):
        return self._verify_transaction(self.root, transaction.calculate_hash())

    def _search(self, trans, attr, results, d):
        # 遍历所有交易
        for tx in trans:
            # 如果 attribute 是字典，使用键访问
            if isinstance(tx.attribute, dict) and isinstance(attr.attribute, dict):
                if all(tx.attribute[attr_name] == attr.attribute[attr_name] for attr_name in attr.attribute.keys()):
                    results.append(tx)
            # 如果 attribute 是列表，使用索引访问
            elif isinstance(tx.attribute, list) and isinstance(attr.attribute, list):
                if all(tx.attribute[i] == attr.attribute[i] for i in range(d)):
                    results.append(tx)
            else:
                print("tx.attribute 和 attr.attribute 的结构不匹配")

    def _verify_transaction(self, node, tx_hash):
        if node is None:
            return False
        if node.hash_value == tx_hash:
            return True
        return self._verify_transaction(node.left, tx_hash) or self._verify_transaction(node.right, tx_hash)

    def to_dict(self):
        return {
            'root': self.root.to_dict() if self.root else None,
            'max_entries': self.max_entries
        }

    @classmethod
    def from_dict(cls, data):
        tree = cls(max_entries=data['max_entries'])
        tree.root = MerkleNode.from_dict(data['root'])
        return tree

# 默克尔树区块链中的一个区块
class MerkleTreeBlock:
    # def __init__(self, merkle_root, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.merkle_root = merkle_root
    def __init__(self, merkle_root, merkle_tree, transactions, timestamp, prev_hash, max_transactions=8):
        # 块头
        self.merkle_root = merkle_root  # Merkle Tree根哈希
        self.timestamp = timestamp  # 区块的时间戳
        self.prev_hash = prev_hash  # 前一个区块的哈希
        self.extra_data = None  # 用于存储额外数据

        # 块体
        self.tree = merkle_tree
        self.transactions = transactions
        self.max_transactions = max_transactions  # 最大交易数量

    # 存储额外数据
    def add_extra_data(self, data):
        self.extra_data = data

    # 计算区块的哈希值
    def calculate_hash(self):
        block_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    # 转成字典形式
    def to_dict(self):
        return {
            'merkle_root': self.merkle_root,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp,
            'prev_hash': self.prev_hash,
            'extra_data': self.extra_data
        }

    # 从字典形式恢复成默克尔树区块类
    @classmethod
    def from_dict(cls, data):
        transactions = [Transaction.from_dict(tx) for tx in data['transactions']]
        return cls(
            merkle_root=data['merkle_root'],
            transactions=transactions,
            timestamp=data['timestamp'],
            prev_hash=data['prev_hash']
        )