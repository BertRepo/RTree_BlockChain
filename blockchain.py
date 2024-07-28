import hashlib
import json
from collections import deque, defaultdict
from couchdb.http import ResourceConflict, ResourceNotFound

'''
    R树和默克尔树共用部分 交易类 和 区块链类
'''
# 交易类  R树和默克尔树对应的区块共用这个类 TODO: 后面考虑分开实现 因为其中包含边界值
class Transaction:
    def __init__(self, tx_hash, attribute, bounds):
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
        self._load_chain_from_db()

    def _load_chain_from_db(self):
        for id in self.db:
            doc = self.db[id]
            block_type = doc.get('type', None)

            # TODO: 这个判断 屎山代码 后面有空最好修改下 没有必要判断每个区块类型（同区块类型都是放在同一个数据库中的）
            if block_type == 'RTreeBlock':
                transactions = [Transaction.from_dict(tx_dict) for tx_dict in doc['transactions']]
                block = Block(r_tree_root=doc['r_tree_root'], transactions=transactions,
                              timestamp=doc['timestamp'], prev_hash=doc['prev_hash'],
                              max_transactions=self.max_transactions)
            elif block_type == 'MerkleTreeBlock':
                transactions = [Transaction.from_dict(tx_dict) for tx_dict in doc['transactions']]
                block = MerkleTreeBlock(merkle_root=doc['merkle_root'], transactions=transactions,
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
        r_tree_root = r_tree.calculate_merkle_root()  # 修正: calculate_merkle_root 方法
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
        if isinstance(block, Block):
            transactions_dict = [tx.to_dict() for tx in block.transactions]
            self.db.save({
                'type': 'RTreeBlock',
                'transactions': transactions_dict,
                'r_tree_root': block.r_tree_root,
                'timestamp': block.timestamp,
                'prev_hash': block.prev_hash,
                'extra_data': block.extra_data
            })
        elif isinstance(block, MerkleTreeBlock):
            transactions_dict = [tx.to_dict() for tx in block.transactions]
            self.db.save({
                'type': 'MerkleTreeBlock',
                'transactions': transactions_dict,
                'merkle_root': block.merkle_root,
                'timestamp': block.timestamp,
                'prev_hash': block.prev_hash,
                'extra_data': block.extra_data
            })
        else:
            print("Unsupported block type")

    def validate_block(self, prev_block, new_block):
        return new_block.prev_hash == prev_block.calculate_hash()

    def search_transaction_by_attribute(self, attribute):
        for block in self.chain:
            matching_transactions = [tx for tx in block.transactions if tx.attribute == attribute]
            if matching_transactions:
                return matching_transactions
        return None

    def _get_current_timestamp(self):
        from datetime import datetime
        return datetime.now().timestamp()


'''
    R树部分
'''

# R 树节点类
class RTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []  # 叶子节点存放数据，非叶子节点存放子节点

# R 树类
class RTree:
    def __init__(self, max_entries=4):
        self.root = RTreeNode()
        self.max_entries = max_entries # 每个节点中允许的最大条目数

    def insert(self, tx):
        entry = (tx, tx.bounds) # 将交易和边界 组成元祖
        node = self._choose_leaf(self.root, entry)  # 从根节点开始，递归地选择最合适的叶节点
        node.entries.append(entry)  # 插入新条目 到选中的叶节点
        # 节点中的条目数超过了 节点的最大容量，就对该节点 分裂处理
        if len(node.entries) > self.max_entries:
            self._split_node(node)

    def search(self, bounds):
        results = []
        print(f"Searching with bounds: {bounds}")  # 调试信息
        self._search(self.root, bounds, results)
        return results

    def calculate_merkle_root(self):
        # 假设有一个从 RTree 计算 Merkle 根的方法
        return hashlib.sha256(json.dumps("dummy_root").encode()).hexdigest()

    def _choose_leaf(self, node, entry):
        if node.is_leaf:
            return node
        else:
            best_child = min(node.entries, key=lambda child: self._calc_enlargement(child[1], entry[1]))
            return self._choose_leaf(best_child[0], entry)

    def _split_node(self, node):
        mid = len(node.entries) // 2
        new_node = RTreeNode(is_leaf=node.is_leaf)
        node.entries, new_node.entries = node.entries[:mid], node.entries[mid:]
        if node == self.root:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries.append((self.root, self._get_mbr(self.root.entries)))
            new_root.entries.append((new_node, self._get_mbr(new_node.entries)))
            self.root = new_root
        else:
            parent = self._find_parent(self.root, node)
            parent.entries.append((new_node, self._get_mbr(new_node.entries)))
            if len(parent.entries) > self.max_entries:
                self._split_node(parent)

    def _find_parent(self, current_node, target_node):
        if current_node.is_leaf or current_node == target_node:
            return None
        for child, mbr in current_node.entries:
            if child == target_node:
                return current_node
            parent = self._find_parent(child, target_node)
            if parent:
                return parent
        return None

    def _calc_enlargement(self, mbr1, mbr2):
        combined_mbr = (min(mbr1[0], mbr2[0]), min(mbr1[1], mbr2[1]), max(mbr1[2], mbr2[2]), max(mbr1[3], mbr2[3]))
        return self._calc_area(combined_mbr) - self._calc_area(mbr1)

    def _calc_area(self, mbr):
        return (mbr[2] - mbr[0]) * (mbr[3] - mbr[1])

    def _get_mbr(self, entries):
        min_x = min(entry[1][0] for entry in entries)
        min_y = min(entry[1][1] for entry in entries)
        max_x = max(entry[1][2] for entry in entries)
        max_y = max(entry[1][3] for entry in entries)
        return (min_x, min_y, max_x, max_y)

    def _search(self, node, bounds, results):
        for entry in node.entries:
            print(f"Checking entry: {entry}, bounds: {bounds}")  # 调试信息
            if self._intersects(entry[1], bounds):
                if node.is_leaf:
                    results.append(entry[0])
                else:
                    self._search(entry[0], bounds, results)

    def _intersects(self, mbr1, mbr2):
        print(f"mbr1: {mbr1}, mbr2: {mbr2}")  # 调试信息
        if not (isinstance(mbr1, tuple) and len(mbr1) == 4 and isinstance(mbr2, tuple) and len(mbr2) == 4):
            raise ValueError("MBR 应该是一个包含四个元素的元组。")

        return not (mbr2[0] > mbr1[2] or mbr2[2] < mbr1[0] or mbr2[1] > mbr1[3] or mbr2[3] < mbr1[1])


# R树区块链中的一个区块，包含RTree的根哈希、交易、时间戳和前一个区块的哈希
class Block:
    def __init__(self, r_tree_root, transactions, timestamp, prev_hash, max_transactions=8):
        self.r_tree_root = r_tree_root
        self.transactions = transactions  # 只存储交易或交易的哈希
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.extra_data = None
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

# 默克尔树中的一个节点，包含哈希值及其左右子节点
class MerkleNode:
    def __init__(self, hash_value, left=None, right=None):
        self.hash_value = hash_value
        self.left = left
        self.right = right

# 默克尔树类
class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.build_merkle_tree([tx.calculate_hash() for tx in self.transactions])

    def build_merkle_tree(self, hash_list):
        queue = deque(hash_list)
        while len(queue) > 1:
            new_queue = deque()
            while queue:
                left_hash = queue.popleft()
                right_hash = queue.popleft() if queue else left_hash
                parent_hash = hashlib.sha256((left_hash + right_hash).encode()).hexdigest()
                new_queue.append(parent_hash)
            queue = new_queue

        return MerkleNode(queue[0]) if queue else None

    def get_root_hash(self):
        return self.root.hash_value if self.root else None

    def verify_transaction(self, transaction):
        return self._verify_transaction(self.root, transaction.calculate_hash())

    def _verify_transaction(self, node, tx_hash):
        if node is None:
            return False
        if node.hash_value == tx_hash:
            return True
        return self._verify_transaction(node.left, tx_hash) or self._verify_transaction(node.right, tx_hash)

# 默克尔树区块链中的一个区块
class MerkleTreeBlock:
    # def __init__(self, merkle_root, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.merkle_root = merkle_root
    def __init__(self, merkle_root, transactions, timestamp, prev_hash, max_transactions=8):
        self.merkle_root = merkle_root  # Merkle Tree
        self.transactions = transactions
        self.timestamp = timestamp  # 区块的时间戳
        self.prev_hash = prev_hash  # 前一个区块的哈希
        self.extra_data = None  # 用于存储额外数据
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



# 继承自 Blockchain，增加了全局索引和缓存功能
class BlockchainWithIndex(Blockchain):
    def __init__(self, db, index_db, max_transactions=8):
        super().__init__(db, max_transactions)
        self.index_db = index_db
        try:
            self.global_attribute_index = defaultdict(list, self.index_db['global_attribute_index'])
        except (KeyError, ResourceNotFound):
            self.global_attribute_index = defaultdict(list)
        self.cache = {}
        self.pending_blocks = deque()


    def add_block(self, block):
        super().add_block(block)
        self.pending_blocks.append(block)

        if len(self.pending_blocks) >= 10:
            self._update_global_index_batch()

    def _update_global_index_batch(self):
        while self.pending_blocks:
            block = self.pending_blocks.popleft()
            for tx in block.transactions:
                attribute = tx.attribute
                self.global_attribute_index[attribute].append(block.calculate_hash())

        try:
            self.index_db['global_attribute_index'] = dict(self.global_attribute_index)
        except ResourceConflict:
            # 如果冲突，获取最新版本并重试
            latest_doc = self.index_db.get('global_attribute_index')
            if latest_doc:
                self.global_attribute_index = defaultdict(list, latest_doc)
                self.index_db['global_attribute_index'] = dict(self.global_attribute_index)
        except Exception as e:
            print(f"Error saving global_attribute_index to index_db: {e}")

    def search_transaction_by_attribute(self, attribute):
        if attribute in self.cache:
            return self.cache[attribute]

        block_hashes = self.global_attribute_index.get(attribute, [])
        for block_hash in block_hashes:
            block = self.get_block_by_hash(block_hash)
            if block:
                matching_transactions = list(filter(lambda tx: tx.attribute == attribute, block.transactions))
                if matching_transactions:
                    result = matching_transactions[0]
                    self.cache[attribute] = result
                    return result
        return None

    def get_block_by_hash(self, block_hash):
        for block in self.chain:
            if block.calculate_hash() == block_hash:
                return block
        return None


# B+树中的节点（BMTree中的一个节点，可以是叶节点或内部节点） 改成R树的节点
class Node:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf  # 表示节点是否是叶节点
        self.keys = []
        self.transactions = []  # 存储与键关联的交易的列表
        self.children = []  # 子节点
        self.parent = None  # 父节点

    # 向节点添加交易，并根据交易属性排序
    def add_transaction(self, transaction):
        self.transactions.append(transaction)
        self.keys.append(transaction.attribute)
        self.keys.sort()

    # 计算节点的哈希值
    def calculate_hash(self):
        return hashlib.sha256(json.dumps(self.keys, sort_keys=True).encode()).hexdigest()

# BMTree（B+树的变种），用于存储和索引交易
class BMTree:
    def __init__(self, order):
        self.root = Node()
        self.order = order

    # 向树中插入交易
    def insert(self, transaction):
        """插入到合适的叶节点"""
        leaf_node, index = self._find_leaf_node(self.root, transaction.attribute)
        self._insert_into_node(leaf_node, index, transaction)

    # 找到合适的叶节点用于插入交易数据
    def _find_leaf_node(self, node, key):
        """寻找应当插入给定键的叶节点"""
        if node.is_leaf:
            index = 0
            for k in node.keys:
                if key < k:
                    break
                index += 1
            return node, index

        index = 0
        for i in range(len(node.keys)):
            if key < node.keys[i]:
                return self._find_leaf_node(node.children[i], key)
            index = i + 1

        return self._find_leaf_node(node.children[-1], key)

    # 在节点的特定位置插入交易
    def _insert_into_node(self, node, index, transaction):
        """在给定节点的特定位置插入交易"""
        node.transactions.insert(index, transaction)
        node.keys.insert(index, transaction.attribute)
        if len(node.keys) > self.order - 1:
            self._split_node(node)

    # 当节点过满时，分裂节点
    def _split_node(self, node):
        """如果一个节点的键数量过多，分割这个节点"""
        mid = len(node.keys) // 2
        new_node = Node(is_leaf=node.is_leaf)

        parent = node.parent

        if parent is None:
            parent = Node(is_leaf=False)
            parent.keys.append(node.keys[mid])
            parent.children.append(node)
            parent.children.append(new_node)
            self.root = parent
            node.parent = parent
            new_node.parent = parent
        else:
            insert_index = 0
            for k in parent.keys:
                if node.keys[mid] < k:
                    break
                insert_index += 1

            parent.keys.insert(insert_index, node.keys[mid])
            parent.children.insert(insert_index + 1, new_node)
            new_node.parent = parent

        new_node.keys = node.keys[mid + 1:]
        new_node.transactions = node.transactions[mid + 1:]
        node.keys = node.keys[:mid]
        node.transactions = node.transactions[:mid]

        if not node.is_leaf:
            new_node.children = node.children[mid + 1:]
            node.children = node.children[:mid + 1]
            for child in new_node.children:
                child.parent = new_node

        if len(parent.keys) > self.order - 1:
            self._split_node(parent)

    # 根据属性搜索交易
    def search(self, attribute):
        return self._search(self.root, attribute)

    def _search(self, node, attribute):
        """在B+树中搜索一个给定属性的交易"""
        if node.is_leaf:
            for tx in node.transactions:
                if tx.attribute == attribute:
                    return tx
            return None
        for i in range(len(node.keys)):
            if attribute < node.keys[i]:
                return self._search(node.children[i], attribute)
        return self._search(node.children[-1], attribute)

    def calculate_merkle_root(self):
        """ 计算整个B+树的默克尔根"""
        return self._calculate_merkle_root(self.root)

    def _calculate_merkle_root(self, node):
        if node.is_leaf:
            return node.calculate_hash()
        child_hashes = [self._calculate_merkle_root(child) for child in node.children]
        return hashlib.sha256(json.dumps(child_hashes, sort_keys=True).encode()).hexdigest()

    # 获取所有交易
    def get_all_transactions(self):
        leaf_nodes = self._get_leaf_nodes(self.root)
        all_transactions = []
        for leaf in leaf_nodes:
            all_transactions.extend(leaf.transactions)
        return all_transactions

    def _get_leaf_nodes(self, node):
        if node is None:
            return []
        if node.is_leaf:
            return [node]

        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(self._get_leaf_nodes(child))

        return leaf_nodes

    def _dict_to_node(self, node_data):
        if node_data is None:
            return None
        node = Node(is_leaf=node_data['is_leaf'])
        node.keys = node_data['keys']
        node.transactions = [Transaction.from_dict(tx_data) for tx_data in node_data['transactions']]
        node.children = [self._dict_to_node(child_data) for child_data in node_data['children']]
        return node

    # 将BMTree从字典格式恢复
    @classmethod
    def from_dict(cls, data):
        order = data.get('order', 8)
        root_data = data['root']
        bm_tree = cls(order)
        bm_tree.root = bm_tree._dict_to_node(root_data)
        return bm_tree

    def _node_to_dict(self, node):
        if node is None:
            return None
        return {
            'is_leaf': node.is_leaf,
            'keys': node.keys,
            'transactions': [tx.to_dict() for tx in node.transactions],
            'children': [self._node_to_dict(child) for child in node.children]
        }

    # 将BMTree转换为字典格式
    def to_dict(self):
        # 这里是一个示例，你需要根据你的实际需求保存所有重要的字段。
        return {
            'order': self.order,
            # 如果你需要保存整个树结构，这里可能需要递归地保存所有节点。
            'root': self._node_to_dict(self.root)
        }