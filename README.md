

## 各文件说明

1. ***blockchanin.py*** 文件中构建了三种树结构模型，分别为MerkleRBTree、Fabric及本方案模型

2. ***global_method.py*** 绘制条形图、柱状图

3. ***in_memory_test.py*** 树构建测试代码
4. ***main.py*** 代码运行主文件，包含单区块以及链两种模型
5. ***with_external_databases_test.py*** 文件包含了分别在三种模型中进行插入、查找、并发功能的比较

## 使用环境
1. ***Windows10***或***Windows11***系统，3.11版本的Python，其余包的版本没有要求
2. 于 ***https://www.python.org/*** 安装python安装包并将其加入计算机环境变量

## 数据集
1. 使用来源于 NCBI 的 BioSample 数据库的数据集，存放在Hyperledger-fabric映射的couchdb数据库中