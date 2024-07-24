

## 各文件说明

1. ***blockchanin.py*** 文件中构建了两种树结构模型，分别为MerkleTree、RTree、Fabric（未确认）

2. ***global_method.py*** 绘制条形图、柱状图

3. ***in_memory_test.py*** 树构建内存中测试代码

4. ***main.py*** 代码运行主文件，包含单区块以及链两种模型

5. ***with_external_databases_test.py*** 文件，是链接CouchDB后，模拟链下状态分别在三种模型中进行插入、查找、并发功能的比较

## 使用环境
1. ***任何***系统环境均可，3.11版本的Python，其余包的版本没有要求
2. 必须提前安装好CouchDB，账号密码分别是 admin 123456，另外不用创建任何数据库
3. 于 ***https://www.python.org/*** 安装python安装包并将其加入计算机环境变量

## 数据集
1. 生成随机数来模拟数据集，后面酌情修改

[//]: # (2. 使用来源于 NCBI 的 BioSample 数据库的数据集，存放在Hyperledger-fabric映射的couchdb数据库中)