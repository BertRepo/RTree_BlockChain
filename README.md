

## 各文件说明
1. ***blockchanin.py*** 文件中构建了两种树结构模型，分别为MerkleTree、RTree、Fabric（未确认）

2. ***global_method.py*** 绘制条形图、柱状图

3. ***in_memory_test.py*** 在内存中测试 MerkleTree和RTree 两种树结构本身的 构建、查询 的时耗

4. ***main.py*** 代码运行主文件，包含单区块以及链两种模型

5. ***with_external_databases_test.py*** 文件，是链接CouchDB后，模拟链下状态分别在三种模型中进行插入、查找、并发功能的比较

## 使用环境
1. ***任何***系统环境均可，3.11版本的Python，其余包的版本没有要求
2. 必须提前安装好CouchDB，账号密码分别是 admin 123456，另外不用创建任何数据库
3. 于 ***https://www.python.org/*** 安装python安装包并将其加入计算机环境变量

## 数据集
1. 使用来自医院合作方的一万条结构化数据，先心病数据集。其中，一共29个属性，连续值属性有多少个，离散值属性有多少个，无空间范围属性，有几个时间范围属性（想办法更完善的介绍一下数据集）。

| 数据名 | 缩写 | 类型及长度 | 是否允许为空  | 数据示例 |
|-------|-----|----------|---------------| ------- |
|筛查中心名称|screening_center|VARCHAR(60)|是|浙江省先天性心脏病筛查中心|
|筛查医院名称|screening_hos_name|VARCHAR(60)|是|宁波市医疗中心李惠利医院|
|母亲身份证号|mother_cert_id|VARCHAR(255)|是|916E05A...|
|婴儿性别|baby_sex|VARCHAR(20)|是|2|
|婴儿体重|baby_weight|VARCHAR(20)|是|3600|
|婴儿生日|baby_birthday|DATETIME(19)|是|YYYY-MM-DD HH:MM:SS|
|婴儿数量|baby_num|VARCHAR(20)|是|1|
|出生类型|born_type|VARCHAR(20)|是|2|
|妊娠周|pregnant_week|VARCHAR(20)|是|40|
|筛查注册时间|screen_register_time|DATETIME(19)|是|YYYY-MM-DD HH:MM:SS|
|上肢动脉血氧饱和度|upper_limbs_spo2|VARCHAR(20)|是|96|
|下肢动脉血氧饱和度|lower_limbs_spo2|VARCHAR(20)|是|96|
|医生名称|doctor_name|VARCHAR(20)|是|张**|
|超声分析结果|voice_analysis_result|VARCHAR(255)|是|null|
|结果描述|result_desc|VARCHAR(255)|是|null|
|病人类型|patient_type|VARCHAR(20)|是|1|
|报告时间|report_add_date|DATETIME(19)|是|YYYY-MM-DD HH:MM:SS|
|医院编号|dia_hospital_id|VARCHAR(60)|是|dc70e279...|
|医院名称|dia_hospital_name|VARCHAR(60)|是|***********医院|
|医生姓名|dia_doctor_name|VARCHAR(20)|是|张**|
|类型|dia_type|VARCHAR(20)|是|2|
|日期|dia_date|DATETIME(19)|是|YYYY-MM-DD HH:MM:SS|
|结果|dia_result|VARCHAR(20)|是|1|
|结果描述|dia_result_desc|VARCHAR(255)|是|卵圆孔未闭 ((（x≤3）2mm));|
|建议类型|dia_advice_type|VARCHAR(20)|是|1|
|复查时间|review_date|DATETIME(19)|是|YYYY-MM-DD HH:MM:SS|
|建议|dia_advice|VARCHAR(255)|是|建议6个月内在诊断机构进行复查\n注意事项：平时需注意是否存在呼吸急促、紫绀、多汗、反复肺炎、体重不增加等情况，如果有这些情况，应及时至医院接受检查。\n|
|超声分析状态|voice_analysis_status|VARCHAR(255)|是|null|
|超声结果分析|voice_result_desc|VARCHAR(255)|是|null|

[//]: # (2. 使用来源于 NCBI 的 BioSample 数据库的数据集，存放在Hyperledger-fabric映射的couchdb数据库中)

## RTree实现思路

