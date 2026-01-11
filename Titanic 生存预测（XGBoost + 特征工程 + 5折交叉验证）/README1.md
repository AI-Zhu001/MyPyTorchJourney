# Titanic 生存预测（XGBoost + 特征工程 + 5折交叉验证）

这是我在学习机器学习/数据处理过程中做的一个小练习：使用 Titanic 数据集完成数据清洗、特征工程、训练 XGBoost 分类模型，并用 Stratified 5-Fold Cross Validation（分层五折交叉验证）评估效果，最后生成提交文件 `submission_xgb.csv` 作为结果留存。
## 环境与依赖

- Python 3.x
- 依赖：
  - pandas
  - scikit-learn
  - xgboost

安装依赖：

```bash
pip install -r requirements.txt

数据说明
需要两份 CSV 文件（同目录即可）：

train.csv：训练集（包含 Survived 标签）
test.csv：测试集（不包含 Survived 标签）

运行方式
在当前目录运行：
python Titanic-xgb-cv.py


运行后会输出：
5 折准确率数组（每一折的 accuracy）

平均准确率

生成提交文件：submission_xgb.csv

做了什么
1）基础预处理

缺失值处理：

Age / Fare 使用中位数填充

Embarked 使用众数填充

类别编码：

Sex：male=0, female=1

Embarked：S=0, C=1, Q=2

2）特征工程（稳定组合）

Title：从 Name 提取称谓（Mr/Miss/Mrs/Master/Rare/Unknown）

FamilySize：SibSp + Parch + 1

IsAlone：是否独自一人（FamilySize==1）

FarePerPerson：人均票价（Fare / FamilySize）

3）模型与评估

模型：XGBoost XGBClassifier

评估：Stratified 5 折交叉验证（保证每折类别比例一致）

最终：用全量训练集训练一次，并对测试集预测，输出 submission_xgb.csv

评估结果（示例）

我本次运行得到的 5 折准确率：

0.8547486

0.85955056

0.8258427

0.83707865

0.83146067

平均准确率：

0.8417362375243236

（不同环境/版本可能会有轻微波动）

输出文件

submission_xgb.csv

