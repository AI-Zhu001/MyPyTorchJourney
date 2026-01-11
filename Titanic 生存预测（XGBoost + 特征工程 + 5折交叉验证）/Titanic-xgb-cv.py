import re
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗 + 特征工程：
    - 缺失值：Age/Fare 中位数、Embarked 众数
    - 编码：Sex、Embarked
    - 特征：Title、FamilySize、IsAlone、FarePerPerson
    """
    df = df.copy()

    # 缺失值处理
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # 类别编码
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # 家庭相关特征
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title：从 Name 提取称谓
    def get_title(name: str) -> str:
        m = re.search(r",\s*([^.]*)\.", str(name))
        return m.group(1).strip() if m else "Unknown"

    df["Title"] = df["Name"].apply(get_title)

    # 合并稀有称谓（避免类别太碎）
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # 映射为数值
    title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4, "Unknown": 5}
    df["Title"] = df["Title"].map(title_map).fillna(5).astype(int)

    # 人均票价
    df["FarePerPerson"] = (df["Fare"] / df["FamilySize"]).fillna(df["Fare"].median())

    return df


def build_model() -> XGBClassifier:
    """
    你已经验证过效果较好的参数（偏保守，防过拟合）
    """
    return XGBClassifier(
        n_estimators=900,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.2,
        reg_alpha=0.0,
        random_state=42,
        eval_metric="logloss",
    )


def main():
    # 1) 读取数据（默认同目录）
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # 2) 特征工程
    train_data = add_features(train_data)
    test_data = add_features(test_data)

    features = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "FamilySize", "IsAlone", "Title", "FarePerPerson"
    ]

    X = train_data[features]
    y = train_data["Survived"]
    X_test = test_data[features]

    # 3) 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_for_cv = build_model()

    scores = cross_val_score(model_for_cv, X, y, cv=cv, scoring="accuracy")
    print("5折准确率:", scores)
    print("平均准确率:", scores.mean())

    # 4) 全量训练 + 预测测试集
    final_model = build_model()
    final_model.fit(X, y)

    test_pred = final_model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": test_pred
    })

    out_path = "submission_xgb.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"提交文件已生成：{out_path}")


if __name__ == "__main__":
    main()
