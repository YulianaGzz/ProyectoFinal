import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ===========================

train = pd.read_csv(
    "train_labeled.csv"
)

test = pd.read_csv(
    "test_labeled.csv"
)

# ===========================

encoder = LabelEncoder()

train["distribution"] = encoder.fit_transform(
    train["distribution"]
)

test["distribution"] = encoder.transform(
    test["distribution"]
)

# ===========================

features = [

    "n",

    "distribution",

    "mean_pairwise_dist",

    "std_coords",

    "mean_centroid_dist",

    "mean_nn_dist"

]

X_train = train[features]

y_train = train["best_algorithm"]

X_test = test[features]

# ===========================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

preds = model.predict(
    X_test
)

# ===========================

regrets = []

for i in range(len(test)):

    row = test.iloc[i]

    pred_algo = preds[i]

    predicted_cost = row[
        f"cost_{pred_algo}"
    ]

    optimal_cost = min(

        row["cost_2opt_hill_climbing"],

        row["cost_sa_2opt"],

        row["cost_mst_2opt"]

    )

    regrets.append(
        predicted_cost
        -
        optimal_cost
    )

print()

print(
    "Average Regret:"
)

print(
    sum(regrets)
    /
    len(regrets)
)