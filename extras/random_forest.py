import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==========================================

train = pd.read_csv("train_labeled.csv")

test = pd.read_csv("test_labeled.csv")

# ==========================================

encoder = LabelEncoder()

train["distribution"] = encoder.fit_transform(
    train["distribution"]
)

test["distribution"] = encoder.transform(
    test["distribution"]
)

# ==========================================

features = [

    "n",

    "distribution",

    "mean_pairwise_dist",

    "std_coords",

    "mean_centroid_dist",

    "mean_nn_dist"

]

# ==========================================

X_train = train[features]

y_train = train["best_algorithm"]

X_test = test[features]

y_test = test["best_algorithm"]

# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

# ==========================================

print("\nACCURACY\n")

print(
    accuracy_score(
        y_test,
        pred
    )
)

print("\nCONFUSION MATRIX\n")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_test,
        pred
    )
)