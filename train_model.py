import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# =====================================================

train = pd.read_csv("train_labeled.csv")

test = pd.read_csv("test_labeled.csv")

# =====================================================

encoder = LabelEncoder()

train["distribution"] = encoder.fit_transform(
    train["distribution"]
)

test["distribution"] = encoder.transform(
    test["distribution"]
)

# =====================================================

features = [

    "n",

    "distribution",

    "mean_pairwise_dist",

    "std_coords",

    "mean_centroid_dist",

    "mean_nn_dist"

]

# =====================================================

X_train = train[features]

y_train = train["best_algorithm"]

X_test = test[features]

y_test = test["best_algorithm"]

# =====================================================

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

# =====================================================

print("\nAccuracy:\n")

print(
    accuracy_score(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# =====================================================

print("\nFeature Importance:\n")

for feature, score in zip(

    features,

    model.feature_importances_

):

    print(
        feature,
        "->",
        round(score, 4)
    )
