import pandas as pd

df = pd.read_csv("train_labeled.csv")

for n in sorted(df["n"].unique()):

    print("\n================")
    print("n =", n)
    print("================")

    print(
        df[df["n"] == n]["best_algorithm"]
        .value_counts()
    )