import pandas as pd

df = pd.read_csv("train_labeled.csv")

print("\nGANADOR POR DISTRIBUCION\n")

for distribution in df["distribution"].unique():

    print("\n====================")
    print(distribution)
    print("====================")

    subset = df[
        df["distribution"] == distribution
    ]

    print(
        subset["best_algorithm"]
        .value_counts()
    )