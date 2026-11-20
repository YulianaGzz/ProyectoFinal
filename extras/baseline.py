import pandas as pd

df = pd.read_csv("train_labeled.csv")

print("\nBalance de clases:\n")

counts = df["best_algorithm"].value_counts()

print(counts)

print("\nPorcentajes:\n")

print(
    round(
        counts / len(df) * 100,
        2
    )
)

print("\nClase dominante:\n")

print(counts.idxmax())