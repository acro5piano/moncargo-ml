import pandas as pd

df = pd.read_csv(
    "./data_for_machine_learning_2023-11-05T01_13_45.27414546Z.csv", delimiter=","
)

df = df[["website_type", "tracking_type", "tracking_number"]]

df["Y"] = df.apply(
    lambda row: row["website_type"] + "__" + row["tracking_type"], axis=1
)

df["Y"] = pd.factorize(df["Y"].astype("category"))[0]

for i in range(0, 20):
    df["X" + str(i)] = df.apply(
        lambda row: 0
        if len(row["tracking_number"]) <= i
        else ord(row["tracking_number"][i]),
        axis=1,
    )

df.to_csv("./prepared.csv", index=False)
