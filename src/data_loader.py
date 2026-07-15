import pandas as pd

def load_data():
    train_df = pd.read_csv("data/Training.csv")
    test_df = pd.read_csv("data/Testing.csv")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data()

    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)

    print("\nFirst 5 rows:")
    print(train_df.head())

    print("\nColumn names (last 5, includes target):")
    print(train_df.columns[-5:].tolist())

    print("\nNumber of unique diseases:", train_df["prognosis"].nunique())
    print("\nDisease list:")
    print(train_df["prognosis"].unique())