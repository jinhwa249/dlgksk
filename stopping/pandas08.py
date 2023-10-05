# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: csv
# import numpy as np
import pandas as sns
from pandas04 import pdprint


def main():
    df= sns.load_dataset("titanic")
    print(df.info())
    print(df.head())
    
    print(df["who"].value_counts())
    print(df["pclass"].value_counts())
    print(df["pclass"].astype("int8").head())
    df["pclass"] = df["pclass"].astype("int8")
    print(df.info())
    
    print(df.sort_values("age").head())
    print(df.sort_vales("age", ascending=False).head())
    
    cond = df["age"] >= 70
    print(df.loc[cond][["age", "who", "alive"]].head())
    
if __name__ == "__main__":
    main()   