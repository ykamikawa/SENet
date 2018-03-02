# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def even_separate(df, category_df, test_size=0.2):
    train_df = pd.DataFrame(columns=["file_name", "category_id"])
    val_df = pd.DataFrame(columns=["file_name", "category_id"])
    for i in range(len(category_df)+1):
        train, val = train_test_split(df[df.category_id == i], test_size=test_size)
        train_df = train_df.append(train)
        val_df = val_df.append(val)
    return train_df.sample(frac=1).reset_index(drop=True), val_df.sample(frac=1).reset_index(drop=True)
