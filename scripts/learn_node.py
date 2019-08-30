#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from machine.learn import Learn, plot_history
import pandas as pd

if __name__ == "__main__":

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"
    box_list_path = data_path + "/box_list.csv"
    train_path = data_path + "/train"
    box_df = pd.read_csv(box_list_path, header=0)

    learn = Learn(data_path, box_df)
    learn.read_data(train_path)
    model = learn.create_model()
    history = learn.learning(model)
    plot_history(history)
