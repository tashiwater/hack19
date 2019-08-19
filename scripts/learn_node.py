#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from machine.learn import Learn

if __name__ == "__main__":

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"

    learn = Learn(data_path)
    learn.read_data()
    model = learn.create_model()
    history = learn.learning(model)
    plot_history(history)
