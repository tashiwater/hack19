#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from gspread_change import SpreadManager
import os

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    # data_path = current_path + "/../../data"
    spreadManager = SpreadManager('PartsList',current_path + '/PartsList-8533077dcf0f.json')
    spreadManager.add('15mm')
    