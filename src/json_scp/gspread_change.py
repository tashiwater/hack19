#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class SpreadManager:
    def __init__(self, file_name, file_path):
        scope = ['https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(file_path, scope)
        gc = gspread.authorize(credentials)
        self.wks = gc.open(file_name).sheet1

        #wks.update_acell('A1', 'Hello World!')
        #print(wks.acell('A1'))

    def write_by_cellname(self, cell_name, input_data):
        self.wks.update_acell(cell_name,input_data)
        #print(self.wks.acell(cell_name))

    def write(self, row, col, input_data):
        self.wks.update_cell(row, col,input_data)
        print(self.wks.cell(row, col))

    def read(self, cell_name):
        return int(self.wks.acell(cell_name).value)

    def add(self, data_name):
        cell = self.wks.find(data_name)
        #print("Found something at (%s,%s)" % (cell.row, cell.col))
        cell_writen = self.wks.cell(cell.row,cell.col+1)
        #print(cell_writen.value)
        self.write(cell_writen.row, cell_writen.col, str(int(cell_writen.value)+1))

if __name__ == "__main__":
    spreadManager = SpreadManager('PartsList', 'PartsList-8533077dcf0f.json')
    spreadManager.add('5mm')
    