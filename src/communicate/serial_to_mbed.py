#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import serial
import serial.tools.list_ports


class MySerial():
    def search_com_port(self):
        coms = serial.tools.list_ports.comports()
        comlist = []
        for com in coms:
            comlist.append(com.device)
        print('Connected COM ports: ' + str(comlist))
        use_port = comlist[0]
        print('Use COM port: ' + use_port)
        return use_port

    def init_port(self):
        use_port = self.search_com_port()
        self.ser = serial.Serial(use_port)
        self.ser.baundrate = 9600
        self.ser.timeout = 5  # sec

    def write(self, data):
        self.ser.flushOutput()
        data_str = str(data)
        self.ser.write(data_str.encode(encoding='utf-8'))
        self.ser.flushOutput()
        print("send:", data)

    def read(self, r_size):
        self.ser.flushInput()
        r_data = self.ser.read_until(size=r_size)  # size分Read

        got_str = r_data.decode(encoding="utf-8")
        print('Recv: ' + got_str)
        return got_str


if __name__ == '__main__':
    myserial = MySerial()
    myserial.init_port()
    while True:
        myserial.write([77, 121, 5, int(input())])
        myserial.read(20)
