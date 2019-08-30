#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import cv2
from machine.matching import Match
class Move():
    def __init__(self, data_path, testdata_path, find_parts, box_df, tf2machine):
        self.find_parts = find_parts
        self.box_df = box_df
        self.match = Match(data_path, testdata_path, box_df)
        self.tf2machine = tf2machine
    
    def run(self):
        if self.find_parts.get_testdata() is False:
            return False
        cv2.waitKey(500)
        if self.match.get_test_data() is False:
            return False
        self.match.predict()
        use_obj = self.match.df.index[0]
        print("target_img", use_obj)
        x = self.match.objs[use_obj].x_m
        y = self.match.objs[use_obj].y_m
        z = self.match.max_index[use_obj]
        print("img posi", x, y)
        print("class", z)

        to_mbed_x, to_mbed_y = self.tf2machine.get_xy_mm(x, y)
        id = int(z)
        box = self.box_df.iloc[id]
        to_mbed_z = box.box_x
        to_mbed_w = box.box_y
        print("pub to mbed ", to_mbed_x, to_mbed_y, to_mbed_z, to_mbed_w)
        return True
        

if __name__ == "__main__":
    # ctr-Cで消せるようにする
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    #path
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../data"
    camera_info_path = data_path + "/camera_info"
    testdata_path = data_path + "/test"
    tempsave_path = data_path + "/temp"
    box_list_path = data_path + "/box_list.csv"
    checkpoint_path = data_path+"/checkpoint/cp.ckpt"
    
    #ARマーカー検出用クラス
    camera_matrix = np.loadtxt(camera_info_path + "/cameraMatrix.csv",
                               delimiter=",")
    distCoeffs = np.loadtxt(
        camera_info_path + "/distCoeffs.csv", delimiter=",")
    ar_marker_size = 0.02  # ARマーカー一辺[m]
    ar_detect = ARDetect(ar_marker_size, aruco.DICT_4X4_50,
                         camera_matrix, distCoeffs)

    #対象領域決定用クラス
    x_scale_m = 0.2 # ARマーカの間隔（端から端）
    y_scale_m = 0.159
    remove_x_m = 0.017 #ROIと認識しない領域
    remove_y_m = 0.005
    locate_2d = Locate2d(ar_detect, x_scale_m, y_scale_m,
                         remove_x_m, remove_y_m)

    #カメラから画像を取得する関数
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    def get_frame():
        for i in range(5):
            _, frame = cap.read()
        frame = cv2.imread(tempsave_path + "/raw.png")
        return frame
    #パーツ検出
    find_parts = FindParts(testdata_path, tempsave_path,
                           locate_2d, get_frame)

    # 画像認識
    box_df = pd.read_csv(box_list_path, header=0)
    match = Match(data_path, testdata_path, box_df)  
    
    #座標変換用
    tf2machine = Transfrom2Machine(offset_x_mm = 194, offset_y_mm= 198)

    while True:
        cv2.waitKey(500)
        if find_parts.get_testdata() is False:
            continue
        cv2.waitKey(500)
        if match.get_test_data() is False:
            continue
        match.predict()
        use_obj = match.df.index[0]
        print("target_img", use_obj)
        x = match.objs[use_obj].x_m
        y = match.objs[use_obj].y_m
        z = match.max_index[use_obj]
        print("img posi", x, y)
        print("class", z)

        to_mbed_x, to_mbed_y = tf2machine.get_xy_mm(
            match.objs[use_obj].x_m, match.objs[use_obj].y_m)
        id = int(z)
        box = box_df.iloc[id]
        to_mbed_z = box.box_x
        to_mbed_w = box.box_y
        print("pub to mbed ", to_mbed_x, to_mbed_y, to_mbed_z, to_mbed_w)

        cv2.waitKey(0)
