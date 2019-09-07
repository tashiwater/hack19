# hack19
ねじ仕分け機.画像認識でねじを区別し、pyserialでmbedを動かす

## Features

- launch
    - test.launch : ROSでの実行対象launch.これを起動すればよい
- scripts : ROSで実行するpyファイル.未更新
    - communicater_node.py : communicaterのmain文.これを実行すればcommunicaterが動く
    - ..._node.py : ...のmain文
- src : クラスおよびwindowsで実行するpyファイル
    - AR : ARマーカー関連
        - armarkermake.py ARマーカー作成用
        - calibration.py : カメラキャリブレーション時にのみ実行
        - find_parts.py : パーツを発見してimg,xmlファイルを作る
            - ar_detect.py : ARマーカーを検出し、位置を出す
            - contours.py : 二値化してboundary検出
            - locate2d.py : 画像からROIを決め、パーツの座標を計算するツール提供
        - transform.py : 画像座標系から機体座標系に変換
    - machine : 機体の動き,機械学習関連
        - communicater.py : ROSの通信仲介.未更新
        - img_generator.py : 画像かさ増し用.未更新
        - matching.py : CNNによる画像認識.未更新
            - learn.py : モデル作成用
            - Object.py : xml読み取り
        - move.py : 機体に一連の動作をさせ、ねじの仕分けを実行する
        - neji_matching.py : 対角線の長さによる分別
        - pickselector.py : ネジの先端、終端からもっとも画像内で遠い点を選ぶ
    - communicate : 通信関連
        - serial_to_mbed.py
        - slack.py
    - gui
        - box_config.py : ボックスに対応するネジ長さ決定場面のクラス
    - manager.py : windowsで動かすとき(GUIなし）の全体実行部分.未更新
    - gui.py : GUIで動かすときの実行部分。

## Requirement

- python 2 or 3 or more
- cv2
<!-- - pip install tensorflow --upgrade
- pip install keras --upgrade -->
- pip install pyserial
- pip install numpy
- pip install pandas
- pip install kivy
- pip install kivy-garden
- garden install contextmenu
- pip install docutils pygments pypiwin32 kivy.deps.sdl2
- pip install kivy.deps.glew
- pip install gspread
- pip install oauth2client

## Usage

- windows ver
    1. python manager.py

- ROS ver
    1. roslaunch test test.launch

## Installation
$ git clone https://github.com/...
