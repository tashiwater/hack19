import requests


class SlackBot():
    def __init__(self, token, channel):
        self.token = token
        self.channel = channel

    def send(self, comment, title, img_path):
        param = {
            'token': self.token,
            'channels': self.channel,
            'filename': "filename",
            'initial_comment': comment,
            'title': title
        }
        files = {'file': open(img_path, 'rb')}  # ファイル名を入力
        requests.post(url="https://slack.com/api/files.upload",
                      params=param, files=files)


if __name__ == "__main__":
    token = "xoxb-747399510036-750051690710-kGAhp4qvKZdynosrYrPEuGJd"  # 取得したトークン
    channel = "CMZRH5VR6"  # チャンネルID
    slack_bot = SlackBot(token, channel)
    slack_bot.send("finish parts sort", "stop comment", "contour.png")
