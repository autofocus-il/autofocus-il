import socket
import time
import struct


class GazepointClient:
    def __init__(self, host="127.0.0.1", port=4242):
        self.host = host  # SENSOR PARAM
        self.port = port  # SENSOR PARAM
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()
        # Client socket (Windows) <- Gaze Sensor(Server)

    def connect(self):
        self.socket.connect((self.host, self.port))  # Socket Connection
        self.send_commands()  # GazePoint Sensor Init Commands

    def send_commands(
        self,
    ):  # Commands to send to the gazepoint sensor to make it start transmission
        commands = [
            # '<SET ID="ENABLE_SEND_CURSOR" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n',
        ]
        for cmd in commands:
            self.socket.send(cmd.encode())

    def receive_data(self):  # Takes data from the Sensor and ports it to the server
        data = self.socket.recv(1024).decode()  # Buffer Size for Sensor Data
        # print(data)
        return self.validate_data(data)

    def validate_data(self, data):
        if "REC" not in data or "REC" not in data[:5]:
            x = None
            y = None
        elif "FPOGX" not in data or "FPOGY" not in data or "FPOGS" not in data:
            x = None
            y = None
        else:
            # sample data
            # <REC FPOGX="0.08254" FPOGY="0.84945" FPOGS="1178.26758" FPOGD="0.37781" FPOGID="1308" FPOGV="1" />
            data = data.split(" ")
            data = [d.split("=") for d in data][1:-1]
            data = {d[0]: float(d[1][1:-1]) for d in data[:5]}
            x = data["FPOGX"]
            y = data["FPOGY"]
        return [x, y]

    def parse(self, data):
        # sample data
        # <REC FPOGX="0.08254" FPOGY="0.84945" FPOGS="1178.26758" FPOGD="0.37781" FPOGID="1308" FPOGV="1" />
        data = data.split(" ")
        data = [d.split("=") for d in data][1:]

        # sample data
        # [['<REC', 'FPOGX', '"0.08254"', 'FPOGY', '"0.84945"', 'FPOGS', '"1178.26758"', 'FPOGD', '"0.37781"', 'FPOGID', '"1308"', 'FPOGV', '"1"', '/>']]
        # Split and return the numbers
        data = [float(d[1][1:-1]) for d in data if "FPOG" in d[0]]

        return data

    def close(self):
        self.socket.close()


if __name__ == "__main__":
    gaze = GazepointClient()

    print("Connected")
    while True:
        res = gaze.receive_data()
        print(res)
        # print(gaze.validate_data(res))
        # time.sleep(0.5)
