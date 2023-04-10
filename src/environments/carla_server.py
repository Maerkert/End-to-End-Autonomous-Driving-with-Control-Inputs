import os
import random
import subprocess
import psutil
import carla

OS = "WINDOWS"  # "LINUX" or "WINDOWS
CONNECTION_RETRIES = 5


def port_is_used(port):
    return port in [conn.laddr.port for conn in psutil.net_connections()]


class CarlaServer:

    def __init__(self):
        self.server_port = None
        self.traffic_manager_port = None

        self.process = None

        self.init_server()

    def init_server(self):
        if OS == "LINUX":
            server_command = ["{}/CarlaUE4.sh".format("/opt/carla-simulator")]
        elif OS == "WINDOWS":
            server_command = ["{}/CarlaUE4.exe".format(os.environ["CARLA_ROOT"])]
        else:
            raise ValueError("OS must be either LINUX or WINDOWS")

        server_command = server_command + ["-RenderOffScreen"]

        self.server_port = random.randint(15000, 32000)
        while port_is_used(self.server_port) or port_is_used(self.server_port + 1):
            self.server_port += 1

        self.traffic_manager_port = self.server_port + 10
        while port_is_used(self.traffic_manager_port):
            self.traffic_manager_port += 10

        server_command += ["--carla-rpc-port={}".format(self.server_port)]

        server_command = " ".join(map(str, server_command))
        print("Starting server with command: " + server_command)
        self.process = subprocess.Popen(
            server_command,
            shell=True,
            preexec_fn=None if OS == "WINDOWS" else os.setsid,
            stdout=open(os.devnull, "w"))
        print("Started server with Process ID: " + str(self.process.pid))

    def connect_client(self) -> carla.Client:
        for i in range(CONNECTION_RETRIES):
            try:
                print("Trying to connect to the server on port " + str(self.server_port))
                client = carla.Client("localhost", self.server_port)
                client.set_timeout(30.0)
                return client
            except Exception as e:
                print("Failed to connect to the server. Retrying...")
        raise Exception("Failed to connect to the server")

    def destroy(self):
        process = psutil.Process(self.process.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
