import os
import random
import subprocess
import time

import psutil
import carla

OS = "WINDOWS"  # "LINUX" or "WINDOWS
CONNECTION_RETRIES = 5
MAP = "Town01"


def port_is_used(port):
    return port in [conn.laddr.port for conn in psutil.net_connections()]


class CarlaServer:

    def __init__(self, port: int = None):
        self.server_port = None if port is None else port

        self.process = None

        self.init_server()

    def init_server(self):
        if OS == "LINUX":
            server_command = ["{}/CarlaUE4.sh".format("/opt/carla-simulator")]
        elif OS == "WINDOWS":
            server_command = ["{}/CarlaUE4.exe".format(os.environ["CARLA_ROOT"])]
        else:
            raise ValueError("OS must be either LINUX or WINDOWS")

        server_command = server_command + ["-RenderOffScreen", "-fps=10"]

        if self.server_port is None:
            self.server_port = random.randint(15000, 32000)
            while port_is_used(self.server_port) or port_is_used(self.server_port + 1):
                self.server_port += 1

        server_command += ["--carla-rpc-port={}".format(self.server_port)]

        server_command = " ".join(map(str, server_command))
        print("Starting server with command: " + server_command)
        self.process = subprocess.Popen(
            server_command,
            shell=True,
            preexec_fn=None if OS == "WINDOWS" else os.setsid,
            stdout=open(os.devnull, "w"))
        print("Started server with Process ID: " + str(self.process.pid))

        time.sleep(10)
        print("Connecting main client")
        self.main_client = self.connect_client(self.server_port)

        world = self.main_client.get_world()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)
        print("Settings applied")

        #print("Loading Map")
        #self.main_client.load_world(MAP)

    @staticmethod
    def connect_client(server_port: int) -> carla.Client:
        for i in range(CONNECTION_RETRIES):
            try:
                print("Trying to connect to the server on port " + str(server_port))
                client = carla.Client("localhost", server_port)
                client.set_timeout(30.0)
                client.get_world().get_map()
                return client
            except Exception as e:
                print("Failed to connect to the server. Retrying...")
        raise Exception("Failed to connect to the server")

    def destroy(self):
        process = psutil.Process(self.process.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()


if __name__ == '__main__':
    server = None
    try:
        server = CarlaServer(2000)
        time.sleep(999999)
    finally:
        if server is not None:
            server.destroy()
