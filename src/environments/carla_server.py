import os
import random
import signal
import subprocess
import time
from typing import Dict, Any, Tuple

import cv2
import psutil
import logging
import carla

OS = "WINDOWS"  # "LINUX" or "WINDOWS

SHOW_WINDOW = True
WINDOW_SIZE_X = 800
WINDOW_SIZE_Y = 600

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

        if SHOW_WINDOW:
            server_command = server_command + ["-windowed", f"-ResX={WINDOW_SIZE_X}", f"-ResY={WINDOW_SIZE_Y}"]
        else:
            server_command = server_command + ["-RenderOffScreen"]

        self.server_port = random.randint(15000, 32000)
        while port_is_used(self.server_port) or port_is_used(self.server_port + 1):
            self.server_port += 1

        self.traffic_manager_port = self.server_port + 10
        while port_is_used(self.traffic_manager_port):
            self.traffic_manager_port += 10

        server_command += ["--carla-rpc-port={}".format(self.server_port)]

        # ToDo Load selected Map
        server_command += ["--map=Town01"]

        server_command = " ".join(map(str, server_command))
        print("Starting server with command: " + server_command)
        self.process = subprocess.Popen(
            server_command,
            shell=True,
            preexec_fn=None if OS == "WINDOWS" else os.setsid,
            stdout=open(os.devnull, "w"))
        print("Started server with Process ID: " + str(self.process.pid))

        # Spawning main client
        #self.main_client = None
        #client = carla.Client("localhost", self.server_port)
        #client.set_timeout(10.0)

        #self.world = client.get_world()
        #self.map = self.world.get_map()

        #weather = getattr(carla.WeatherParameters, environment_config["domain_permutations"]["weather"])
        #self.world.set_weather(weather)

    def connect_client(self) -> carla.Client:
        for i in range(CONNECTION_RETRIES):
            try:
                print("Trying to connect to the server on port " + str(self.server_port))
                client = carla.Client("localhost", self.server_port)
                client.set_timeout(10.0)
                return client
            except Exception as e:
                print("Failed to connect to the server. Retrying...")
        raise Exception("Failed to connect to the server")

    def destroy(self):
        process = psutil.Process(self.process.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()


if __name__ == "__main__":
    server: CarlaServer = None
    try:
        server = CarlaServer()
        time.sleep(20)
        print("Getting client")
        client = server.connect_client()
        print("Connected to client")
        time.sleep(120)
    finally:
        if server is not None:
            server.destroy()
