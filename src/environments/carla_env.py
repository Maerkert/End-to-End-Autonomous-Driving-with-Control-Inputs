import json
import logging
import time
from copy import deepcopy
from typing import Dict, Tuple

import carla
import cv2
import numpy as np
import random

from src.environments import carla_server
from src.environments.carla_server import CarlaServer
from src.environments.sensors import sensors

MIN_STEPS = 100
MAX_STEPS = 1000

SENSOR_CONFIG = json.load(open("./res/configs/sensor_setups/tesla_model_3_monocular.json", "r"))

SENSOR_DATA_QUEUE = {}

logging.basicConfig(level=logging.DEBUG)


class CarlaEnv:

    def __init__(self, client: carla.Client):
        self.client = client

        self.world = None
        self.map = None
        self.spawn_points = None
        self.blueprint_library = None
        self.actors = []

        self.sensors = []

        self.steps = 0
        self.current_observation = None

        self.reset()

    def reset(self):
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()

        self._spawn_hero()
        self.step((0, 0))

        self.steps = 0
        self.current_observation = self._get_observation()

        return self.current_observation

    def step(self, action: Tuple[float, float]):
        control = carla.VehicleControl()
        control.steer = action[0]
        control.throttle = action[1]
        self.hero.apply_control(control)
        self.world.tick() # ToDo: Move to server and allow for multi clients
        self.steps += 1

        self.current_observation = self._get_observation()
        reward = self._compute_reward()
        done = self._get_is_done()
        info = {"steps": self.steps, "done": done, "reward": reward}
        return self.current_observation, reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        data = {}
        for sensor in self.sensors:
            data[sensor.name] = sensor.get_data()
        return data

    def _compute_reward(self) -> float:
        return 0

    def _get_is_done(self):
        return self.steps >= MAX_STEPS

    def close(self):
        self._destroy_actors()

    def _spawn_hero(self):
        hero_bp = self.blueprint_library.filter(SENSOR_CONFIG["blueprint"])[0]
        self.hero = self.world.spawn_actor(hero_bp, random.choice(self.spawn_points))
        self.actors.append(self.hero)
        for sensor_name in SENSOR_CONFIG["sensors"].keys():
            logging.debug("Spawning sensor: {}".format(sensor_name))
            sensor_config = SENSOR_CONFIG["sensors"][sensor_name]
            sensor = sensors.get_sensor(sensor_config["type"], sensor_name,
                                        sensor_config, self.hero)
            self.sensors.append(sensor)
            logging.debug("Spawned sensor: {}".format(sensor_name))
        self.world.tick()

    def _destroy_actors(self):
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        for actor in self.actors:
            actor.destroy()


if __name__ == "__main__":
    server: CarlaServer = None
    try:
        server = CarlaServer()
        time.sleep(20)
        print("Getting client")
        client = server.connect_client()
        print("Connected to client")

        print("Creating environment")
        env = CarlaEnv(client)
        print("Created environment")

        for i in range(1000):
            obs, reward, done, info = env.step((0.5, 0.0))

            cv2.imshow("Image", obs["front_camera"])
            cv2.waitKey(1)
            time.sleep(0.1)

        print("Observation: ", obs)

    finally:
        if server is not None:
            server.destroy()

