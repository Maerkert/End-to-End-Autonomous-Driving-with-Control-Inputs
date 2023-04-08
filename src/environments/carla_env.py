import json
from copy import deepcopy
from typing import Dict

import carla
import numpy as np
import random

from src.environments import carla_server

MIN_STEPS = 100
MAX_STEPS = 1000

WINDOW_SIZE_X = 800
WINDOW_SIZE_Y = 600


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
        self.sensor_data_queue = {}
        self.current_observation = None

        self.reset()

    def _spawn_hero(self):
        hero_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        self.hero = self.world.spawn_actor(hero_bp, random.choice(self.spawn_points))
        self.actors.append(self.hero)

        sensor_config = json.load(open("", "r"))

        for sensor in sensor_config["sensors"]:
            sensor_bp = self.blueprint_library.find(sensor["type"])
            sensor_bp.set_attribute("role_name", sensor["name"])
            for attribute, value in sensor["attributes"].items():
                sensor_bp.set_attribute(attribute, value)
            sensor_transform = carla.Transform(carla.Location(x=sensor["x"],
                                                              y=sensor["y"],
                                                              z=sensor["z"]),
                                               carla.Rotation(pitch=sensor["pitch"],
                                                              yaw=sensor["yaw"],
                                                              roll=sensor["roll"]))
            sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.hero)
            if sensor.type_id.startswith("sensor.camera"):
                sensor.listen(lambda data: CarlaEnv._process_data(data, sensor))
            self.sensors.append(sensor)

    def _destroy_actors(self):
        for actor in self.actors:
            for sensor in actor.get_children():
                sensor.destroy()
            actor.destroy()

    def reset(self):
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()

        self._destroy_actors()

        self._spawn_hero()

        self.steps = 0
        self.current_observation = self._get_observation()

        return self.current_observation

    def step(self, action):
        control = carla.VehicleControl()
        control.steer = action[0]
        control.throttle = action[1]
        self.hero.apply_control(control)
        self.steps += 1

        self.current_observation = self._get_observation()
        reward = self._compute_reward()
        done = self._get_is_done()
        info = {"steps": self.steps, "done": done, "reward": reward}
        return self.current_observation, reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        return deepcopy(self.sensor_data_queue)

    def _compute_reward(self) -> float:
        return 0

    def _get_is_done(self):
        return self.steps >= MAX_STEPS

    def close(self):
        self._destroy_actors()

    @staticmethod
    def _process_data(data, sensor):
        if sensor.type_id.startswith("sensor.camera"):
            data.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (WINDOW_SIZE_Y, WINDOW_SIZE_X, 4))
            array = array[:, :, :3]
            return array
