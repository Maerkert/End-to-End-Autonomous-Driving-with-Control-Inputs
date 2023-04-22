import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple
import carla
import cv2
import numpy as np
import random
from src.environments.reward_functions import reward_functions
from src.environments.sensors import sensors

MIN_STEPS = 100
MAX_STEPS = 1000

SENSOR_CONFIG = json.load(open("./res/configs/sensor_setups/tesla_model_3_monocular.json", "r"))

logging.basicConfig(level=logging.DEBUG)


class CarlaEnv:

    def __init__(self, client: carla.Client):
        self.client = client

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()

        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        self.actors = []

        self.sensors = []

        self.steps = 0
        self.current_observation = None
        self.autopilot = False

        self.reset()

    def reset(self):
        self._destroy_actors()
        self._spawn_hero()

        self.step((0, 0))

        self.steps = 0
        self.current_observation = self._get_observation()

        return self.current_observation

    def step(self, action: Tuple[float, float]):
        if not self.autopilot:
            control = carla.VehicleControl()
            control.steer = action[0]
            control.throttle = action[1]
            self.hero.apply_control(control)
        else:
            if self.hero.is_at_traffic_light():
                traffic_light = self.hero.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

        self.world.tick() # ToDo: Move to server and allow for multi clients
        self.steps += 1

        self.current_observation = self._get_observation()
        reward = self._compute_rewards(self.current_observation, 10, 0)
        done = self._get_is_done()
        info = {"steps": self.steps, "done": done, "reward": reward}
        return self.current_observation, reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        data = {}
        for sensor in self.sensors:
            data[sensor.name] = sensor.get_data()
        distance_center, distance_left_lane, distance_right_lane = self._get_distances_to_lane_center()
        data["distance_center"] = distance_center
        data["distance_left_lane"] = distance_left_lane
        data["distance_right_lane"] = distance_right_lane
        #data["in_intersection"] = self.hero.is_in_intersection()
        # ToDo: Observe if autopilot does a left turn, right turn or straight
        return data

    def _get_distances_to_lane_center(self):
        hero_location = self.hero.get_location()
        waypoint = self.map.get_waypoint(hero_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        waypoint_location = waypoint.transform.location
        waypoint_direction = waypoint.transform.get_forward_vector()
        hero_location.z = waypoint_location.z
        distance_center = _distance_point_to_line(hero_location, waypoint_location, waypoint_direction)

        waypoint_left_lane = waypoint.get_left_lane()
        distance_left_lane = None
        if waypoint_left_lane is not None:
            waypoint_location = waypoint_left_lane.transform.location
            waypoint_direction = waypoint_left_lane.transform.get_forward_vector()
            distance_left_lane = _distance_point_to_line(hero_location, waypoint_location, waypoint_direction)

        waypoint_right_lane = waypoint.get_right_lane()
        distance_right_lane = None
        if waypoint_right_lane is not None:
            waypoint_location = waypoint_right_lane.transform.location
            waypoint_direction = waypoint_right_lane.transform.get_forward_vector()
            distance_right_lane = _distance_point_to_line(hero_location, waypoint_location, waypoint_direction)

        return distance_center, distance_left_lane, distance_right_lane

    def _compute_rewards(self, observations: Dict, target_velocity: float, direction: int) -> float:
        velocity = self.hero.get_velocity()
        velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        reward_velocity = reward_functions.velocity_reward_function(velocity, target_velocity)

        # Distance to lane center and lane change logic
        if direction == -1 and observations.get("distance_left_lane", None) is not None:
            reward_line_center = reward_functions.line_center_reward_function(observations["distance_left_lane"])
        elif direction == 1 and observations.get("distance_right_lane", None) is not None:
            reward_line_center = reward_functions.line_center_reward_function(observations["distance_right_lane"])
        else:
            reward_line_center = reward_functions.line_center_reward_function(observations["distance_center"])

        reward_collision = -1 if observations["collision"] else 0

        if observations["line_invasion"] and direction == 0:
            reward_lane_invasion = -1
        else:
            reward_lane_invasion = 0

        return {"velocity": reward_velocity, "line_center": reward_line_center,
                "collision": reward_collision, "lane_invasion": reward_lane_invasion}

    def _get_is_done(self):
        return self.steps >= MAX_STEPS

    def close(self):
        self._destroy_actors()
        self.traffic_manager.shutdown()

    def _spawn_hero(self):
        hero_bp = self.blueprint_library.filter(SENSOR_CONFIG["blueprint"])[0]
        self.hero = self.world.spawn_actor(hero_bp, random.choice(self.spawn_points))
        self.actors.append(self.hero)
        for sensor_name in SENSOR_CONFIG["sensors"].keys():
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

    def set_auto_pilot(self, autopilot: bool):
        self.autopilot = autopilot
        self.hero.set_autopilot(autopilot, 8000)

    def log_observations(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        save_dict = {}
        for key, value in self.current_observation.items():
            if type(value) is np.ndarray and len(value.shape) == 2:
                value = np.expand_dims(value, axis=2)
            if type(value) is np.ndarray and len(value.shape) == 3:
                cv2.imwrite(os.path.join(path, "{}.png".format(key)), value)
            else:
                if type(value) is np.ndarray:
                    value = value.tolist()
                save_dict[key] = value
        with open(os.path.join(path, "observations.json"), "w") as f:
            json.dump(save_dict, f)


def _distance_point_to_line(point: carla.Location, line_anchor: carla.Location, line_direction: carla.Vector3D) -> float:
    point_to_anchor = line_anchor - point
    proj_onto_line = point_to_anchor.dot(line_direction)
    distance_vector = point_to_anchor - proj_onto_line * line_direction
    distance = float(distance_vector.length())
    return abs(distance)
