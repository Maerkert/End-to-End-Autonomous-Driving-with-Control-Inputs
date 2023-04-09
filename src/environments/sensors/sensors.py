import logging
from abc import abstractmethod
import carla
import numpy as np


def get_sensor(sensor_type: str, name: str, attributes: dict, parent_actor: carla.Actor):
	if sensor_type == "sensor.camera.rgb":
		return Camera(name, attributes, parent_actor)
	elif sensor_type == "sensor.other.imu":
		return IMU(name, attributes, parent_actor)
	elif sensor_type == "sensor.other.gnss":
		return GNSS(name, attributes, parent_actor)
	else:
		raise ValueError("Invalid sensor type: {}".format(sensor_type))


class Sensor:

	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		self.name = name
		self.attributes = attributes
		self.parent = parent_actor
		self.data_buffer = None
		self.sensor = self._spawn_sensor()

	def _spawn_sensor(self):
		logging.debug("Spawning sensor: {}".format(self.name))
		world = self.parent.get_world()
		blueprint_library = world.get_blueprint_library()

		sensor_type = self.attributes["type"]
		sensor_bp = blueprint_library.find(sensor_type)

		sensor_bp.set_attribute("role_name", self.name)
		for attribute, value in self.attributes.get("attributes", {}).items():
			sensor_bp.set_attribute(attribute, str(value))

		sensor_location = carla.Location(x=self.attributes["x"],
										 y=self.attributes["y"],
										 z=self.attributes["z"])
		sensor_rotation = carla.Rotation(pitch=self.attributes["pitch"],
										 yaw=self.attributes["yaw"],
										 roll=self.attributes["roll"])
		sensor_transform = carla.Transform(sensor_location, sensor_rotation)

		sensor = world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.parent)

		sensor.listen(lambda data: self.sensor_callback(data))

		return sensor

	def sensor_callback(self, data: carla.SensorData):
		self.data_buffer = data

	@abstractmethod
	def get_data(self) -> np.array:
		raise NotImplementedError("This method must be implemented by the child class")

	def destroy(self):
		self.sensor.destroy()


class Camera(Sensor):

	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self):
		if self.data_buffer is not None:
			array = np.frombuffer(self.data_buffer.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (self.data_buffer.height, self.data_buffer.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			return array
		return None


class IMU(Sensor):

	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self):
		if self.data_buffer is not None:
			return self.data_buffer
		return None


class GNSS(Sensor):

	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self):
		if self.data_buffer is not None:
			return self.data_buffer
		return None
