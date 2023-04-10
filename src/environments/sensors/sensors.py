import logging
import threading
from abc import abstractmethod
import carla
import cv2
import numpy as np
from carla import ColorConverter


def get_sensor(sensor_type: str, name: str, attributes: dict, parent_actor: carla.Actor):
	"""
	Create a sensor object based on the sensor type and the attributes.
	:param sensor_type: The type of the sensor as string.
	:param name: Name for the sensor
	:param attributes: Attributes of the sensor as dictionary, following the CARLA sensor attributes.
	:param parent_actor: The actor to which the sensor is attached.
	:return: carla.Sensor object
	"""
	if sensor_type == "sensor.camera.rgb":
		return Camera(name, attributes, parent_actor)
	elif sensor_type == "sensor.other.imu":
		return IMU(name, attributes, parent_actor)
	elif sensor_type == "sensor.other.gnss":
		return GNSS(name, attributes, parent_actor)
	elif sensor_type == "sensor.camera.semantic_segmentation":
		return SegmentationCamera(name, attributes, parent_actor)
	elif sensor_type == "sensor.camera.depth":
		return DepthCamera(name, attributes, parent_actor)
	else:
		raise ValueError("Invalid sensor type: {}".format(sensor_type))


class Sensor:
	"""
	Base class for all sensors.
	"""
	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		self.name = name
		self.attributes = attributes
		self.parent = parent_actor

		self.data_buffer = None
		self.data_thread_lock = threading.Lock()

		self.sensor = self._spawn_sensor()

	def _spawn_sensor(self):
		logging.debug("Spawning sensor: {}".format(self.name))

		world = self.parent.get_world()
		blueprint_library = world.get_blueprint_library()

		sensor_bp = blueprint_library.find(self.attributes["type"])
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
		self.data_thread_lock.acquire()
		self.data_buffer = data
		self.data_thread_lock.release()

	@abstractmethod
	def get_data(self) -> np.array:
		raise NotImplementedError("This method must be implemented by the child class")

	def destroy(self):
		self.sensor.destroy()


class Camera(Sensor):
	"""
	RGB Camera sensor.
	"""
	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self):
		if self.data_buffer is not None:
			self.data_thread_lock.acquire()
			self.data_buffer.convert(ColorConverter.Raw)
			array = np.frombuffer(self.data_buffer.raw_data, dtype=np.dtype("uint8")).copy()
			self.data_thread_lock.release()

			array = np.reshape(array, (self.data_buffer.height, self.data_buffer.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]

			return array
		return None


class SegmentationCamera(Sensor):
	"""
	Segmentation Camera sensor using the CityScapes palette.
	"""
	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self) -> np.array:
		if self.data_buffer is not None:
			self.data_thread_lock.acquire()
			self.data_buffer.convert(ColorConverter.CityScapesPalette)
			array = np.frombuffer(self.data_buffer.raw_data, dtype=np.dtype("uint8")).copy()
			self.data_thread_lock.release()

			array = np.reshape(array, (self.data_buffer.height, self.data_buffer.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]

			return array
		return None


class DepthCamera(Sensor):
	"""
	Depth Camera sensor. Returns a 1-channel array with the depth in meters (float16).
	"""
	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self) -> np.array:
		if self.data_buffer is not None:
			self.data_thread_lock.acquire()
			self.data_buffer.convert(ColorConverter.LogarithmicDepth)
			array = np.frombuffer(self.data_buffer.raw_data, dtype=np.dtype("int")).copy()
			self.data_thread_lock.release()

			array = np.reshape(array, (self.data_buffer.height, self.data_buffer.width, 1))

			return array
		return None


class IMU(Sensor):
	"""
	IMU sensor.
	"""

	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self):
		if self.data_buffer is not None:
			array = np.array([self.data_buffer.accelerometer.x,
							  self.data_buffer.accelerometer.y,
							  self.data_buffer.accelerometer.z,
							  self.data_buffer.gyroscope.x,
							  self.data_buffer.gyroscope.y,
							  self.data_buffer.gyroscope.z,
							  self.data_buffer.compass])
			return array
		return None


class GNSS(Sensor):
	"""
	GNSS sensor.
	"""
	def __init__(self, name: str, attributes: dict, parent_actor: carla.Actor):
		super().__init__(name, attributes, parent_actor)

	def get_data(self):
		if self.data_buffer is not None:
			array = np.array([self.data_buffer.latitude,
							  self.data_buffer.longitude,
							  self.data_buffer.altitude])
			return array
		return None
