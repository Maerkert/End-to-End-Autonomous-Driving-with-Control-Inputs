from typing import Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.environments.carla_env import CarlaEnv
from src.environments.carla_server import CarlaServer
import logging

logging.basicConfig(level=logging.INFO)


class RedrawableImagePlot:

	def __init__(self, image_dict: Dict[str, np.ndarray]):
		self.image_dict = {}
		for key, value in image_dict.items():
			if type(value) == np.ndarray and len(value.shape) == 3:
				self.image_dict[key] = value
		if len(self.image_dict) == 0:
			raise ValueError("No images in image_dict")

		self.fig, self.axs = plt.subplots(1, len(self.image_dict))
		self.redraw()
		plt.plot()

	def update(self, image_dict: Dict[str, np.ndarray]):
		self.image_dict = {}
		for key, value in image_dict.items():
			if type(value) == np.ndarray and len(value.shape) == 3:
				self.image_dict[key] = value
		self.redraw()

	def redraw(self):
		for i, (key, value) in enumerate(self.image_dict.items()):
			self.axs[i].imshow(value)
			self.axs[i].set_title(key)
		plt.draw()


if __name__ == "__main__":
	env = None
	try:
		print("Getting client")
		client = CarlaServer.connect_client(2000)
		print("Connected to client")

		print("Creating environment")
		env = CarlaEnv(client)
		print("Created environment")

		obs, reward, done, info = env.step((0.0, 0.0))
		redrawable_image_plot = RedrawableImagePlot(obs)

		for i in range(1000):
			obs, reward, done, info = env.step((0.0, 0.3))
			redrawable_image_plot.update(obs)

		print("Observation: ", obs)
	finally:
		if env is not None:
			env.close()
