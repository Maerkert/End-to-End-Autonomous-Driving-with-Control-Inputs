import cv2

from src.environments.carla_env import CarlaEnv
from src.environments.carla_server import CarlaServer

if __name__ == '__main__':

	env = None
	try:
		# Create the environment
		client = CarlaServer.connect_client(2000)
		env = CarlaEnv(client)

		env.set_auto_pilot(True)

		for i in range(10000):
			obs, reward, done, info = env.step((0, 0))
			print(i)
			cv2.imshow("image", obs["front_camera"])
			cv2.waitKey(1)

			env.log_observations("./output/observations/test/" + str(i))

	finally:
		if env is not None:
			env.close()