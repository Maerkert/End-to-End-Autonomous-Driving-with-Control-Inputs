
def velocity_reward_function(velocity: float, goal_velocity: float):
	"""[0,1]"""
	return 1 - ((goal_velocity - min(velocity, 2 * goal_velocity)) / goal_velocity)


def line_center_reward_function(distance_to_line_center: float, line_width: float = 2.5):
	"""[-1,0]"""
	return -(distance_to_line_center / line_width)


def steering_reward_function(steering: float):
	"""[-1,0]"""
	return -((steering * 2) ** 2 / 4.0)


def line_crossing_reward_function(line_crossing: bool):
	"""[-1,0]"""
	if line_crossing:
		return -1
	return 0
