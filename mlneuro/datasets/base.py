import os 
import pickle

from ..utils.io import load_array_dict


def data_path():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(dir_path, 'data')
	return data_path


def load_restaurant_row():
	"""Load an example day of Restaurant Row data from the Redish lab
	"""
	RR_FILE = 'RestaurantRowExampleDay.pickle'
	return load_array_dict(os.path.join(data_path(), RR_FILE))


def load_restaurant_row_spikekde():
	"""Load an example day of Restaurant Row data from the Redish lab
	"""
	RR_FILE = 'RestaurantRowExampleKDEResults.pickle'
	return load_array_dict(os.path.join(data_path(), RR_FILE))


def load_restaurant_row_time_out(which='rr'):
	"""Load an example day with Restaurant Row and Time Out data from the Redish lab
	"""
	RR_FILE = 'RestaurantRowTimeOutExampleDay_RR.pickle'
	TO_FILE = 'RestaurantRowTimeOutExampleDay_TO.pickle'

	if which == 'to':
		return load_array_dict(os.path.join(data_path(), RR_FILE))
	elif which == 'rr':
		return load_array_dict(os.path.join(data_path(), TO_FILE))
	else:
		raise ValueError('Unknown option for which data to load.'
						 'Please specify `rr` for restaurant row or `to` for time out')
