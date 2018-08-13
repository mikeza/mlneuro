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
