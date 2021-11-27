
import os, uuid
from tqdm import tqdm, trange

from dataset_modification import DatasetModifier
from manipulations import *

'''
Build sequences of datasets, where all datasets in a sequence have the same manipulation in the same bounded area.
however, they differ in the visibility of the manipulation (e.g. radius of blurring).
'''

def random_peripheral_blurring_sequence():
	radius = np.random.uniform(80, 120)
	blur_sigmas = [2, 4, 6, 8, 10]
	return [PeripheralBlurring(radius=radius, blur_sigma=blur_sigma) for blur_sigma in blur_sigmas]

def random_center_brightness_shift_sequence():
	radius = np.random.uniform(40, 112)
	shifts = [-0.1, -0.15, -0.2, -0.25, -0.3]
	return [CenterBrightnessShift(radius=radius, shift=shift) for shift in shifts]

def random_striped_hue_shift_sequence():
	size = np.random.uniform(50, 100)
	top = np.random.uniform(0, 224 - size)
	bottom = top + size
	shifts = [0.05, 0.1, 0.15, 0.2, 0.25]
	return [StripedHueShift(top=int(top), bottom=int(bottom), shift=shift) for shift in shifts]

def random_striped_noise_sequence():
	size = np.random.uniform(50, 100)
	top = np.random.uniform(0, 224 - size)
	bottom = top + size
	noise_ps = [0.02, 0.04, 0.06, 0.08, 0.1]
	return [StripedNoise(top=int(top), bottom=int(bottom), noise_p=noise_p) for noise_p in noise_ps]

def random_watermark_sequence():
	height = np.random.uniform(40, 100)
	width = np.random.uniform(100, 200)
	top = np.random.uniform(0, 224 - height)
	bottom = top + height
	left = np.random.uniform(0, 224 - width)
	right = left + width
	fontsizes = [7, 9, 11, 13, 15]
	return [Watermark(top=int(top), bottom=int(bottom), left=int(left), right=int(right), fontsize=fontsize) for fontsize in fontsizes]

def random_class_pairs():
	return random.choice([['brandts cormorant', 'pelagic cormorant'],
						  ['fish crow', 'common raven'],
						  ['forsters tern', 'common tern'],
						  ['pomarine jaeger', 'long tailed jaeger']])

def build_modification(manip, class_pair, folder_name):
	r = 0.5
	p_manip = 1
	correlations = [NoOp(), manip]
	dm = DatasetModifier('bird_dataset', class_pair, [[r, 1 - r], [1 - r, r]],
						 correlations, [[p_manip, 1 - p_manip], [1 - p_manip, p_manip]],
						 os.path.join(f'visibility_experiment', folder_name))
	dm.run()

if __name__ == '__main__':
	class_pairs = [['brandts cormorant', 'pelagic cormorant'],
				   ['fish crow', 'common raven'],
				   ['forsters tern', 'common tern'],
				   ['pomarine jaeger', 'long tailed jaeger']]
	for manip_func, manip_name in zip([random_peripheral_blurring_sequence, random_center_brightness_shift_sequence,
		random_striped_hue_shift_sequence, random_striped_noise_sequence, random_watermark_sequence],
		['peripheral-blurring', 'center-brightness-shift', 'striped-hue-shift', 'striped-noise', 'watermark']):
		print(manip_name)
		for i in range(4):
			manip_sequence = manip_func()
			for j, manip in enumerate(manip_sequence):
				build_modification(manip, class_pairs[i], f'{manip_name}-{i}-{j}')
