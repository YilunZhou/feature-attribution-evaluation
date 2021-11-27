
import os, uuid
from tqdm import tqdm, trange

from dataset_modification import DatasetModifier
from manipulations import *

def random_peripheral_blurring():
	radius = np.random.uniform(80, 130)
	blur_sigma = 2
	return PeripheralBlurring(radius=radius, blur_sigma=blur_sigma)

def random_center_brightness_shift():
	radius = np.random.uniform(40, 112)
	shift = -0.3
	return CenterBrightnessShift(radius=radius, shift=shift)

def random_striped_hue_shift():
	size = np.random.uniform(20, 224)
	top = np.random.uniform(0, 224 - size)
	bottom = top + size
	shift = 0.04
	return StripedHueShift(top=int(top), bottom=int(bottom), shift=shift)

def random_striped_noise():
	size = np.random.uniform(20, 224)
	top = np.random.uniform(0, 224 - size)
	bottom = top + size
	noise_p = 0.02
	return StripedNoise(top=int(top), bottom=int(bottom), noise_p=noise_p)

def random_watermark():
	height = np.random.uniform(40, 200)
	width = np.random.uniform(100, 200)
	top = np.random.uniform(0, 224 - height)
	bottom = top + height
	left = np.random.uniform(0, 224 - width)
	right = left + width
	return Watermark(top=int(top), bottom=int(bottom), left=int(left), right=int(right))

def random_class_pairs():
	return random.choice([['brandts cormorant', 'pelagic cormorant'],
						  ['fish crow', 'common raven'],
						  ['forsters tern', 'common tern'],
						  ['pomarine jaeger', 'long tailed jaeger']])

def build_modification(manip):
	r = 0.5
	p_manip = 1
	class_pair = random_class_pairs()
	correlations = [NoOp(), manip]
	dm = DatasetModifier('bird_dataset', class_pair, [[r, 1 - r], [1 - r, r]],
						 correlations, [[p_manip, 1 - p_manip], [1 - p_manip, p_manip]],
						 os.path.join(f'attr_vs_er_experiment', str(uuid.uuid4())[:6]))
	dm.run()

if __name__ == '__main__':
	for manip_func in tqdm([random_peripheral_blurring, random_center_brightness_shift,
						 random_striped_hue_shift, random_striped_noise, random_watermark]):
		for _ in trange(20):
			manip = manip_func()
			build_modification(manip)
