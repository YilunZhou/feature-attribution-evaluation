
import os, uuid, random
from tqdm import tqdm, trange

from dataset_modification import DatasetModifier
from manipulations import *

def random_similar_class_pairs():
	pair = random.choice([['brandts cormorant', 'pelagic cormorant'],
						  ['fish crow', 'common raven'],
						  ['forsters tern', 'common tern'],
						  ['pomarine jaeger', 'long tailed jaeger']])
	if random.random() < 0.5:
		pair = [pair[1], pair[0]]
	return pair

def random_distinct_class_pairs():
	all_classes = ['brandts cormorant', 'pelagic cormorant', 'fish crow', 'common raven',
				   'forsters tern', 'common tern', 'pomarine jaeger', 'long tailed jaeger']
	while True:
		c1, c2 = random.sample(all_classes, 2)
		if (all_classes.index(c1) // 2) != (all_classes.index(c2) // 2):
			break
	return [c1, c2]

def build_modification(manip, r, class_pair):
	assert r >= 0.5
	p_manip = 1
	correlations = [NoOp(), manip]
	dm = DatasetModifier('bird_dataset', class_pair, [[r, 1 - r], [1 - r, r]],
						 correlations, [[p_manip, 1 - p_manip], [1 - p_manip, p_manip]],
						 os.path.join(f'orig_corr_strength_experiment', str(uuid.uuid4())[:6]))
	dm.run()

if __name__ == '__main__':
	scs = [PeripheralBlurring(radius=80, blur_sigma=10), CenterBrightnessShift(radius=60, shift=-0.3),
		   StripedHueShift(top=80, bottom=130, shift=0.15), StripedNoise(top=80, bottom=130, noise_p=0.03),
		   Watermark(top=150, bottom=200, left=30, right=194, fontsize=13)]
	for manip in scs:
		sp1 = random_similar_class_pairs()
		while True:  # ensure that sp1 and sp2 are two different class pairs
			sp2 = random_similar_class_pairs()
			if set(sp1) != set(sp2):
				break
		dp1 = random_distinct_class_pairs()
		while True:  # ensure that dp1 and dp2 do not have shared classes
			dp2 = random_distinct_class_pairs()
			if len(set(dp1).intersection(set(dp2))) == 0:
				break
		for class_pair in [sp1, sp2, dp1, dp2]:
			for r in tqdm([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
				build_modification(manip, r, class_pair)
