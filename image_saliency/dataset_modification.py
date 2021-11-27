
import os, shutil
from tqdm import tqdm
import numpy as np
import pickle

from manipulations import *

class DatasetModifier():
	'''
	a class that contains the complete procedure for label reassignment (step 1) and input_manipulation injection (step 2)
	'''
	def __init__(self, dataset_root, classes, lr_probs, manipulations, im_probs, output_folder, splits=['train', 'val', 'test']):
		'''
		dataset_root is the name of the folder that contains classes as subfolders
		classes is a list of splits, each of which should contain classes as subfolders
		in other words, '{dataset_root}/{split}/{class}/' should exist as a folder of images for all class in classes, and split in splits
		lr_probs is a K x K matrix of label reassignment probability, where K=len(classes), and (i, j)-th entry represents the probability of
		assigning label j to an image that is in class i. Thus, lr_probs.sum(axis=1) should be an all-1 vector (i.e. each row sums up to 1).
		manipulations is a list of input_manipulation functions, each takes two paramters, an input_fn and an output_fn,
		and saves the input_fn image with input_manipulation injected to the output fn.
		im_probs is a K x L matrix of input_manipulation probability, where L is len(manipulations), with the same semantic structure as lr_probs.
		the modified dataset will be saved to '{output_folder}/step{1|2}', with each split as a subfolder, and each class as a subfolder
		within each split, i.e. in the pytorch ImageFolder format.
		'''
		self.dataset_root = dataset_root
		self.classes = classes
		self.lr_probs = np.array(lr_probs)
		self.manipulations = manipulations
		self.im_probs = np.array(im_probs)
		self.splits = splits
		self.output_folder = output_folder

	def check_folders(self):
		assert not os.path.isdir(self.output_folder), f'Output folder {self.output_folder} already exists! '
		for c in self.classes:
			for s in self.splits:
				assert os.path.isdir(os.path.join(self.dataset_root, s, c)), f'Klass {c} split {s} does not exist! '

	def run(self):
		self.check_folders()
		assert self.lr_probs.shape == (len(self.classes), len(self.classes)), 'Incorrect shape of lr_probs matrix'
		assert np.allclose(self.lr_probs.sum(axis=1), np.ones(len(self.classes))), 'Incorrect lr_probs value'
		assert self.im_probs.shape == (len(self.classes), len(self.manipulations)), 'Incorrect shape of im_probs matrix'
		assert np.allclose(self.im_probs.sum(axis=1), np.ones(len(self.classes))), 'Incorrect im_probs value'
		for s in self.splits:
			for c in self.classes:
				os.makedirs(f'{self.output_folder}/step1/{s}/{c}')
				os.makedirs(f'{self.output_folder}/step2/{s}/{c}')
		self.write_summary()
		self.modify_step1()
		self.modify_step2()

	def modify_step1(self):
		for i, c in enumerate(self.classes):
			trans_probs = self.lr_probs[i]
			for s in self.splits:
				img_fns = os.listdir(os.path.join(self.dataset_root, s, c))
				img_fns = [f for f in img_fns if f.endswith('jpg') or f.endswith('png')]
				for img_fn in img_fns:
					new_c = np.random.choice(self.classes, p=trans_probs)
					shutil.copy(os.path.join(self.dataset_root, s, c, img_fn),
								os.path.join(self.output_folder, 'step1', s, new_c, f'{c}-{img_fn}'))

	def modify_step2(self):
		manip_idxs = list(range(len(self.manipulations)))
		for i, c in enumerate(self.classes):
			trans_probs = self.im_probs[i]
			for s in self.splits:
				img_fns = os.listdir(os.path.join(self.output_folder, 'step1', s, c))
				img_fns = [f for f in img_fns if f.endswith('jpg') or f.endswith('png')]
				for img_fn in img_fns:
					manip_idx = np.random.choice(manip_idxs, p=trans_probs)
					old_fn = os.path.join(self.output_folder, 'step1', s, c, img_fn)
					new_fn = os.path.join(self.output_folder, 'step2', s, c, f'manip{manip_idx}-{img_fn}')
					self.manipulations[manip_idx](old_fn, new_fn)

	def max_accuracy_without_manipulation(self):
		return self.lr_probs.max()

	def write_summary(self):
		f = open(os.path.join(self.output_folder, 'summary.txt'), 'w')
		f.write(f'dataset root: {self.dataset_root}\n')
		f.write(f'classes: {self.classes}\n')
		f.write(f'splits: {self.splits}\n')
		f.write(f'output folder: {self.output_folder}\n')
		f.write(f'manipulations: {[c.name() for c in self.manipulations]}\n')
		f.write(f'label reassignment probability: \n')
		np.savetxt(f, self.lr_probs)
		f.write(f'input_manipulation probability: \n')
		np.savetxt(f, self.im_probs)
		f.close()
		state_dict = {'dataset_root': self.dataset_root, 'classes': self.classes, 'splits': self.splits,
					  'output_folder': self.output_folder, 'manipulations': [c.name() for c in self.manipulations],
					  'lr_probs': self.lr_probs, 'im_probs': self.im_probs}
		pickle.dump(state_dict, open(os.path.join(self.output_folder, 'summary.pkl'), 'wb'))


def main():
	dm = DatasetModifier('bird_dataset', ['brandts cormorant', 'pelagic cormorant'], [[0.5, 0.5], [0.5, 0.5]],
						 [NoOp(), PeripheralBlurring()], [[1, 0], [0, 1]], 'test_peripheral_blurring')
	dm.run()

if __name__ == '__main__':
	main()
