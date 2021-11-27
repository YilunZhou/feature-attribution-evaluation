
import os, sys
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

class Trainer(object):
	def __init__(self, dataset_root, step=2, val_split='val', batchsize=128, max_epoch=50, device='cuda'):
		self.max_epoch = max_epoch
		self.log_folder = os.path.join(dataset_root, 'training')
		os.makedirs(self.log_folder)
		self.log_file = open(os.path.join(self.log_folder, 'progress.log'), 'w')
		self.device = device

		transform = [
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.485, 0.456, 0.406),
								 std= (0.229, 0.224, 0.225))
		]
		train_dataset = ImageFolder(root=os.path.join(dataset_root, f'step{step}', 'train'), transform=transforms.Compose(transform))
		val_dataset = ImageFolder(root=os.path.join(dataset_root, f'step{step}', val_split), transform=transforms.Compose(transform))
		self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
		self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

		num_classes = len(train_dataset.classes)
		self.model = models.resnet34(pretrained=False)
		self.model.fc = nn.Linear(512, num_classes)
		self.model.to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters())
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=30, factor=0.1)
		self.loss = nn.CrossEntropyLoss()

	def train(self):
		best_acc = -1
		last_acc = -1
		for epoch in range(self.max_epoch):
			self.model.train()
			train_loss = []
			for imgs, labels in tqdm(self.train_loader, ncols=60):
				self.optimizer.zero_grad()
				imgs = imgs.to(self.device)
				labels = labels.to(self.device)
				output = self.model(imgs)
				loss_val = self.loss(output, labels)
				train_loss.append(loss_val.item())
				loss_val.backward()
				self.optimizer.step()

			val_acc = self.check_accuracy()
			self.scheduler.step(val_acc)
			if val_acc > best_acc:
				torch.save(self.model.state_dict(), os.path.join(self.log_folder, 'best.pt'))
				best_acc = val_acc
			if val_acc > last_acc + 0.05:
				torch.save(self.model.state_dict(), os.path.join(self.log_folder, f'epoch_{epoch + 1}.pt'))
				last_acc = val_acc
			info = f'Epoch: {epoch + 1}, Train Loss: {np.mean(train_loss):0.3f}, Val Acc: {val_acc:0.3f}'
			print(info)
			self.log_file.write(info + '\n')
			self.log_file.flush()
			if val_acc > 0.97:
				break

	def check_accuracy(self):
		self.model.eval()
		num_total = 0
		num_correct = 0
		with torch.no_grad():
			for imgs, labels in tqdm(self.val_loader, ncols=60):
				imgs = imgs.to(self.device)
				logits = self.model(imgs).cpu().numpy()
				preds = logits.argmax(axis=1)
				num_correct += (preds == labels.numpy()).sum()
				num_total += len(labels)
		return num_correct / num_total

def train_subfolders(root_folder):
	for fd in tqdm(os.listdir(root_folder), ncols=70):
		if os.path.isdir(os.path.join(root_folder, fd, 'training')):
			continue
		trainer = Trainer(dataset_root=os.path.join(root_folder, fd))
		trainer.train()

if __name__ == '__main__':
	# to train the models for one set of experiment, comment out the respective line.
	print('Nothing to do here. Please comment out one line in the __main__ of train.py for model training. ')
	# train_subfolders('attr_vs_er_experiment')
	# train_subfolders('visibility_experiment')
	# train_subfolders('orig_corr_strength_experiment')
