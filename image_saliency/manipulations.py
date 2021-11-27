
import shutil, uuid, os, random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib

class Manipulation():
	'''
	the base class for all input manipulation operation.
	subclasses need to overwrite __call__(), bounded_area_mask(), and name(), at the bare minimum.
	'''
	def __init__(self):
		pass

	def __call__(self, in_fn, out_fn):
		'''read the image located at in_fn, apply input manipulation, and save the new image at out_fn'''
		raise NotImplementedError

	def bounded_area_mask(self):
		'''
		return a 224 x 224 binary np.array with 1 at locations for which the input manipulation is applied, and 0 otherwise
		it is assumed that the mask is independent of the image
		'''
		raise NotImplementedError

	def name(self):
		'''return the name of the input manipulation, as well as any parameters'''
		raise NotImplementedError

	@staticmethod
	def from_text(txt):
		return eval(txt.replace(';', ','))

class NoOp(Manipulation):
	def __call__(self, in_fn, out_fn):
		if out_fn is not None:
			shutil.copy(in_fn, out_fn)
		else:
			img = Image.open(in_fn)
			return img

	def bounded_area_mask(self):
		return np.zeros((224, 224))

	def name(self):
		return 'NoOp()'

class PeripheralBlurring(Manipulation):
	def __init__(self, radius=130, blur_sigma=1):
		self.radius = radius
		self.blur_sigma = blur_sigma

	def __call__(self, in_fn, out_fn):
		img = Image.open(in_fn)
		mask = Image.new('L', (224, 224), 255)
		draw = ImageDraw.Draw(mask)
		draw.ellipse([112 - self.radius, 112 - self.radius, 112 + self.radius, 112 + self.radius], fill=0)
		blurred = img.filter(ImageFilter.GaussianBlur(radius=self.blur_sigma))
		img.paste(blurred, mask=mask)
		if out_fn is not None:
			img.save(out_fn)
		else:
			return img

	def bounded_area_mask(self):
		mask = Image.new('L', (224, 224), 1)
		draw = ImageDraw.Draw(mask)
		draw.ellipse([112 - self.radius, 112 - self.radius, 112 + self.radius, 112 + self.radius], fill=0)
		return np.array(mask)

	def name(self):
		return f'PeripheralBlurring(radius={self.radius}; blur_sigma={self.blur_sigma})'

class CenterBrightnessShift(Manipulation):
	def __init__(self, radius=50, shift=-0.2):
		self.radius = radius
		self.shift = shift

	def __call__(self, in_fn, out_fn):
		img = Image.open(in_fn)
		xs = np.linspace(-1, 1, 224)
		ys = np.linspace(-1, 1, 224)
		xs, ys = np.meshgrid(xs, ys)
		rs = (xs ** 2 + ys ** 2) ** 0.5
		ds = (np.cos(0.25 * np.pi / (self.radius / 112 / 2) * rs) ** 2) * self.shift
		ds[rs > self.radius / 112] = 0
		img_np = np.array(img).astype('float32') / 255
		img_np = matplotlib.colors.rgb_to_hsv(img_np)
		img_np[:, :, 2] += ds
		img_np = np.clip(img_np, 0, 1)
		img_np = (matplotlib.colors.hsv_to_rgb(img_np) * 255).astype('uint8')
		img = Image.fromarray(img_np)
		if out_fn is not None:
			img.save(out_fn)
		else:
			return img

	def bounded_area_mask(self):
		xs = np.linspace(-1, 1, 224)
		ys = np.linspace(-1, 1, 224)
		xs, ys = np.meshgrid(xs, ys)
		rs = (xs ** 2 + ys ** 2) ** 0.5
		mask = (rs <= self.radius / 112).astype('int')
		return mask

	def name(self):
		return f'CenterBrightnessShift(radius={self.radius}; shift={self.shift})'

class StripedHueShift(Manipulation):
	def __init__(self, top=62, bottom=162, shift=0.04):
		self.top = top
		self.bottom = bottom
		self.shift = shift

	def __call__(self, in_fn, out_fn):
		img = Image.open(in_fn)
		xs = np.arange(224)
		ds = np.sin(2 * np.pi / (self.bottom - self.top) * (xs - 112))
		ds = ds ** 2 * np.sign(ds) * self.shift
		ds[xs < self.top] = 0
		ds[xs > self.bottom] = 0
		ds = np.tile(ds, (224, 1)).T
		img_np = np.array(img).astype('float32') / 255
		img_np = matplotlib.colors.rgb_to_hsv(img_np)
		img_np[:, :, 0] += ds
		img_np[img_np < 0] += 1
		img_np[img_np > 1] -= 1
		img_np = np.clip(img_np, 0, 1)
		img_np = (matplotlib.colors.hsv_to_rgb(img_np) * 255).astype('uint8')
		img = Image.fromarray(img_np)
		if out_fn is not None:
			img.save(out_fn)
		else:
			return img

	def bounded_area_mask(self):
		mask = np.zeros((224, 224))
		mask[self.top : self.bottom + 1] = 1
		return mask

	def name(self):
		return f'StripedHueShift(top={self.top}; bottom={self.bottom}; shift={self.shift})'

class StripedNoise(Manipulation):
	def __init__(self, top=102, bottom=122, noise_p=0.02):
		self.top = top
		self.bottom = bottom
		self.noise_p = noise_p

	def __call__(self, in_fn, out_fn):
		img = Image.open(in_fn)
		img_np = np.array(img)
		for i in range(self.top, self.bottom + 1):
			for j in range(224):
				if np.random.random() < self.noise_p:
					img_np[i, j, :] = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
		img = Image.fromarray(img_np)
		if out_fn is not None:
			img.save(out_fn)
		else:
			return img

	def bounded_area_mask(self):
		mask = np.zeros((224, 224))
		mask[self.top : self.bottom + 1] = 1
		return mask

	def name(self):
		return f'StripedNoise(top={self.top}; bottom={self.bottom}; noise_p={self.noise_p})'

class Watermark(Manipulation):
	def __init__(self, top=200, bottom=220, left=20, right=204, fontsize=13):
		self.top = top
		self.left = left
		self.bottom = bottom
		self.right = right
		self.fontsize = fontsize

	def __call__(self, in_fn, out_fn):
		img = Image.open(in_fn)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype('fonts/OpenSans-Regular.ttf', self.fontsize)
		pos = (np.random.uniform(self.left + 40, self.right - 40), np.random.uniform(self.top + 10, self.bottom - 10))
		black = (0, 0, 0, 50)
		white = (255, 255, 255, 50)
		if np.random.random() < 0.5:
			color = [black, white]
		else:
			color = [white, black]
		draw.text(pos, 'IMG', fill=color[0], font=font, anchor='rm')
		draw.text(pos, str(uuid.uuid4())[:4].upper(), fill=color[1], font=font, anchor='lm')
		if out_fn is not None:
			img.save(out_fn)
		else:
			return img

	def bounded_area_mask(self):
		mask = np.zeros((224, 224))
		mask[self.top : self.bottom, self.left : self.right] = 1
		return mask

	def name(self):
		return f'Watermark(top={self.top}; left={self.left}; bottom={self.bottom}; right={self.right}; fontsize={self.fontsize})'

def test_manipulation(root_dir='bird_dataset/val'):
	no_op = NoOp()
	peripheral_blurring = PeripheralBlurring(radius=130, blur_sigma=1)
	center_brightness_shift = CenterBrightnessShift(radius=50, shift=-0.2)
	striped_hue_shift = StripedHueShift(top=62, bottom=162, shift=0.04)
	striped_noise = StripedNoise(top=102, bottom=122, noise_p=0.02)
	watermark = Watermark(top=200, bottom=220, left=20, right=204)
	classes = os.listdir(root_dir)
	images = []
	for c in classes:
		images = images + [os.path.join(root_dir, c, img) for img in os.listdir(os.path.join(root_dir, c))]
	random.shuffle(images)
	for manip in [no_op, peripheral_blurring, center_brightness_shift, striped_hue_shift, striped_noise, watermark]:
		plt.figure(figsize=[9, 3])
		for i in range(6):
			img = Image.open(images[i])
			plt.subplot(3, 6, i + 1)
			plt.imshow(img)
			plt.axis('off')
			plt.subplot(3, 6, i + 1 + 6)
			img = manip(in_fn=images[i], out_fn=None)
			plt.imshow(img)
			plt.axis('off')
			plt.subplot(3, 6, i + 1 + 12)
			plt.imshow(manip.bounded_area_mask())
			plt.axis('off')
		plt.tight_layout()
		plt.suptitle(manip.name())
		plt.show()

if __name__ == '__main__':
	test_manipulation()
