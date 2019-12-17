import sys
sys.path.insert(0, '.')

import cv2
import torch
import torchvision
from PIL import Image
import pytorch_to_caffe


# define your model
from led3d import led3d
net=led3d(975)

def preprocess_img(img_path):
	data = cv2.imread(img_path)
	data = cv2.resize(data, (128, 128))
	data[:, :, 0] = data[:, :, 0] / 255
	data[:, :, 1] = data[:, :, 1] / 255
	data[:, :, 2] = data[:, :, 2] / 255

	data = torch.from_numpy(data).float()
	data = data.view(1, data.size(2), data.size(0), data.size(1))

	data = torch.autograd.Variable(data)
	return data

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Pytorch2Caffe')
	parser.add_argument('--name', type=str, help='save name')
	parser.add_argument('--model_path', type=str, help='converted model path')
	parser.add_argument('--img_path', type=str, default='../001763.jpg')

	opt = parser.parse_args()
	net.load_state_dict(torch.load(opt.model_path))
	net.eval()

	data = preprocess_img(opt.img_path)

	pytorch_to_caffe.trans_net(net, data, opt.name)
	pytorch_to_caffe.save_prototxt('{}.prototxt'.format(opt.name))
	pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(opt.name))