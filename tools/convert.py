import sys
sys.path.insert(0, '.')
sys.path.append("../")

import cv2
import torch
import torchvision
from PIL import Image
import pytorch_to_caffe


# define your model
from example.convert2caffe.led_pcb.led3d_pcb import led3d
net=led3d()

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
	parser.add_argument('--name', default="global_pcb", type=str, help='save name')
	parser.add_argument('--model_path', default="/home/file_collections/gitlab/PytorchToCaffe/example/convert2caffe/led_pcb/led3d_baseline_attention_arcface_preludropout_all_82.pth", type=str, help='converted model path')
	parser.add_argument('--img_path', type=str, default='../001763.jpg')

	opt = parser.parse_args()

	net_dict = net.state_dict()
	model_loaded = torch.load(opt.model_path, map_location='cuda:0')
	model_loaded = {k.split("module.")[-1]: v for k, v in model_loaded.items() if k.split("module.")[-1] in net_dict}
	net.load_state_dict(model_loaded)
	net.eval()

	data = preprocess_img(opt.img_path)

	pytorch_to_caffe.trans_net(net, data, opt.name)
	pytorch_to_caffe.save_prototxt('{}.prototxt'.format(opt.name))
	pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(opt.name))