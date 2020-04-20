import numpy as np
import argparse
import torch
from torch.autograd import Variable
import time
import cv2

# define your model
from example.RetinaFace.retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
net = RetinaFace(cfg=cfg_mnet, phase='test')
net = load_model(net, "/home/file_collections/gitlab/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth", False)

import caffe

#caffe load formate
def load_image_caffe(imgfile):
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def load_image_pytorch(imgfile):
    img = cv2.imread(imgfile).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.width, args.height))

    img[:, :, 0] = img[:, :, 0] / 255
    img[:, :, 1] = img[:, :, 1] / 255
    img[:, :, 2] = img[:, :, 2] / 255

    tmp = torch.from_numpy(img).float()
    tmp = tmp.permute(2, 0, 1)
    tmp = tmp.view(1, tmp.size(0), tmp.size(1), tmp.size(2))
    img = tmp.numpy()

    return img


def forward_pytorch(weightfile, image):
    # net_dict = net.state_dict()
    # model_loaded = torch.load(weightfile, map_location='cuda:0')
    # model_loaded = {k.split("module.")[-1]: v for k, v in model_loaded.items() if k.split("module.")[-1] in net_dict}
    # net.load_state_dict(model_loaded)
    # net.eval()

    if args.cuda:
        net.cuda()
    print(net)
    net.eval()
    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    t0 = time.time()
    blobs = net.forward(image)
    t1 = time.time()
    return t1-t0, blobs, net.parameters()

# Reference from:
def forward_caffe(protofile, weightfile, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs[args.input_layer].reshape(1, 3, args.height, args.width)
    net.blobs[args.input_layer].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default="/home/file_collections/gitlab/PytorchToCaffe/example/2D_Detection (copy).prototxt", type=str)
    parser.add_argument('--weightfile', default="/home/file_collections/gitlab/PytorchToCaffe/example/2D_Detection.caffemodel", type=str)
    parser.add_argument('--model', default="/home/file_collections/gitlab/PytorchToCaffe/example/convert2caffe/led_pcb/led3d_baseline_attention_arcface_preludropout_all_82.pth", type=str)
    parser.add_argument('--imgfile', default="/home/hdd/Pictures/test/detect_img.jpg", type=str)
    parser.add_argument('--height', default=640, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    parser.add_argument('--input_layer', default='blob1', type=str)
    parser.add_argument('--out_layer', default='cat_blob2', type=str)

    args = parser.parse_args()

    protofile = args.protofile
    weightfile = args.weightfile
    imgfile = args.imgfile

    image = load_image_pytorch(imgfile)
    time_pytorch, pytorch_blobs, pytorch_models = forward_pytorch(args.model, image)
    time_caffe, caffe_blobs, caffe_params = forward_caffe(protofile, weightfile, image)

    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    blob_name = args.out_layer
    if args.cuda:
        pytorch_data = pytorch_blobs.data.cpu().numpy().flatten()
    else:
        pytorch_data = pytorch_blobs.data.numpy().flatten()

    print(pytorch_data)
    caffe_data = caffe_blobs[blob_name].data[0][...].flatten()
    diff = abs(pytorch_data - caffe_data).sum()
    print('%-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' % (blob_name, pytorch_data.shape, caffe_data.shape, diff/pytorch_data.size))
