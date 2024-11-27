import argparse
import os
import yaml

import torch
from models.yolo import Model


def removal(device, cfg_path, weight_path, data_path, idxmax):
    save_path = os.path.split(opt.weights)[0] + '/converted.pt'
    with open(data_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    nc = result['nc']
    device = torch.device(device)
    cfg = cfg_path
    model = Model(cfg, ch=3, nc=nc, anchors=3)
    #model = model.half()
    model = model.to(device)
    _ = model.eval()
    ckpt = torch.load(weight_path, map_location=device)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < idxmax:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
    _ = model.eval()

    m_ckpt = {'model': model.half(),
              'optimizer': None,
              'best_fitness': None,
              'ema': None,
              'updates': None,
              'opt': None,
              'git': None,
              'date': None,
              'epoch': -1}
    torch.save(m_ckpt, save_path)
    print('save_path:', save_path)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov9-c.pt', help='填入需要去除辅助分支的权重路径')
    parser.add_argument('--cfg', type=str, default='./models/detect/gelan-c.yaml', help='填入与上方pt路径不带辅助分支的yaml路径')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--idx', type=int, default=22, help='填入检测层的序号')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    removal(opt.device, opt.cfg, opt.weights, opt.data, opt.idx)
    # with open(opt.cfg, 'r', encoding='utf-8') as f:
    #     result = yaml.load(f.read(), Loader=yaml.FullLoader)
    # print(result['head'][-1][0], len(result['head'][-1][0]))
    # print(len(result['backbone']))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
