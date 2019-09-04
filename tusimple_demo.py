import argparse
import cv2
import torch

from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *

import pdb

image_resize_width = 512

net = SCNN(input_size=(image_resize_width, 288), pretrained=False)
mean=(0.3598, 0.3653, 0.3662) # CULane mean, std
std=(0.2573, 0.2663, 0.2756)
transform = Compose(Resize((image_resize_width, 288)), ToTensor(),
                    Normalize(mean=mean, std=std))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform({'img': img})['img']
    x.unsqueeze_(0)

    save_dict = torch.load(weight_path, map_location='cpu')

    net.load_state_dict(save_dict['net'])
    net.eval()

    seg_pred, exist_pred = net(x)[:2]
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()
    seg_pred = seg_pred[0]
    exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (image_resize_width, 288))
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]
    cv2.imshow("lane", lane_img)
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    for x in getLane.prob2lines_CULane(seg_pred, exist):
        print(x)

    if args.visualize:
        print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)])
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite("demo/demo_result.jpg", img)


if __name__ == "__main__":
    main()