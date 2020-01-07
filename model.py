import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pdb

class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        # pdb.set_trace()
        # (Pdb) img.shape
        # torch.Size([1, 3, 288, 512])

        x = self.backbone(img)
        # (Pdb) x.shape
        # torch.Size([1, 512, 36, 64])

        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([1, 128, 36, 64])
        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([1, 5, 18, 32])

        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([1, 2880])

        # (Pdb) exist_pred.shape
        # torch.Size([1, 4])

        # (Pdb) seg_gt, exist_gt
        # (None, None)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        # pdb.set_trace()
        # (Pdb) seg_pred.shape, exist_pred.shape, loss_seg.shape, loss_exist.shape, loss.shape
        # (torch.Size([1, 5, 288, 512]), torch.Size([1, 4]), torch.Size([]), torch.Size([]), torch.Size([]))
        
        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def message_passing_forward(self, x):
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([1, 128, 36, 64])

        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)

        # pdb.set_trace()
        # (Pdb) for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):print(ms_conv, v, r)
        # Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False) True False
        # Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False) True True
        # Conv2d(128, 128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False) False False
        # Conv2d(128, 128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False) False True

        # (Pdb) x.shape
        # torch.Size([1, 128, 36, 64])

        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """

        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([1, 128, 36, 64])
        # conv = Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        # vertical = True
        # reverse = False

        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]

        # pdb.set_trace()
        # (Pdb) torch.cat(out, dim=dim).shape
        # torch.Size([1, 128, 36, 64])

        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w/16) * int(input_h/16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # pdb.set_trace()
        # input_size = (512, 288) --> ? 5*(32*18)
        # (Pdb) pp self.fc_input_feature
        # 2880

        # (Pdb) a
        # self = SCNN(
        #   (backbone): Sequential(
        #     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU(inplace=True)
        #     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #     (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (9): ReLU(inplace=True)
        #     (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (12): ReLU(inplace=True)
        #     (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #     (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (16): ReLU(inplace=True)
        #     (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (19): ReLU(inplace=True)
        #     (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (22): ReLU(inplace=True)
        #     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #     (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (26): ReLU(inplace=True)
        #     (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (29): ReLU(inplace=True)
        #     (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (32): ReLU(inplace=True)
        #     (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (36): ReLU(inplace=True)
        #     (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (39): ReLU(inplace=True)
        #     (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (42): ReLU(inplace=True)
        #     (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #   )
        # )
        # input_size = (512, 288)
        # ms_ks = 9


        # ----------------- process backbone -----------------
        #     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (36): ReLU(inplace=True)

        #     (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (39): ReLU(inplace=True)

        #     (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (42): ReLU(inplace=True)
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
            # pdb.set_trace()
            # (Pdb) pp conv
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (Pdb) pp dilated_conv
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))

        # pdb.set_trace()
        self.backbone._modules.pop('33')
        # (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.backbone._modules.pop('43')
        # (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        # pdb.set_trace()
        # (Pdb) pp self.message_passing
        # ModuleList(
        #   (up_down): Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        #   (down_up): Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        #   (left_right): Conv2d(128, 128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
        #   (right_left): Conv2d(128, 128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
        # )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()


if __name__ == '__main__':
    import time
    from tqdm import tqdm

    use_gpu = True
    epochs = 100
    heigh = 720
    width = 1280

    model = SCNN(input_size=(heigh, width), pretrained=False)
    # print(model)
    input = torch.rand(1, 3, heigh, width)

    model.eval()
    if use_gpu:
        model = model.cuda()
        input = input.cuda()

    start = time.time()
    for i in tqdm(range(epochs)):
        with torch.no_grad():
           output = model(input)
    spends = (time.time() - start) * 1000.0 / epochs  # 1s = 1000 ms

    print("Average spend {} ms".format(spends))
    # GPU Average spend 395.5410861968994 ms
    # CPU Average spend 5552.44259595871 ms


