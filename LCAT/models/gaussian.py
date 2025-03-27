import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        # kernel=[
        #         [-1,0,1],
        #         [-2,0,2],
        #         [-1,0,1]
        #         ]

        #kernel = [
        #     [1, 1, 1],
        #     [1,-8, 1],
        #     [1, 1, 1]
        # ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(128,128,1,1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1)#x.unsqueeze(0)
        return x


if __name__=='__main__':
    input_x = cv2.imread("./data/Brendan_Stai_0001.jpg")
    print(input_x.shape)
    cv2.imshow("input_x", input_x)
    input_x = Variable(torch.from_numpy(input_x.astype(np.float32))).permute(2, 0, 1)
    gaussian_conv = GaussianBlurConv()
    out_x = gaussian_conv(input_x)
    out_x[out_x<-0.2]=0
    out_x = out_x.squeeze(0).permute(1, 2, 0).data.numpy().astype(np.uint8)
    cv2.imshow("out_x", out_x)
    cv2.waitKey(0)
