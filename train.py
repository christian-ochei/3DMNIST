import random
import threading

import torch
import torch.nn as nn
import numpy as np
import translations
import networks
import cv2
import win32api
import torchvision.datasets as datasets
import time
from torchvision import transforms

to_tensor = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
])

def visualize(renderer,volume):
    while True:
        try:
            time.sleep(0.02)
            cursor = win32api.GetCursorPos()

            rot = torch.tensor([cursor[0]-1920/2,cursor[1]-1080/2,0]) / 500

            result = renderer(translations._RotationMatrix.matrix(rot), volume)
            result = cv2.resize(((result[0].detach().numpy()) * 255), (300, 300)).astype('uint8')

            cv2.imshow('result',result)
        except:
            print('Exception in visualize')
            ...
        cv2.waitKey(20)


def train():
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=to_tensor)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=40,
                                              shuffle=True,
                                              num_workers=3)

    training = True

    renderer = networks.DifferentiableRenderer()

    discriminator = networks.LowGradeDiscriminator((40,40))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(),lr=0.004)

    volume   = networks.VoxelGrid((28,28,28))
    volume_optim = torch.optim.Adam(volume.parameters(),lr=0.07)

    criterion = nn.BCELoss()
    threading.Thread(target=visualize,args=(renderer,volume)).start()

    print('loading ...')
    # ground_truth = torch.randn(3,1,28,28)
    ground_truth, _ = next(iter(data_loader))
    # print(torch.min(ground_truth),torch.max(ground_truth),'ground_truth')
    ground_truth_ = ground_truth[:, 0]

    B = 10
    offset = (40-28)//2

    print('training')

    while training:
        # print(ground_truth.shape)
        ground_truth = torch.zeros(B,40,40)
        ground_truth[:,offset:-offset, offset:-offset] = ground_truth_[np.random.randint(0,len(ground_truth_)-1,B)]


        # visualize(renderer,volume)
        # Train GAN ---------------------------
        discriminator_optim.zero_grad()
        real = ground_truth
        real_d = discriminator(real)
        real_err = criterion(real_d, torch.ones((len(ground_truth),1)))
        real_err.backward()
        # print(f"{real_err=}")
        D_x = real_d.mean().item()

        R = torch.randn(3)
        R[2] = 0

        image_f = renderer(
            translations._RotationMatrix.matrix(R),volume
        )


        fake_d  = discriminator(image_f.detach())
        errD_fake = criterion(fake_d, torch.zeros((1,1)))
        errD_fake.backward()
        # print(f"{errD_fake=}")


        D_G_z1 = fake_d.mean().item()
        errD = real_err + errD_fake
        discriminator_optim.step()

        # Generator ---------------------------------------

        volume_optim.zero_grad()
        output = discriminator(image_f)
        # print(f"{output=}")

        # Calculate G's loss based on this output
        errG = criterion(output, torch.ones((1,1)))
        # Calculate gradients for G
        errG.backward()
        # print(
        # volume.absorbance.grad,'volume.absorbance'
        # )
        D_G_z2 = output.mean().item()
        # Update G
        volume_optim.step()

        if not random.randint(0,50):
            print('saving','.'*random.randint(0,5))
            torch.save(discriminator.state_dict(), "discriminator.pt")
            torch.save(volume.state_dict(), "volume.pt")

    ...

if __name__ == '__main__':
    train()
    ...
