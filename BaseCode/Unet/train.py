from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
from DataUtils.data import *
from Unet.unet import *
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_path = 'Params/unet.pth'

data_path = 'D:/PythonProject/BaseCode/Data/VOC2007'
save_path = 'TrainImage'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=32, shuffle=True, drop_last=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successfully load weights')
    else:
        print('no weights')

    opt = optim.Adam(net.parameters(), lr=1e-4)
    loss_fun = nn.BCELoss()

    epoch = 1
    while True:
        for i, (image, segment_image) in enumerate(data_loader):
            image = image.to(device)
            segment_image = segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5==0:
                print(f'epoch: {epoch} {i},train_loss: {train_loss.item():.4f}')

            if i%50==0:
                torch.save(net.state_dict(), weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')

        epoch += 1

