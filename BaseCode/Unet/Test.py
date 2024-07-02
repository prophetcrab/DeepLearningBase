import os.path

from Unet.unet import *
from DataUtils.utils import *
from DataUtils.data import *
from torchvision.utils import save_image
net = UNet().cuda()

weights = 'Params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully load weights')
else:
    print('no loading')


_input = input('please enter an image path: ')
img = keep_image_size_open(_input)
img_data = transform(img)
img_data = img_data.unsqueeze(dim=0).cuda()
out = net(img_data)
save_image(out, 'Result/result.jpg')

print(out.shape)