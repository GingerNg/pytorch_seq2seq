import torch
import os
from .file_utils import mkdir_folder

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")


def save_checkpoint(model, epoch, save_folder):
    # if cfg.GPUS > 1:
    checkpoint = {'model': model,
                  'model_state_dict': model.state_dict(),
                  # 'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch}
    mkdir_folder(save_folder)
    save_path = os.path.join(save_folder, 'epoch_{}.pth'.format(epoch))
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model
