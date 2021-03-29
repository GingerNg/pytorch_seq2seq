from tensorboardX import SummaryWriter


def get_writer(folder_path):
    writer = SummaryWriter(folder_path)
    return writer