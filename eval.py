import torch
import pandas as pd
import argparse
import model
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter, accuracy


def eval(args):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.ImageFolder("./dataset/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    net = model.resnet20()
    if args.cuda:
        net = net.cuda()
        net.load_state_dict(torch.load(args.weight_file))
    else:
        net.load_state_dict(torch.load(args.weight_file, map_location='cpu'))

    Category = []
    for input, _ in test_loader:
        if args.cuda:
            input = input.cuda()
        output = net(input)
        output = torch.argmax(output, dim=1)
        Category = Category + output.tolist()

    Id = list(range(0, 8000))
    samples = {
       'Id': Id,
       'Category':Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Category'])

    df.to_csv('./submit.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, help="batch size")
    parser.add_argument('--weight-file', required=True, default="./model_best.pth.tar", help="model weight path")
    parser.add_argument("--cuda", action='store_true', help="use GPU ")
    args = parser.parse_args()
    eval(args)

