import torch
import argparse
import model
import time
import dataset
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter, accuracy
from torch.utils.data.sampler import SubsetRandomSampler

def main(args):
    net = model.ResNet18()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    if args.cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = dataset.DatasetMNIST("./train_data.csv", transform=train_transform)
    val_dataset = dataset.DatasetMNIST("./train_data.csv", transform=test_transform)

    validation_ratio = 0.1
    random_seed = 10

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_ratio * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, sampler=valid_sampler)


    max_acc = 0

    for epoch in range(args.epoch):
        print("----- epoch : {} -----".format(epoch))
        train(train_loader, epoch, net, optimizer, criterion, args)
        top1 = validate(val_loader, epoch, net, optimizer, criterion, args)
        scheduler.step()

        if max_acc <= top1:
            torch.save(net.state_dict(), "./model_best.pth.tar")

    print(f"Best Accuracy : {max_acc}")



def train(train_loader, epoch, net, optimizer, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    net.train()

    for idx, (img, label) in enumerate(train_loader):
        if args.cuda:
            img = img.cuda()
            label = label.cuda()

        # compute output
        output = net(img)
        loss = criterion(output, label)

        # measure accuracy and record loss, accuracy 
        prec1 = accuracy(output, label, topk=(1,))[0]
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        .format(
                            epoch, idx, len(train_loader), batch_time=batch_time,
                            loss=losses, top1=top1))


def validate(val_loader, epoch, net, optimizer, criterion, args):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        net.eval()
        end = time.time()
        for idx, (img, label) in enumerate(val_loader):
            if args.cuda:
                label = label.cuda()
                img = img.cuda()

            output = net(img)

            # get loss from loss function.
            loss = criterion(output, label)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, label, topk=(1,))[0]

            # record loss and accuracy
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 50 == 0:
                print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            .format(
                                idx, len(val_loader), batch_time=batch_time, loss=losses,
                                top1=top1))
    
    return top1.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.1, help="scheduler gamma")
    parser.add_argument("--batch-size", default=256, help="train config file")
    parser.add_argument("--learning-rate", default=0.1, help="learing rate of optimizer")
    parser.add_argument("--epoch", default=200, help="default")
    parser.add_argument("--cuda", action='store_true', help="use GPU ")
    args = parser.parse_args()
    main(args)