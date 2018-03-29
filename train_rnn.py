import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datafeeder import McepDataSet, collate_fn
from model import McepNet

# hyper parameters
ssp = "clb"
tsp = "slt"
data_root = "/mnt/lustre/sjtu/users/kc430/data/my/vc/cmu_arctic/"
scp = "scp/train.scp"
epochs = 10
batch_size = 2
use_cuda = torch.cuda.is_available()


def get_args():
    def str_to_bool(string):
        return True if string == 'True' or string == "true" else False
    parser = argparse.ArgumentParser(description='The Voice Conversion using RNN')
    parser.add_argument('--ssp', type=str, default=ssp, help='the source speaker')
    parser.add_argument('--tsp', type=str, default=tsp, help='the target speaker')
    parser.add_argument('--data_root', type=str, default=data_root, help='the root of data, by preprocess.py')
    parser.add_argument('--epochs', type=int, default=epochs, help='the epochs of training')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='the batch size of dataloader')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--in_dim', type=int, default=25, help='the input dim of network')
    parser.add_argument('--out_dim', type=int, default=25, help='the output dim of network')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the hidden dim of LSTM Cell')
    parser.add_argument('--num_layers', type=int, default=2, help='the num layers of LSTM Cell')
    parser.add_argument('--bidirectional', type=str_to_bool, default=True, help='whether use bidirectional LSTM')

    args = parser.parse_args()
    return args


def debug_args(args):
    print("*"*100)
    print('Hyper parameters as below:')
    for k, v in vars(args).items():
        print("{:<30} {:<30}".format(k, v))
    print("*" * 100)


def save_checkpoint(net, optimizer, cpt_name):
    torch.save(
        {
            'model': net,
            'optimizer': optimizer,
        },
        'checkpoints/{}.cpt'.format(cpt_name)
    )


def main():
    args = get_args()
    debug_args(args)

    train_dataset = McepDataSet(args.ssp, args.tsp, args.data_root, "scp/train.scp")
    dev_dataset = McepDataSet(args.ssp, args.tsp, args.data_root, "scp/dev.scp")

    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=2, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, num_workers=2, shuffle=False, collate_fn=collate_fn)

    dataloaders = {'train': train_dataloader, 'dev': dev_dataloader}

    net = McepNet(args.in_dim, args.out_dim, args.hidden_dim, args.num_layers, args.bidirectional)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.2, min_lr=1e-5, verbose=True)

    if use_cuda:
        net, criterion = net.cuda(), criterion.cuda()

    for epoch in range(args.epochs):
        print_loss = {'train': 0., 'dev': 0.}
        for phase in ['train', 'dev']:
            for inputs, outputs, lengths in dataloaders[phase]:
                sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long().numpy()
                inputs, outputs = Variable(inputs[indices]), Variable(outputs[indices])

                h, c = net.init_hidden(len(sorted_lengths))

                if use_cuda:
                    inputs, outputs, h, c = inputs.cuda(), outputs.cuda(), h.cuda(), c.cuda()

                optimizer.zero_grad()
                predicts = net(inputs, sorted_lengths, h, c)

                loss = criterion(predicts.view(-1, args.out_dim), outputs.view(-1, args.in_dim))
                print_loss[phase] += loss.data[0]

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            print_loss[phase] /= len(dataloaders[phase])
        print('Epoch {:<10} Train Loss: {:<20.4f} Dev Loss: {:<20.4f}'.format(epoch, print_loss['train'], print_loss['dev']))
        scheduler.step(print_loss['dev'])
    save_checkpoint(net, optimizer, '{}-{}-rnn'.format(args.ssp, args.tsp))


if __name__ == '__main__':
    main()