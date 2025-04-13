"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt
import nice
import pickle


def train(flow, trainloader, optimizer, epoch, device):
    flow.train()  # set to training mode
    epoch_loss = 0.
    for inputs, _ in trainloader:
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(
            device)  # change  shape from BxCxHxW to Bx(C*H*W)
        # TODO Fill in

        # zero the grads
        optimizer.zero_grad()

        # forward
        loss = - flow(inputs).mean()
        
        epoch_loss += loss.item()

        # backward
        loss.backward()

        # step
        optimizer.step()

    return epoch_loss / len(trainloader)


def test(flow, testloader, filename, epoch, sample_shape, device, should_sample: bool = False):
    flow.eval()  # set to inference mode
    epoch_loss = 0.
    with torch.no_grad():
        if should_sample:
            samples = flow.sample(100)
            a, b = samples.min(), samples.max()
            samples = (samples - a) / (b - a + 1e-10)
            samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                         './samples/' + filename + 'epoch%d.png' % epoch)
        # TODO full in

        for inputs, _ in testloader:
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(
                device)  # change  shape from BxCxHxW to Bx(C*H*W)
            # forward
            loss = - flow(inputs).mean()

            epoch_loss += loss.item()

        return epoch_loss / len(testloader)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'

    full_dim = 28 * 28
    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    # TODO fill in

    # history to plot
    loss_train_history = []
    loss_test_history = []

    for epoch in range(args.epochs):
        # train
        loss_train = train(flow, trainloader, optimizer, epoch, device)
        loss_train_history.append(loss_train)
        # eval
        loss_test = test(flow, testloader, model_save_filename, epoch,
                         sample_shape, device)
        loss_test_history.append(loss_test)
        print(f'Epoch: {epoch} / {args.epochs}. Train loss = {loss_train}. Test loss = {loss_test}')

    # sample
    test(flow, testloader, model_save_filename, args.epochs,
         sample_shape, device, should_sample=True)
    
    # save history
    train_filename = f'train_{args.dataset}_{args.coupling_type}.pkl'
    test_filename = f'test_{args.dataset}_{args.coupling_type}.pkl'
    
    # After training, generate the plot
    plt.figure(figsize=(10, 6))  # Set the size of the plot
    plt.plot(range(args.epochs), loss_train_history, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(range(args.epochs), loss_test_history, label='Test Loss', color='red', linestyle='--', marker='x')
    
    # Add labels and title
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Train and Test Loss Over Epochs', fontsize=14)
    
    # Add a legend
    plt.legend()
    
    # Grid and layout adjustments for a better look
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot as an image
    plot_filename = f'loss_plot_{args.dataset}_{args.coupling_type}.png'
    plt.savefig(plot_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
    