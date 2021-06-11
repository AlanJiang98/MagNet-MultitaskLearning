import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

from modules import MnistMolel, FMnistMolel, Cifar10Molel

from utils import get_dataset, mkdir, get_logger, to

def updatemtl(model, optimizer, x, y, alpha):
    optimizer.zero_grad()
    pred_cls, pred_x = model(x)
    loss1 = F.cross_entropy(pred_cls, y)
    loss2 = F.mse_loss(pred_x, x)
    loss = loss1 + alpha * loss2
    loss.backward()
    optimizer.step()
    return loss, loss1, loss2

def trainmtl(args, model, device, train_loader, optimizer,epoch, alpha, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss, loss1, loss2 = updatemtl(model, optimizer, data, target, alpha)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss all: {:.6f} \tLoss cls:{:.6f}\tLoss ae:{:.6f}'.format(epoch,
                                                                           batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.item(), loss1.item(), loss2.item()))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss all: {:.6f} \tLoss cls:{:.6f}\tLoss ae:{:.6f}'.format(epoch,
                                                                           batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.item(), loss1.item(), loss2.item()))

def evalmtl(model, device, eval_loader, alpha, logger):
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    test_loss_all = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            pred_cls, pred_data = model(data)
            test_loss1 += F.cross_entropy(pred_cls, target, reduction='sum').item()
            test_loss2 += F.mse_loss(pred_data, data, reduction='sum').item()
            test_loss_all = test_loss1 + alpha* test_loss2
            pred = pred_cls.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss1 /= len(eval_loader.dataset)
    test_loss2 /= len(eval_loader.dataset)
    test_loss_all /= len(eval_loader.dataset)
    print('\nValidation set: Loss all:{:.6f}\t Loss cls:{:.6f}\t Loss ae:{:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_all, test_loss1, test_loss2, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)
    ))
    logger.info('\nValidation set: Loss all:{:.6f}\t Loss cls:{:.6f}\t Loss ae:{:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_all, test_loss1, test_loss2, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)
    ))
    return correct / len(eval_loader.dataset)

def get_tfp_mtl(model, data, device, rate=0.001, p=2, logger=None):
    num = int(rate * len(data.dataset))
    errors = []
    for idx, (x, _) in enumerate(data):
        x = x.to(device)
        with torch.no_grad():
            _,x_ae = model(x)
            error_a = (torch.abs(x_ae - x)) ** p
            error = error_a.mean(dim=(1, 2, 3))
            errors.append(error)
    error_all = torch.cat(errors, dim=0)
    error_sort = torch.sort(error_all, dim=0)
    logger.info('Tfp = {}'.format(error_sort.values[-num].item()))
    return error_sort.values[-num].item()

def testmtl(model, device, test_loader, alpha, logger):
    model.eval()
    test_loss1 = 0
    test_loss2 = 0
    test_loss_all = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred_cls, pred_data = model(data)
            test_loss1 += F.cross_entropy(pred_cls, target, reduction='sum').item()
            test_loss2 += F.mse_loss(pred_data, data, reduction='mean').item()
            test_loss_all = test_loss1 + alpha* test_loss2
            pred = pred_cls.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss1 /= len(test_loader.dataset)
    test_loss2 /= len(test_loader.dataset)
    test_loss_all /= len(test_loader.dataset)
    print('\nTest set: Loss all:{:.6f}\t Loss cls:{:.6f}\t Loss ae:{:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_all, test_loss1, test_loss2, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    logger.info('\nTest set: Loss all:{:.6f}\t Loss cls:{:.6f}\t Loss ae:{:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_all, test_loss1, test_loss2, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    return correct / len(test_loader.dataset)

# def update_cls(model, optimizer, x, y):
#     optimizer.zero_grad()
#     output = model(x)
#     loss = F.cross_entropy(output, y)
#     loss.backward()
#     optimizer.step()
#     return loss
#
# def train_cls(args, model, device, train_loader, optimizer, epoch, logger):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         loss = update_cls(model, optimizer, data, target)
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                                                                            batch_idx * len(data), len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader), loss.item()))
#             logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                                                                            batch_idx * len(data), len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader), loss.item()))
#
# def eval_cls(model, device, eval_loader, logger):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in eval_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(eval_loader.dataset)
#     print('\nValidation set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(eval_loader.dataset),
#         100. * correct / len(eval_loader.dataset)
#     ))
#     logger.info('\nValidation set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(eval_loader.dataset),
#         100. * correct / len(eval_loader.dataset)
#     ))
#     return correct / len(eval_loader.dataset)
#
# def test_cls(model, device, test_loader,logger):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)
#     ))
#     logger.info('\nTest set: Average loss:{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)
#     ))
#     return correct / len(test_loader.dataset)

def train_ae(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, x)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx*len(x),
                                                                            len(train_loader.dataset),
                                                                            100.0 * batch_idx / len(train_loader),
                                                                            loss.item()))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx*len(x),
                                                                            len(train_loader.dataset),
                                                                            100.0 * batch_idx / len(train_loader),
                                                                            loss.item()))

def get_tfp(model, data, device, rate=0.001, p=2, logger=None):
    num = int(rate * len(data.dataset))
    errors = []
    for idx, (x, _) in enumerate(data):
        x = x.to(device)
        with torch.no_grad():
            x_ae = model(x)
            error_a = (torch.abs(x_ae - x)) ** p
            error = error_a.mean(dim=(1, 2, 3))
            errors.append(error)
    error_all = torch.cat(errors, dim=0)
    error_sort = torch.sort(error_all, dim=0)
    logger.info('tfp ={}'.format(error_sort.values[-num].item()))
    return error_sort.values[-num].item()

def eval_ae(model, device, eval_loader, logger):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in eval_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data).item()
    test_loss /= len(eval_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    logger.info('\nValidation set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss

def test_ae(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data).item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    logger.info('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss

def test_all(model, device, test_loader, logger):
    model.eval()
    test_cls_loss = 0
    test_correct = 0
    test_aeid_loss = 0
    test_aemtl_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = to(data), to(label)
            pred_label, pred_data_id, pred_data_mtl, _ = model(data)
            test_cls_loss += F.cross_entropy(pred_label, label, reduction='sum').item()
            pred = pred_label.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(label.view_as(pred)).sum().item()
            test_aeid_loss += F.mse_loss(pred_data_id, data).item()
            test_aemtl_loss += F.mse_loss(pred_data_mtl, data).item()
    test_cls_loss /= len(test_loader.dataset)
    print('\nTest cls set: Average loss:{:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_cls_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)
    ))
    logger.info('\nTest cls set: Average loss:{:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_cls_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)
    ))
    test_aeid_loss /= len(test_loader.dataset)
    test_aemtl_loss /= len(test_loader.dataset)
    print('\nTest Independent set: Average loss: {:.6f}\n'.format(test_aeid_loss))
    logger.info('\nTest Independent set: Average loss: {:.6f}\n'.format(test_aeid_loss))
    print('\nTest MTL set: Average loss: {:.6f}\n'.format(test_aemtl_loss))
    logger.info('\nTest MTL set: Average loss: {:.6f}\n'.format(test_aemtl_loss))


def main():
    parser = argparse.ArgumentParser(description='MNIST Classifier Pytorch implementation')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to train')
    parser.add_argument('--id', type=int, default=1, help='Model id')
    parser.add_argument('--stop_loss', type=float, default=0.0001, help='AE loss threshold when stop training')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha of MTL')
    parser.add_argument('--T', type=float, default=2.0, help='T of dis')

    parser.add_argument('--train_batch_size', type=int,default=64, metavar='N',
                        help='input batch size for train the classifier(default:64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for test the classifier(default:1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default:10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    #parser.add_argument('--model', type=int, default=1, help='1:MLP,2:RELU,3:Tanh')
    #parser.add_argument('--adv', action='store_true', help='adversarial training')

    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_data, eval_data = get_dataset(args.dataset, '../data', 32, True, 0.1)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    eval_loader = DataLoader(eval_data, batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(get_dataset(args.dataset, '../data', 32, False), batch_size=args.test_batch_size, shuffle=True, **kwargs)

    modelname = args.dataset
    models = {'mnist':MnistMolel, 'fmnist':FMnistMolel, 'cifar10':Cifar10Molel}

    # train mtl model
    if args.id == 1:
        model = models[modelname](1, args.T, 3 if args.dataset == 'cifar10' else 1).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        best_eval_acc = -1
        mkdir('./train_models/')
        logfile = './train_models/{}_MTL.txt'.format(args.dataset)
        logger = get_logger(logfile)

        for epoch in range(1, args.epochs + 1):
            trainmtl(args, model, device, train_loader, optimizer, epoch, args.alpha, logger=logger)
            correct = evalmtl(model, device, eval_loader, args.alpha, logger=logger)
            if correct > best_eval_acc:
                best_eval_acc = correct
                torch.save(model.state_dict(), './train_models/{}.pth'.format(args.dataset))
                tfp = get_tfp_mtl(model, eval_loader, device, logger=logger)
        print('Best Eval Acc: :', best_eval_acc)
        logger.info('Best Eval Acc: {}'.format(best_eval_acc))
        test_acc = testmtl(model, device, test_loader, args.alpha,logger=logger)
        print('Test Acc: ', test_acc)
        print('Tfp = ', tfp)
        logger.info('Final Tfp={}'.format(tfp))

    if args.id == 2:
        model = models[modelname](1, args.T,3 if args.dataset == 'cifar10' else 1).to(device)
        model.load_state_dict(torch.load('./train_models/{}.pth'.format(args.dataset)))
        model.set_id(2)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        logfile = './train_models/{}_AEid.txt'.format(args.dataset)
        logger = get_logger(logfile)
        best_eval_loss = 100.0

        for epoch in range(1, args.epochs + 1):
            train_ae(args, model, device, train_loader, optimizer, epoch, logger)
            eval_loss = eval_ae(model, device, eval_loader, logger)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), './train_models/{}.pth'.format(args.dataset))
                tfp = get_tfp(model, eval_loader, device, logger=logger)

        print('Best loss:', best_eval_loss)
        print('Tfp = ', tfp)
        logger.info('Best loss:'.format(best_eval_loss))
        logger.info('Final Tfp={}'.format(tfp))
        test_loss = test_ae(model, device, test_loader, logger)
        print('Test Loss: ', test_loss)

    if args.id == 3:
        model = models[modelname](1, args.T,3 if args.dataset == 'cifar10' else 1).to(device)
        model.load_state_dict(torch.load('./train_models/{}.pth'.format(args.dataset)))
        model = to(model)
        model.set_id(3)

        logfile = './train_models/{}_test.txt'.format(args.dataset)
        logger = get_logger(logfile)

        test_all(model, device, test_loader, logger)

        for x, _ in train_loader:
            break
        x = x[:64].to(device)
        with torch.no_grad():
            x_cls, x_id, x_mtl, _ = model(x)
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(x.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Recon Independent Images")
        plt.imshow(np.transpose(vutils.make_grid(x_id.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Recon MTL Images")
        plt.imshow(np.transpose(vutils.make_grid(x_mtl.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
        plt.savefig("./train_models/Autoencoder_results_{}.jpg".format(args.dataset))
        plt.show()

if __name__ == '__main__':
    main()
