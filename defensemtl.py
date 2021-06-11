import torch
from scipy.stats import entropy
from numpy.linalg import norm
import argparse
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from utils import to, setup_run, get_logger, mkdir

from modules import MnistMolel, FMnistMolel, Cifar10Molel

def JSD(P, Q):
    _P = P.cpu().numpy()
    _Q = Q.cpu().numpy()
    _P = _P / norm(_P, ord=1)
    _Q = _Q / norm(_Q, ord=1)
    M = 0.5 * (_P + _Q)
    dis = 0.5 * entropy(_P, M) + 0.5* entropy(_Q, M)
    return torch.from_numpy(np.array(dis))

def pre_defense(model, data, p=2):
    model_mtl = to(model).eval()
    total = 0
    base_correct = 0
    errors = [[], []]
    preds = [[], []]
    diss = [[], []]

    for id, (x, y) in enumerate(data):
        with torch.no_grad():
            pred_cls, pred_x_id, pred_x_mtl, dis_or = model_mtl(x)
            pred = pred_cls.argmax(dim=1)
            base_correct += pred.eq(y).sum().item()
            total += pred.size(0)
            for i, pred_x in enumerate([pred_x_id, pred_x_mtl]):
                error_a = (torch.abs(pred_x - x)) ** p
                error = error_a.mean(dim=(1, 2, 3))
                errors[i].append(error)
                dis_a = torch.zeros(error.size(0))
                with torch.no_grad():
                    pred_result = torch.zeros(error.size(0))
                    pred, _, _, dis_new = model(pred_x)
                    pred = pred.argmax(dim=1)
                    for idx in range(pred.size(0)):
                        if pred[idx] == y[idx]:
                            pred_result[idx] = 1
                        dis_a[idx] =JSD(dis_or[idx], dis_new[idx])
                    preds[i].append(pred_result)
                diss[i].append(dis_a)

    for i in range(2):
        errors[i] = torch.cat(errors[i], dim=0).cpu().numpy()
        preds[i] = torch.cat(preds[i], dim=0).cpu().numpy()
        diss[i] = torch.cat(diss[i], dim=0).cpu().numpy()

    return base_correct, total, errors, preds, diss

def defense(model, tfp, data, p=2, tfp_errors=None, tfp_diss=None):
    base_correct, total, errors, preds, diss = pre_defense(model, data, p)
    base_acc = base_correct / total
    errors_sort = [[],[]]
    diss_sort = [[],[]]
    if tfp_errors == None or tfp_diss ==None:
        for i in range(2):
            errors_sort[i] = np.sort(errors[i])
            diss_sort[i] = np.sort(diss[i])
        tfp_errors = [[], []]
        tfp_diss = [[],[]]
        #print(errors_sort)
        for i in range(2):
            for rate in tfp:
                tfp_errors[i].append(errors_sort[i][-int(rate*total)])
                tfp_diss[i].append(diss_sort[i][ -int(rate*total)])

    results = [[], []]

    for method_id in range(2):
        # different method
        for tfp_id in range(len(tfp_errors[0])):
            d, e, p, de, dp, ep, dep = 0, 0, 0, 0, 0, 0, 0
            for i in range(total):
                if errors[method_id][i] > tfp_errors[method_id][tfp_id]:
                    e += 1
                if diss[method_id][i] > tfp_diss[method_id][tfp_id]:
                    d += 1
                p += preds[method_id][i]
                if errors[method_id][i] > tfp_errors[method_id][tfp_id] and diss[method_id][i] > tfp_diss[method_id][tfp_id]:
                    de += 1
                if errors[method_id][i] > tfp_errors[method_id][tfp_id] or diss[method_id][i] > tfp_diss[method_id][tfp_id] or preds[method_id][i] == 1:
                    dep += 1
                if diss[method_id][i] > tfp_diss[method_id][tfp_id] and preds[method_id][i] == 1:
                    dp += 1
                if errors[method_id][i] > tfp_errors[method_id][tfp_id] and preds[method_id][i] == 1:
                    ep += 1
            results[method_id].append([d/total, e/total, p/total, de/total, dp/total, ep/total, dep/total])
    return tfp_errors, tfp_diss, results, base_acc
# def defense(model_mtl, tfp, data, p=2, adv=True):
#     model_mtl = to(model_mtl).eval()
#     total = 0
#     base_correct = 0
#     reform_detect = [0, 0]
#
#     for id, (x, y) in enumerate(data):
#         with torch.no_grad():
#             pred_cls, pred_x_id, pred_x_mtl = model_mtl(x)
#             pred = pred_cls.argmax(dim=1)
#             base_correct += pred.eq(y).sum().item()
#
#         for i, pred_x in enumerate([pred_x_id, pred_x_mtl]):
#             error_a = (torch.abs(pred_x - x))**p
#             error = error_a.mean(dim=(1, 2, 3))
#             ae_dt = torch.zeros(error.size(0))
#
#             with torch.no_grad():
#                 pred, _, _ = model_mtl(pred_x)
#                 pred = pred.argmax(dim=1)
#                 total += pred.size(0)
#                 for i in range(error.size(0)):
#                     if error[i] > tfp:
#                         ae_dt[i] = 1
#                         ae_detect[ps] += 1
#                     if pred[i] == y[i]:
#                         reform_detect[ps] += 1
#                     if adv == True:
#                         if(ae_dt[i] == 1 or pred[i] == y[i]):
#                             defensed_correct[ps] += 1
#                     else:
#                         if(ae_dt[i] == 0 and pred[i] == y[i]):
#                             defensed_correct[ps] += 1
#             ps += 1
#     total /= 2
#     return [base_correct / total, defensed_correct[0] / total, ae_detect[0]/total, reform_detect[0]/total,
#             defensed_correct[1]/total, ae_detect[1]/total, reform_detect[1]/total]

def get_attacked_data_loader(datasets, attackmothod, modelname, batch_size=64):
    data_path = './save_attacks/{}_{}.pth'.format(datasets, attackmothod)
    data = torch.load(data_path)
    tensor_dataset = TensorDataset(to(data['x']), to(data['y']))
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

def print_results(results,base_acc, logger, tfp, attackname, modelname, datasetname):
    #print(results)
    method = ['Independent AE', 'MTL AE']
    print('Dataset: {} Modelname: {} Attack: {}'.format(datasetname, modelname, attackname))
    print('No Defense Acc:{:.4f}'.format(base_acc))
    logger.info('Dataset: {} Modelname: {} Attack: {}'.format(datasetname, modelname, attackname))
    logger.info('No Defense Acc:{:.4f}'.format(base_acc))
    for method_id in range(2):
        print(method[method_id])
        logger.info(method[method_id])
        for tfp_id in range(len(tfp)):
            print('Tfp = {:.6f}'.format(tfp[tfp_id]))
            logger.info('Tfp = {:.6f}'.format(tfp[tfp_id]))
            print('Distr:{:.4f}\tAEError:{:.4f}\tReformer:{:.4f}\tDis&AE:{:.4f}\tDis&Reformer:{:.4f}\tAE&Reformer:{:.4f}\tAll:{:.4f}\n'.format(
                results[method_id][tfp_id][0],results[method_id][tfp_id][1],results[method_id][tfp_id][2],results[method_id][tfp_id][3],
                results[method_id][tfp_id][4],results[method_id][tfp_id][5],results[method_id][tfp_id][6]
            ))
            logger.info('Distr:{:.4f}\tAEError:{:.4f}\tReformer:{:.4f}\tDis&AE:{:.4f}\tDis&Reformer:{:.4f}\tAE&Reformer:{:.4f}\tAll:{:.4f}\n'.format(
                results[method_id][tfp_id][0],results[method_id][tfp_id][1],results[method_id][tfp_id][2],results[method_id][tfp_id][3],
                results[method_id][tfp_id][4],results[method_id][tfp_id][5],results[method_id][tfp_id][6]
            ))


def main():
    parser = argparse.ArgumentParser(description='Attacks PyTorch')
    parser.add_argument('--min_value', type=float, default=0.0)
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--tfp', type=float, default=0.005)
    parser.add_argument('--T', type=float, default=2.0)
    args = parser.parse_args()
    setup_run()
    models = {'mnist':MnistMolel, 'fmnist':FMnistMolel, 'cifar10':Cifar10Molel}
    modelname = args.dataset
    datasets = args.dataset
    model = models[modelname](3, args.T, 3 if args.dataset == 'cifar10' else 1)
    model.load_state_dict(torch.load('./train_models/{}.pth'.format(datasets), map_location='cpu'))
    model = to(model).eval()
    model.set_id(3)

    mkdir('./defenses_mtl/')
    log_file = './defenses_mtl/{}_defense_log.txt'.format(datasets)
    logger = get_logger(log_file)

    tfp = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9]

    tfp_errors = None
    tfp_diss = None

    for attacks in ['Noattack', 'FGSM0.3', 'FGSM0.15','JSMA', 'DeepFool', 'CW2_k0']:
        data = get_attacked_data_loader(datasets, attacks, modelname, batch_size=64)
        tfp_errors, tfp_diss, results, base_acc = defense(model,tfp, data, 2, tfp_errors, tfp_diss)
        print_results(results, base_acc, logger, tfp, attacks, modelname, datasets)

if __name__ == '__main__':
    main()