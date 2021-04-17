import argparse
import sys
import os
import pickle
import numpy as np 
import pickle, gzip
import shutil
import logging
from collections import OrderedDict
# from tabulate import tabulate

import torch
import torch.nn.functional as F 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter 
import time


from loader import MyDataLoader
from model import SuperResolution
from model import TestMeshShuffle

best_loss = 100000

def getPSNRLoss():
  mseloss_fn = nn.MSELoss(reduction='none')

  def PSNRLoss(output, target):
    loss = mseloss_fn(output, target)
    loss = torch.mean(loss, dim=(1,2))
    loss = 10 * torch.log10(loss)
    mean = torch.mean(loss)
    return mean

  return PSNRLoss

loss_function = getPSNRLoss()

def cal_ssim(im1,im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    im1 = im1.detach().numpy()
    im2 = im2.detach().numpy()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 1
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim

def cal_batch_ssim(output, target):
  ssim = 0
  for i in range(output.size()[0]):
    tmp_ssim = cal_ssim(output[i], target[i])
    ssim += tmp_ssim
  return ssim.mean()


def train(args, model, train_loader, optimizer, epoch, device):
  model.train()
  tot_loss = 0.
  tot_ssim = 0.
  count = 0.
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #  output = model(data)
    #prof.export_chrome_trace('trace')

    # np.save('train_res.npy',output)
    loss = loss_function(output, target)
    ssim = cal_batch_ssim(output, target)
    # print(ssim)
    tot_ssim += ssim * data.size()[0]

    loss.backward()
    optimizer.step()
    tot_loss += loss.item() * data.size()[0]
    count += data.size()[0]

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

  tot_loss /= count
  tot_ssim /= count
  print('Train Epoch: {} Loss: {:.6f}'.format(epoch, tot_loss))
  print('Train Epoch: {} ssim: {:.6f}'.format(epoch, tot_ssim))
  return tot_loss

def test(args, model, test_loader, epoch, device):
  global best_loss
  model.eval()
  test_loss = 0
  count = 0
  best = 100
  tot_ssim = 0
  i = 0
  if not os.path.isdir('out/'):
    os.mkdir('out/')

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      data, target = data.to(device), target.to(device)
      output = model(data)

      name = 'out/r'+str(i)+'.npy'
      np.save(name,output)
      i += 1
      loss = loss_function(output, target)
      ssim = cal_batch_ssim(output, target)
      # print(ssim)
      tot_ssim += ssim * data.size()[0]

      test_loss += loss.item() * data.size()[0]
      count += data.size()[0]

    test_loss /= count
    tot_ssim /= count
    print('Test Epoch: {} Loss: {:.6f}'.format(epoch, test_loss))
    print('Test Epoch: {} ssim: {:.6f}'.format(epoch, tot_ssim))
  return test_loss


def isnotin(it):
  buffer = ['G','F2V','L','NS','EW','unique','separated_src_idx', 'NS_grad_op', 'EW_grad_op', 'unique_mat', 'all_grad_ops']
  for b in buffer:
    if b in it.split('.'):
      return False
  return True


def main():
  parser = argparse.ArgumentParser(description='Spherical Super Resolution')
  parser.add_argument('--model_idx',type=int,default=0,metavar='N',
    help= 'model index')
  parser.add_argument('--batch-size', type = int, default = 64, metavar = 'N',
    help = 'input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type = int, default = 64, metavar = 'N',
    help = 'input batch size for testing (default: 64)')
  parser.add_argument('--epochs', type = int, default = 100, metavar = 'N',
    help = 'number of epochs to train (default: 100)')
  parser.add_argument('--lr', type = float, default = 1e-2,metavar = 'LR',
    help = 'learning rate (default: 0.01')
  parser.add_argument('--no-cuda', action = 'store_true', default = False,
    help = 'disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
  parser.add_argument('--mesh_folder', type=str, default="mesh_files",
        help='path to mesh folder (default: mesh_files)')
  parser.add_argument('--max_level', type=int, default=9, help='max mesh level')
  parser.add_argument('--min_level', type=int, default=7, help='min mesh level')
  parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status')
  parser.add_argument('--feat', type=int, default=16, help='filter dimensions')
  parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
  parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
  parser.add_argument('--in_ch', type=str, default="rgb", choices=["rgb", "rgbd"], help="input channels")

  parser.add_argument('--train_data_folder', type=str, default="train_data",
        help='path to data folder (default: train_data)')
  parser.add_argument('--test_data_folder', type=str, default="test_data",
        help='path to data folder (default: test_data)')
  parser.add_argument('--load', type = int, default = 1,
    help='Load model or not')


  args = parser.parse_args()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device)

  torch.manual_seed(args.seed)

  # trainset = MyDataLoader(args.train_data_folder, sp_level = args.max_level, in_ch=len(args.in_ch))
  testset = MyDataLoader(args.test_data_folder, sp_level = args.max_level, in_ch = len(args.in_ch))

  # train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
  test_loader = DataLoader(testset, batch_size = args.test_batch_size, shuffle = True)

  model = SuperResolution(mesh_folder=args.mesh_folder, in_ch=len(args.in_ch), out_ch=len(args.in_ch), \
                          max_level=args.max_level, min_level=args.min_level, fdim=args.feat)
  #model = TestMeshShuffle(mesh_folder=args.mesh_folder, in_ch=len(args.in_ch), out_ch=len(args.in_ch), \
  #                        max_level=args.max_level, min_level=args.min_level, fdim=args.feat)

  # for multiple GPU use
  model = nn.DataParallel(model)
  if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
  else:
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

  if args.decay:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma = 0.9)

  start_epoch = 0
  best_loss = 100

  if args.load == 1:
    print('------------Load Model-----------------')
    # assert os.path.isdir('checkpoint')
    checkpoint = torch.load('./checkpoint/Model0')
    # start_epoch = checkpoint['epoch']
    # best_loss = checkpoint['best_loss']
    state = checkpoint['state_dict']
    
    def load_my_state_dic(self, state_dict, exclude = 'none'):
      own_state = self.state_dict()
      for name, param in state_dict.items():
        if name not in own_state:
          continue
        if exclude in name:
          continue
        if isinstance(param, Parameter):
          param = param.data
        own_state[name].copy_(param)

    load_my_state_dic(model, state)

  print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

  # for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
    # start_time = time.time()
    # train_loss = train(args, model, train_loader, optimizer, epoch, device)
    # if args.decay:
    #   scheduler.step()
    # end_time = time.time()
    # period = end_time-start_time
    # print('Train Time for each epoch: {}'.format(period))

  start_time = time.time()
  test_loss = test(args, model, test_loader, epoch, device)
  end_time = time.time()
  period = end_time-start_time
  print('Test Time for each epoch: {}'.format(period))

    # state_dict_no_buffer = [it for it in model.state_dict().items() if isnotin(it[0])]
    # state_dict_no_buffer = OrderedDict(state_dict_no_buffer)

    # if test_loss < best_loss:
    #   print('---------------------Save Model------------------')
    #   state = {
    #   'state_dict': state_dict_no_buffer,
    #   # 'epoch': epoch,
    #   'best_loss': test_loss,
    #   # 'optimizer': optimizer.state_dict(),
    #   }
    #   if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')

    #   torch.save(state, './checkpoint/Model'+str(args.model_idx))
    #   # torch.save(model, 'FullModel.pkl')
    #   best_loss = test_loss

if __name__ == '__main__':
  main()
