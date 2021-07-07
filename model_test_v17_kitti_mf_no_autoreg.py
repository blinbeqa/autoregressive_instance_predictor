

# system modules
import os, argparse
from datetime import datetime

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distri

import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

# computer vision/image processing modules
import skimage

# math/probability modules
import random
import numpy as np

# custom utilities
from dataloader import MNIST_Dataset, KTH_Dataset, KITTI_Dataset 
from convlstmnet import ConvLSTMNet
from contextvpnet_hyb import ContextVPNet
from mcnet import MCNet

import matplotlib.image as imgblin
# from model_v16_res_2dir_lstm_dec import NonLocalLSTM
#from model_v16_down_lstm_up import NonLocalLSTM
# from model_v16_res18_lstm_dec import NonLocalLSTM
# from model_v16_res_lstm_dec import NonLocalLSTM
from model_v17_res50_lstm_mf_no_autoreg import NonLocalLSTM
#from model_v17_res50_lstm import NonLocalLSTM
from kitti_raw6 import ImageFolder1
from focal_loss import FocalLoss
from youtube_vos_loader_v3 import ImageFolderYVos

from torchvision.datasets import UCF101
from iou_metric import iou_pytorch, iou_numpy, iou_pytorch_no_mean

import matplotlib.image as imgblin
import skimage
from PIL import Image
from stemseg.data.mots_data_loader import MOTSDataLoader
from stemseg.data.mots_data_loader_test import MOTSDataLoaderTest
from stemseg.data.mots_data_loader_multi_ins import MOTSDataLoaderMultiIns
from stemseg.data.mapillary_data_loader import MapillaryDataLoader
from stemseg.data.common import collate_fn
from vision_common_utils_image_utils import overlay_mask_on_image
from ap import APMetric


# perceptive quality
import PerceptualSimilarity.models as PSmodels

def main(args):
    ## Model preparation (Conv-LSTM or Conv-TT-LSTM or ContextVP)

    ## Distributed setup

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '50200'
    mp.spawn(testing, nprocs=args.gpus, args=(args,))

def testing(gpu, args):


    rank = args.nr * args.gpus + gpu
    distri.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # whether to use GPU (or CPU)
    device = torch.device("cuda")

    # construct the network with the specified hyper-parameters
    if args.network == "contextvp":  # contextVP

        # print("contextvpbabayyy")

        model = ContextVPNet(
            # input to the model
            input_channels=args.img_channels,
            # architecture of the model
            blending_block_channels=(32, 64, 64, 32),
            hidden_channels=(32, 64, 64, 32),
            skip_stride=2,
            # parameters of PMD convolutional layers
            cell=args.model, cell_params={"order": args.model_order,
                                          "steps": args.model_steps, "rank": args.model_rank},
            # parameters of convolutional operations
            kernel_size=args.kernel_size, bias=True,
            # output function and output format
            output_sigmoid=args.use_sigmoid)

    elif args.network == "lstm":  # convolutional tensor-train LSTM

        model = ConvLSTMNet(
            # input to the model
            input_channels=args.img_channels,
            # architecture of the model
            layers_per_block=(3, 3, 3, 3),
            hidden_channels=(32, 48, 48, 32),
            skip_stride=2,
            # parameters of convolutional tensor-train layers
            cell=args.model, cell_params={"order": args.model_order,
                                          "steps": args.model_steps, "rank": args.model_rank},
            # parameters of convolutional operations
            kernel_size=args.kernel_size, bias=True,
            # output function and output format
            output_sigmoid=args.use_sigmoid)
    elif args.network == "nllstm":  # convolutional tensor-train LSTM

        print("nllstm test")
        model = NonLocalLSTM(
            # input to the model
            input_channels=args.img_channels,
            # architecture of the model
            blending_block_channels=(128, 128, 128, 128, 128, 256, 256, 256, 128, 128, 128, 128, 128, 128),
            hidden_channels=(128, 128, 128, 128, 128, 256, 256, 256, 128, 128, 128, 128, 128, 128),
            output_frames=args.output_frames,
            # blending_block_channels=(256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128),
            # hidden_channels=(256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128),
            # blending_block_channels=(64, 64, 128, 128, 128, 256, 256, 256, 256, 128, 128, 128, 64, 64),
            # hidden_channels=(64, 64, 128, 128, 128, 256, 256, 256, 256, 128, 128, 128, 64, 64),
            # blending_block_channels=(256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128),
            # hidden_channels=(256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128),
            skip_stride=2,
            # parameters of PMD convolutional layers
            cell=args.model, cell_params={"order": args.model_order,
                                          "steps": args.model_steps, "rank": args.model_rank},
            # parameters of convolutional operations
            kernel_size=args.kernel_size, bias=True, res_select=args.resselect, autoreg_select=args.autoregsel, number_of_instances=args.number_of_instances,
            # output function and output format
            output_sigmoid=args.use_sigmoid)



    else:
        raise NotImplementedError

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # model in Distributed Data Paralel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    # load the model parameters from checkpoint
    #checkpoint = torch.load(args.checkpoint)
    #model.load_state_dict(torch.load(args.checkpoint, map_location="cuda:0")['state_dict'])
    checkpoint_path = "/globalwork/beqa/new_runs/" + str(args.run_folder_name) + "/checkpoint_contextvp_dist_" + str(args.run_folder_name) + "_ep_" + str(args.test_epoch) + "_.pt"
    print("path", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0"))

    ## Dataset Preparation (Moving-MNIST, KTH)
    Dataset = \
    {"MNIST": MNIST_Dataset, "KTH": KTH_Dataset, "UCF-101": UCF101, "KITTI": KITTI_Dataset, "KITTI_RAW": ImageFolder1,
     "YTV": ImageFolderYVos}[args.dataset]

    DATA_DIR = os.path.join("/home/beqa/PycharmProjects/conv-tt-lstm/conv-tt-lstm-master/code/datasets",
                            {"MNIST": "moving-mnist", "KTH": "kth", "UCF-101": "UCF-101", "KITTI": "kitti_hkl",
                             "KITTI_RAW": "kitti_hkl", "YTV": "kitti_hkl"}[args.dataset])

    # number of total frames
    total_frames = args.input_frames + args.future_frames

    if args.dataset == "KHT" or args.dataset == "MNIST":

        # dataloaer for test set
        test_data_path = os.path.join(DATA_DIR, args.test_data_file)

        test_data = Dataset({"path": test_data_path, "unique_mode": True,
                                "num_frames": total_frames, "num_samples": args.test_samples,
                                "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)

        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler,
                                                        shuffle=False, num_workers=0, drop_last=True)

        test_size = len(test_data_loader) * args.batch_size
    
    elif args.dataset == "KITTI":

         # dataloaer for the testing set

        test_file = os.path.join(DATA_DIR, 'X_test.hkl')
        test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

        test_data = Dataset(test_file, test_sources, total_frames)

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)


        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, sampler=test_sampler,
                                                        shuffle=False, num_workers=0, drop_last=True)

        test_size = len(test_data_loader) * args.batch_size


    elif args.dataset == "KITTI_RAW":

        # DATA_DIR = os.path.join("/globalwork/datasets/KITTI_MOTS",
        #                       "train")
        #
        # train_transforms = transforms.Compose(
        # [transforms.Resize(256),
        #  transforms.CenterCrop([256, 512]),
        #  transforms.ToTensor(),
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                       std=[0.229, 0.224, 0.225])
        # ])
        #
        # mask_transforms = transforms.Compose(
        #     [
        #      transforms.Resize(256),
        #      transforms.CenterCrop([256, 512]),
        #      transforms.ToTensor()
        #
        #      ])
        #
        # dataset_train = ImageFolder1(DATA_DIR, total_frames, "train", transform=train_transforms, transform_masks = mask_transforms)
        #
        # dataset_val = ImageFolder1(DATA_DIR, total_frames, "val", transform=train_transforms, transform_masks = mask_transforms)
        DATA_DIR = os.path.join("/globalwork/datasets/KITTI_MOTS",
                                "train/images")
        JSON_DIR_TRAIN = os.path.join("/home/beqa/PycharmProjects/conv-tt-lstm/conv-tt-lstm-master/code/stemseg",
                                      "kittimots_train_blin.json")
        JSON_DIR_VAL = os.path.join("/home/beqa/PycharmProjects/conv-tt-lstm/conv-tt-lstm-master/code/stemseg",
                                    "kittimots_val.json")


        dataset_train = MOTSDataLoader(DATA_DIR, JSON_DIR_TRAIN, samples_to_create=4000, total_frames=total_frames, id_to_remove=args.inssel)



        print("length", len(dataset_train))


        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)

        train_data_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True,
                                                        sampler=train_sampler,
                                                        shuffle=False, num_workers=0, drop_last=True)

        train_size = len(train_data_loader) * args.batch_size

        # dataloaer for the valiation set
        torch.manual_seed(0)
        random.seed(0)
        if args.number_of_instances > 1:    
            dataset_val = MOTSDataLoaderMultiIns(DATA_DIR, JSON_DIR_VAL, samples_to_create=800, total_frames=total_frames, id_to_remove=args.inssel)
        else:
            dataset_val = MOTSDataLoader(DATA_DIR, JSON_DIR_VAL, samples_to_create=800, total_frames=total_frames, id_to_remove=args.inssel)

            # dataset_val = MOTSDataLoaderTest(DATA_DIR, JSON_DIR_VAL, samples_to_create=700, total_frames=total_frames, id_to_remove=args.inssel)
            # print("test all")
            
        print("length", len(dataset_val))
        # dataset_val = MOTSDataLoader(DATA_DIR, JSON_DIR_VAL, samples_to_create=800, total_frames=total_frames, id_to_remove=args.inssel)
        # print("length", len(dataset_val))

        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)

        test_data_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True,
                                                        sampler=test_sampler,
                                                        shuffle=False, num_workers=0, drop_last=True)

        test_size = len(test_data_loader) * args.batch_size


    elif args.dataset == "YTV":

        DATA_DIR = os.path.join("/globalwork/datasets/youtube-vos",
                                "train")
        DATA_DIR_VAL = os.path.join("/globalwork/datasets/youtube-vos",
                                    "valid")

        train_transforms = transforms.Compose(
            [transforms.Resize(320),
             transforms.CenterCrop([320, 640]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
             ])

        mask_transforms = transforms.Compose(
            [transforms.Resize(320, interpolation=Image.NEAREST),
             transforms.CenterCrop([320, 640]),
             transforms.ToTensor()
             ])

        dataset_train_temp = ImageFolderYVos(DATA_DIR, total_frames, transform=train_transforms,
                                             transform_masks=mask_transforms)
        # torch.manual_seed(0)
        # dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_temp, [10700, 2672])
        temp_tr_length = int(len(dataset_train_temp) * 0.8)
        torch.manual_seed(0)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_temp, [temp_tr_length, len(dataset_train_temp) - temp_tr_length])

        # dataset_val = ImageFolderYVos(DATA_DIR_VAL, total_frames, transform=train_transforms,
        #                            transform_masks=mask_transforms)

        print("length", len(dataset_train))
        print("length", len(dataset_val))

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)

        train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, pin_memory=True,
                                                        sampler=train_sampler,
                                                        shuffle=False, num_workers=0, drop_last=True)

        train_size = len(train_data_loader) * args.batch_size

        # dataloaer for the valiation set

        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)

        test_data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True,
                                                        sampler=valid_sampler,
                                                        shuffle=False, num_workers=0, drop_last=True)

        test_size = len(test_data_loader) * args.batch_size

    ## Main script for test phase
    model.eval()

    MSE = np.zeros(args.future_frames, dtype=np.float32)
    PSNR = np.zeros(args.future_frames, dtype=np.float32)
    SSIM = np.zeros(args.future_frames, dtype=np.float32)
    PIPS = np.zeros(args.future_frames, dtype=np.float32)
    IOU = np.zeros(args.future_frames, dtype=np.float32)
    ap50 = APMetric(threshold=0.9)

    PSmodel = PSmodels.PerceptualLoss(model='net-lin',
                                      net='alex', use_gpu=True, gpu_ids=[0])

    sig = nn.Sigmoid()

    path_for_images = '/globalwork/beqa/new_runs/run16/fotos_color_bw'
    iou1 = [None]
    iou2 = [None]
    iou3 = [None]
    # path_for_images = '/globalwork/beqa/new_runs/eval_thes_pic/ytv_thes'
    with torch.no_grad():

        counter = 0

        white_pixels = 0


        for frames in test_data_loader:

            if args.number_of_instances == 1:

                #print("origshape1", frames[0].size())
                frames = frames[0].to(device)

                inputs = frames[:, :]
                origin = frames[:, -args.output_frames:]

                
                accmulated_loss_bce = 0.0
                accmulated_loss_mse = 0.0

                input_block = inputs[:, :]
                input_block[:, :, 3:4] = (input_block[:, :, 3:4] > 0.5).int()
        
            else:
                
                if args.number_of_instances <= (frames[0].size()[2] - 3):
                    frames = frames[0][:,:,0: 3 + args.number_of_instances].to(device)
                    #print("origshape2", frames.size())
                    inputs = frames[:, :]
                    origin = frames[:, -args.output_frames:]
                    accmulated_loss_bce = 0.0
                    accmulated_loss_mse = 0.0

                    input_block = inputs[:, :]
                    input_block[:, :, 3:3 + args.number_of_instances] = (input_block[:, :, 3:3 + args.number_of_instances] > 0.5).int()
                else:
                    diff_of_instances = args.number_of_instances - (frames[0].size()[2] - 3)
                    fill_dim = torch.empty(frames[0].size()[0], frames[0].size()[1], diff_of_instances, frames[0].size()[3], frames[0].size()[4])
                    fill_input_gap = torch.zeros_like(fill_dim)
                    frames = torch.cat([frames[0][:,:,:], fill_input_gap], dim=2).to(device)
                    #print("origshape3", frames.size())
                    inputs = frames[:, :]
                    origin = frames[:, -args.output_frames:]
                    accmulated_loss_bce = 0.0
                    accmulated_loss_mse = 0.0

                    input_block = inputs[:, :]
                    input_block[:, :, 3:3 + args.number_of_instances] = (input_block[:, :, 3:3 + args.number_of_instances] > 0.5).int()


            pred, tar, pre = model(input_block,
                         input_frames=args.input_frames,
                         future_frames=args.future_frames,
                         output_frames=args.output_frames,
                         teacher_forcing=True
                         )
            pred = sig(pred)
            pred = (pred > 0.5).float()
            means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, None, :, None, None]  # [1, 3, 1, 1]
            scales = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, None, :, None, None]  # [1. 3. 1. 1]
            origin_temp = (origin[:,:,0:3].cpu() * scales) + means
            orig_rgb_img = torch.clamp(origin_temp.permute(0, 1, 3, 4, 2), min=0, max=1).numpy()*255
            origin = origin[:,:,3:3 + args.number_of_instances]
            # pred = (frames[:, 7:8, 3:4] > 0.5).float()
            # pred = pred.repeat([1, 3, 1, 1, 1])

            pred2_temp = (frames[:, :, 0:3].cpu() * scales) + means
            pred2 = (frames[:, :, 3:4] > 0.5).float()
            pred2rgb = torch.clamp(pred2_temp.permute(0, 1, 3, 4, 2), min=0, max=1).numpy()*255

            # clamp the output to [0, 1]

            # pred = torch.clamp(pred, min=0, max=1)
            # pred = (pred > 0.5).float()
            # print("debug")
            #print("pshape", pred.shape)

            # accumlate the statistics per frame
            for t in range(args.future_frames):
                for i in range(args.number_of_instances):
                    origin_, pred_ = origin[:, t], pred[:, t]
                    o_int = origin_.to(torch.int)
                    p_int = (pred_ > 0.5).float().to(torch.int)
                    # print("p", p_int.size())
                    # print("o", o_int.size())
                    # for b in range(args.batch_size):
                    #     ap50.add_score(score=iou_pytorch_no_mean(p_int[:, i], o_int[:, i])[b], pred_bool=(p_int[b, i].max()>0.5).bool(), gt_bool=(o_int[b, i].max()>0.5).bool() )
                    #     print("AP", ap50.return_precision())
                    #     print("AR", ap50.return_recall())
                    #print("iou no mean", iou_pytorch_no_mean(p_int[:, i], o_int[:, i]))
                    IOU[t] += iou_pytorch(p_int[:, i], o_int[:, i])/(len(test_data_loader)*args.number_of_instances)
                    print("IOU", IOU[t])
                    if t == 0:
                        iou1.append(iou_pytorch(p_int[:, i], o_int[:, i]).cpu().numpy().item())
                    if t == 1:
                        iou2.append(iou_pytorch(p_int[:, i], o_int[:, i]).cpu().numpy().item())
                    if t == 2:
                        iou3.append(iou_pytorch(p_int[:, i], o_int[:, i]).cpu().numpy().item())
                    # print("AP", ap50.return_precision())
                    # print("AR", ap50.return_recall())


                # dist = PSmodel(origin_, pred_)
                # PIPS[t] += torch.sum(dist).item() / test_size
            for j in range(args.number_of_instances):
                origin1 = origin[:,:, j: j+1].repeat([1, 1, 1, 1, 1])
                pred1 = pred[:,:, j: j+1].repeat([1, 1, 1, 1, 1])
                origin1 = origin1.permute(0, 1, 3, 4, 2).cpu().numpy()
                pred1 = pred1.permute(0, 1, 3, 4, 2).cpu().numpy()

                pred2_2 = pred2[:,:, j: j+1].repeat([1, 1, 1, 1, 1])
                pred2_2 = pred2_2.permute(0, 1, 3, 4, 2).cpu().numpy()

                # for b in range(args.batch_size):
                #
                #     white_pixels += np.sum(origin[b, 0, :, :, 0]) / test_size
                #     print("wp ", white_pixels)
                

                for t in range(args.future_frames+args.input_frames):
                    for i in range(args.batch_size):
                        # origin_, pred_, orig_rgb_img_ = origin1[i, t], pred1[i, t], orig_rgb_img[i, t]
                        if(t>args.input_frames-1):
                            origin_, pred_, orig_rgb_img_ = origin1[i, t-args.input_frames], pred1[i, t-args.input_frames], orig_rgb_img[i, t-args.input_frames]
                        if args.img_channels == 4 and counter % 5 == 0:

                            blini_b=5
                            #print(pred_.shape)
                            #print(origin_.shape)
                            #print(orig_rgb_img_.max())



                            # origin_ = np.squeeze(origin_, axis=-1)
                            # pred_ = np.squeeze(pred_, axis=-1)
                            # print("origin_")
                            # print(origin_[50, 50, 0])
                            # print(origin_[50, 50, 1])
                            # print(origin_[50, 50, 2])
                            # print("pred")
                            # print(pred_[50, 50, 0])
                            # print(pred_[50, 50, 1])
                            # print(pred_[50, 50, 2])

                            # imgblin.imsave(
                            #     path_for_images + '/' + str(counter) + '_i' + '_' + str(i) + '_' + str(j) + '_' + str(t) + '_p.png', overlay_mask_on_image(orig_rgb_img_, pred_,mask_color=(0, 255, 0)))
                            # imgblin.imsave(
                            #     path_for_images + '/' + str(counter) + '_i' + '_' + str(i) + '_' + str(j) + '_' + str(t) + '_o.png',
                            #     overlay_mask_on_image(orig_rgb_img_, origin_, mask_color=(0, 255, 0)))
                        # print("orgin", origin_.shape)
                        # MSE[t] += skimage.measure.compare_mse(origin_, pred_) / (test_size* args.number_of_instances)

                        # PSNR[t] += skimage.measure.compare_psnr(origin_, pred_) / test_size
                        # SSIM[t] += skimage.measure.compare_ssim(origin_, pred_,
                        #                                         multichannel=(args.img_channels > 1)) / test_size

            counter += 1


    print("MSE: {} (x1e-3); PSNR: {}, SSIM: {}, LPIPS: {}, IOU: {}".format(
            1e3 * np.mean(MSE), np.mean(PSNR), np.mean(SSIM), np.mean(PIPS), np.mean(IOU)))

    print("MSE:", MSE)
    print("IOU:", IOU)
    print("PSNR:", PSNR)
    print("SSIM:", SSIM)
    print("PIPS:", PIPS)
    
    new_list = [x for x in iou1 if isinstance(x, (int, float))]
    new_list2 = [x for x in iou2 if isinstance(x, (int, float))]
    new_list3 = [x for x in iou3 if isinstance(x, (int, float))]
    iou1r = np.sort(np.array(new_list))[::-1]
    iou2r = np.sort(np.array(new_list2))[::-1]
    iou3r = np.sort(np.array(new_list3))[::-1]
    # iou2r = iou2.sort(reverse=True)
    # iou3r = iou3.sort(reverse=True)
    print("iou1", sum(iou1r[:200])/200)
    print("iou2", sum(iou2r[:200])/200)
    print("iou3", sum(iou3r[:200])/200)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Test")

    ## Data format (batch_size x time_steps x height x width x channels)

    ## Distributed parameters
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-epo', '--epoch-num', dest='test_epoch', default='19', type=str,
                        help='Name of folder to store results of run')
    parser.add_argument('-run', '--run', dest='run_folder_name', default='run71', type=str,
                        help='Name of folder to store results of run')
    parser.add_argument('-all', '--all_test', dest='all_test_param', default=0, type=int,
                        help='ranking within the nodes')
    ## Cluster select
    parser.add_argument('-cl', '--cluster', dest='cluster', default='vision', type=str,
                        help='Cluster options: vision, claix')

    parser.add_argument('-rsel', '--res-select', dest='resselect', default='res18', type=str,
                        help='Res select: res18, res34, res50')

    parser.add_argument('-bcew', '--bce-weight', dest='bcew', default=3.0, type=float,
                        help='BCE Weight')
    parser.add_argument('-freeze', '--freeze-weight', dest='freeze_weight', default=True, type=bool,
                        help='Freeze Weight')

    parser.add_argument('-opt', '--opt-level', dest='optlevel', default='O0', type=str,
                        help='Opt-level Nvidia-Apex: O0, O1, O2, O3')

    parser.add_argument('-alf', '--loss-alfa', dest='alfa', default=1.0, type=float,
                        help='Loss alfa: 0.0 - 1.0')

    parser.add_argument('-inssel', '--instance-select', dest='inssel', default=2, type=int,
                        help='Select instance to remove: 1 for cars, 2 for people')

    parser.add_argument('-lsel', '--loss-sel', dest='loss_select', default='bce', type=str,
                        help='Loss select: bce, gdl')

    parser.add_argument('-bcedec', '--bce-decay', dest='bce_dec', default=False, type=bool,
                        help='Loss alfa: 0.0 - 1.0')

    parser.add_argument('-bcedecrate', '--bce-decay-rate', dest='bce_dec_rate', default=0.95, type=float,
                        help='Loss alfa: 0.0 - 1.0')
    parser.add_argument('-focalp', '--focal-alpha', dest='focal_alpha', default=1.0, type=float,
                        help='Loss alfa: 0.0 - 1.0')
    parser.add_argument('-focgma', '--focal-gamma', dest='focal_gamma', default=2.0, type=float,
                        help='Loss alfa: 0.0 - 1.0')

    parser.add_argument('-numins', '--number-of-instances', dest='number_of_instances', default=1, type=int,
                        help='Select number of paralel instances to be prcessed')
    parser.add_argument('-autoreg', '--autoreg-set', dest = 'autoregsel', action = 'store_true',
        help='Freeze Weight')
    parser.add_argument('-no-autoreg', '--no-autoreg-set', dest = 'autoregsel', action = 'store_false',
        help='Freeze Weight')
    parser.set_defaults(autoregsel = True)


    # batch size and the logging period 
    parser.add_argument('--batch-size',  default = 1, type = int,
        help = 'The batch size in training phase.')

    # frame split
    parser.add_argument('--input-frames',  default = 8, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 2, type = int,
        help = 'The number of predicted frames of the model.')
    parser.add_argument('--output-frames', default=2, type=int,
                        help='The number of output frames of the model.')
    
    # frame format h= 128 w = 160 kitti
    parser.add_argument('--img-height',  default = 320, type = int,
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 640, type = int,
        help = 'The image width of each video frame.')
    parser.add_argument('--img-channels', default = 4, type = int,
        help = 'The number of channels in each video frame.')

    ## Devices (CPU, single-GPU or multi-GPU)

    # whether to use GPU for testing
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for testing.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Use CPU for testing.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for testing (given GPU is used)
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for testing.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Use single GPU for testing.')
    parser.set_defaults(multi_gpu = True)

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model name (with time stamp as suffix)
    parser.add_argument('--checkpoint', default = "/globalwork/beqa/new_runs/run12/checkpoint_contextvp_dist_run12_ep_10_.pt", type = str,
        help = 'The name for the checkpoint.')

    # model type and size (depth and width) 
    parser.add_argument('--model', default = 'nllstm', type = str,
        help = 'The model is either \"convlstm\"" or \"convttlstm\".')

    parser.add_argument('--network', default='nllstm', type=str,
                        help='The network model is either \"contextvp\"" or \"lstm\".')

    parser.add_argument('--use-sigmoid', dest = 'use_sigmoid', action = 'store_true',
        help = 'Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid',  dest = 'use_sigmoid', action = 'store_false',
        help = 'Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid = False)

    # parameters of the convolutional tensor-train layers
    parser.add_argument('--model-order', default = 3, type = int, 
        help = 'The order of the convolutional tensor-train LSTMs.')
    parser.add_argument('--model-steps', default = 3, type = int, 
        help = 'The steps of the convolutional tensor-train LSTMs')
    parser.add_argument('--model-rank',  default = 8, type = int, 
        help = 'The tensor rank of the convolutional tensor-train LSTMs.')
    
    # parameters of the convolutional operations
    parser.add_argument('--kernel-size', default = 3, type = int, 
        help = "The kernel size of the convolutional operations.")

    ## Dataset (Input)
    parser.add_argument('--dataset', default = "KITTI_RAW", type = str,
        help = 'The dataset name. (Options: MNIST, KTH, KITTI, KITTI_RAW,YTV)')
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    parser.add_argument('--test-data-file', default = 'mnist_test_cut_seq.npy', type = str,
        help = 'Name of the folder/file for test set.')
    parser.add_argument('--test-samples', default = 1000, type = int,
        help = 'Number of unique samples in test set.')

    main(parser.parse_args())
