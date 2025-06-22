import torch
import pytorch_lightning as pl
import argparse
from model import DNNCS, patchify_tensor, recompose_tensor
import torch.nn.functional as F
from dataset import *
import torch.utils.data as data
import numpy as np
import os
import cv2
import time
from pytorch_msssim import ssim
import piq


def psnr_func(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def eval_metric(SR1, SR2, HR1, HR2):
    u, v, eta = SR1.chunk(3, dim=1)
    SR1 = torch.cat((u, v, eta), dim=3)
    u, v, eta = SR2.chunk(3, dim=1)
    SR2 = torch.cat((u, v, eta), dim=3)
    u, v, eta = HR1.chunk(3, dim=1)
    HR1 = torch.cat((u, v, eta), dim=3)
    u, v, eta = HR2.chunk(3, dim=1)
    HR2 = torch.cat((u, v, eta), dim=3)

    rmse = torch.pow(F.mse_loss(SR1, HR1), 0.5) + torch.pow(F.mse_loss(SR2, HR2), 0.5)
    mae = F.l1_loss(SR1, HR1) + F.l1_loss(SR2, HR2)

    psnr_index = (psnr_func(SR1*255, HR1*255) + psnr_func(SR2*255, HR2*255)) * 0.5
    ssim_index = (ssim(SR1, HR1, data_range=1, size_average=True) + ssim(SR2, HR2, data_range=1, size_average=True)) * 0.5
    gmsd_index = (piq.gmsd(SR1, HR1, data_range=1., reduction='none') + piq.gmsd(SR2, HR2, data_range=1., reduction='none')) * 0.5
    lpips_loss = (piq.LPIPS(reduction='none')(SR1, HR1) + piq.LPIPS(reduction='none')(SR2, HR2)) * 0.5
    # dists_loss = (piq.DISTS(reduction='none')(SR1, HR1) + piq.DISTS(reduction='none')(SR2, HR2)) * 0.5

    return rmse, mae, psnr_index, ssim_index, gmsd_index, lpips_loss
    

def heat_vis(SR1, SR2, mask):
    u1 = cv2.applyColorMap(SR1[:,:,0:1], cv2.COLORMAP_HOT)
    u2 = cv2.applyColorMap(SR2[:,:,0:1], cv2.COLORMAP_HOT)
    v1 = cv2.applyColorMap(SR1[:,:,0:1], cv2.COLORMAP_DEEPGREEN)
    v2 = cv2.applyColorMap(SR2[:,:,0:1], cv2.COLORMAP_DEEPGREEN)
    eta1 = cv2.applyColorMap(SR1[:,:,0:1], cv2.COLORMAP_OCEAN)
    eta2 = cv2.applyColorMap(SR2[:,:,0:1], cv2.COLORMAP_OCEAN)

    SR1 = np.concatenate((u1, v1, eta1), axis=1)
    SR2 = np.concatenate((u2, v2, eta2), axis=1)
    SR1 = 255 - SR1 
    SR2 = 255 - SR2 

    return SR1, SR2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Overall upsampling parameters
    parser.add_argument("--checkpoint", type=str, default='DNNCS_best.ckpt')
    parser.add_argument('--test_patch_size', type=int, default=64, help='input LR patch size')
    parser.add_argument('--stride', type=int, default=4, help='overlapping size')
    parser.add_argument("--up_ratio", type=int, default=1, help="upsampling factor")
    # validation settings
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--save_hr', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='/data/shallow_water/Pred_DNNCS')
    parser.add_argument('--validate_data', type=str, default='/data/shallow_water/demo_data')
    parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
    parser.add_argument("--test_batchSize", type=int, default=1, help="testing batch size")
    
    # ################ PREPARATIONS #################
    opt = parser.parse_args()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DNNCS(dim=64).to(DEVICE)
    model_path = os.path.join('ckpt', opt.checkpoint)
    pretrained_model = torch.load(model_path, map_location=lambda storage, loc: storage)
    checkpoint = {
        k[6:]: v
        for k, v in pretrained_model['state_dict'].items()
    }
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    sub_folder = ['bahamas/1696', 'bahamas/6784', 'bahamas/27136']
    # sub_folder = ['galveston/3397', 'galveston/13000', 'galveston/54000']
    for idx in range(2):
        folder = join(opt.validate_data, sub_folder[idx])
        folder_name = sub_folder[idx+1].split('/')[0]
        val_set = DatasetFromFolderTest_demo(
            lr_folder=sub_folder[idx],
            hr_folder=sub_folder[idx+1],
            data_dir=opt.validate_data,
            patch_size=opt.test_patch_size, 
            upscale_factor=opt.up_ratio,
            file_idx = idx
        )
        test_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.test_batchSize, shuffle=False, num_workers=opt.threads)
        validation_rmse = 0
        validation_mae = 0
        validation_psnr = 0
        validation_ssim = 0
        validation_gmsd = 0
        validation_lpips = 0
        count = 0
        st = time.time()
        dummy_time = 0
        for i, data in enumerate(test_dataloader):
            LR1, LR2, HR1, HR2, HR3, mask, coord, file_name1, file_name2 = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]
            batch_size, channels, img_height, img_width = LR1.size()
            LR1 = LR1.to(DEVICE)
            LR2 = LR2.to(DEVICE)
            HR1 = HR1.to(DEVICE)
            HR2 = HR2.to(DEVICE)
            HR3 = HR3.to(DEVICE)
            mask = mask.to(DEVICE)
            coord = coord.to(DEVICE)

            # print('folder, step==================', sub_folder[idx], i)
            batch_size, channels, img_height, img_width = LR1.size()
            lowres_patches1 = patchify_tensor(LR1, patch_size=opt.test_patch_size, overlap=opt.stride)
            lowres_patches2 = patchify_tensor(LR2, patch_size=opt.test_patch_size, overlap=opt.stride)
            lowres_coord = patchify_tensor(coord, patch_size=opt.test_patch_size, overlap=opt.stride)
            lowres_mask = patchify_tensor(mask, patch_size=opt.test_patch_size, overlap=opt.stride)

            with torch.no_grad():
                num_set = 1
                if lowres_patches1.shape[0] % 128 > 0:
                    num_set = lowres_patches1.shape[0] // 128 + 1
                else:
                    num_set = lowres_patches1.shape[0] // 128
                ps = opt.test_patch_size * opt.up_ratio
                SR1_box = torch.zeros((lowres_patches1.shape[0], 3, ps, ps)).to(HR1.device)
                SR2_box = torch.zeros((lowres_patches1.shape[0], 3, ps, ps)).to(HR1.device)
                SR3_box = torch.zeros((lowres_patches1.shape[0], 3, ps, ps)).to(HR1.device)
                for p in range(num_set):
                    start = p * 128
                    end = min((p+1) * 128, lowres_patches1.shape[0])
                    lr1 = lowres_patches1[start:end]
                    lr2 = lowres_patches2[start:end]
                    coord_patch = lowres_coord[start:end]
                    mask_patch = lowres_mask[start:end]
                    SR1, SR2, SR3 = model(lr1, lr2, coord_patch, mask_patch)
                    SR1_box[start:end, :, :, :] = SR1
                    SR2_box[start:end, :, :, :] = SR2
                    SR3_box[start:end, :, :, :] = SR3

                SR1, SR2, SR3 = recompose_tensor(SR1_box, SR2_box, SR3_box, opt.up_ratio * img_height, opt.up_ratio * img_width,
                                                  overlap=opt.up_ratio * opt.stride)
            # out = bic_model(LR1, LR2)
            # SR1, SR2, SR3 = out.chunk(3, dim=0)

            # with torch.no_grad():
            #     SR1, SR2, SR3 = model(2 * LR1 - 1, 2 * LR2 - 1)
            #     SR1 = 0.5 * SR1 + 0.5
            #     SR2 = 0.5 * SR2 + 0.5
            #     SR3 = 0.5 * SR3 + 0.5
                
            SR1 = SR1 * mask
            SR2 = SR2 * mask
            SR3 = SR3 * mask
            HR1 = HR1 * mask
            HR2 = HR2 * mask
            HR3 = HR3 * mask
            SR1 = torch.clamp(SR1, min=0, max=1)
            HR1 = torch.clamp(HR1, min=0, max=1)
            SR2 = torch.clamp(SR2, min=0, max=1)
            HR2 = torch.clamp(HR2, min=0, max=1)
            dummy_st = time.time()
            rmse_loss = 0
            mae_loss = 0
            psnr_out = 0
            ssim_out = 0
            gmsd_out = 0
            lpips_out = 0
            if opt.eval is True:
                rmse_loss, mae_loss, psnr_out, ssim_out, gmsd_out, lpips_out = eval_metric(SR1, SR2, HR1, HR2)
            # loss3 = F.mse_loss(SR3, HR3)
            validation_rmse += rmse_loss
            validation_mae += mae_loss
            validation_psnr += psnr_out
            validation_ssim += ssim_out
            validation_gmsd += gmsd_out
            validation_lpips += lpips_out
            count += 1
            # save results
            SR1 = SR1.data[0].cpu().permute(1, 2, 0).numpy()
            SR2 = SR2.data[0].cpu().permute(1, 2, 0).numpy()

            SR1 = SR1 * 255
            SR2 = SR2 * 255
            SR1 = SR1.clip(0, 255)
            SR2 = SR2.clip(0, 255)

            SR1 = SR1.astype(np.uint8)
            SR2 = SR2.astype(np.uint8)

            mask = mask.data[0].cpu().permute(1, 2, 0).numpy()
            mask = mask.astype(np.uint8)
            mask = np.concatenate((mask[:,:,0:1], mask[:,:,1:2], mask[:,:,2:3]), axis=1)
            mask = np.squeeze(mask)
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)

            # heatmap1 = cv2.applyColorMap(SR1, cv2.COLORMAP_HOT)
            # heatmap2 = cv2.applyColorMap(SR2, cv2.COLORMAP_HOT)

            SR1_heat, SR2_heat = heat_vis(SR1, SR2, mask)
            save_path1 = join(opt.output_dir, file_name1[0])
            save_path2 = join(opt.output_dir, file_name2[0])
            # save_path3 = '/data/shallow_water/Pred_v3_pcd/' + folder_name + '_' + file_name[0] + '_' + str(3*i+2).zfill(2) + '.jpg'
            cv2.imwrite(save_path1, SR1)
            cv2.imwrite(save_path2, SR2)

            dummy_et = time.time()
            dummy_time += dummy_et - dummy_st

        et = time.time()
        final_rmse_loss = validation_rmse / count
        final_mae_loss = validation_mae / count
        final_psnr = validation_psnr / count
        final_ssim = validation_ssim / count
        final_gmsd = validation_gmsd / count
        final_lpips = validation_lpips / count
        time_run = (et - st - dummy_time) / count
        # print('Folder, total MSE, Spatial loss, Temporal loss ================', sub_folder[idx+1], final_loss.item(), final_spatial.item(), final_temporal.item())
        if opt.eval is True:
            print('Folder, RMSE, MAE, PSNR, SSIM, GMSD, LPIPS======================', 
                sub_folder[idx+1], final_rmse_loss.item(), final_mae_loss.item(), final_psnr.item(), 
                final_ssim.item(), final_gmsd.item(), final_lpips.item())
        print('Running time ================', time_run)
