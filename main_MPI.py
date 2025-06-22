import torch
import pytorch_lightning as pl
import argparse
from model import DNNCS, patchify_tensor, recompose_tensor
import torch.nn.functional as F
from dataset import *
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import os
import random
from einops import rearrange, reduce, repeat
import cv2
import torchvision
import time
import yaml
import seaborn as sns 
import matplotlib.pyplot as plt 


class Datamodule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
    # ============= read hd5 files from single file ======================
    def train_dataloader(self):
        """Load train set loader."""
        self.train_set = DatasetFromFolder_v3(
            data_dir=self.opt.train_data,
            patch_size=self.opt.patch_size, 
            upscale_factor=self.opt.up_ratio,
            data_augmentation=True
            )
        return data.DataLoader(
            self.train_set, 
            batch_size=opt.batchSize, 
            shuffle=True, 
            num_workers=opt.threads,
            drop_last=True,
        )

    def val_dataloader(self):
        """Load val set loader."""
        self.val_set = DatasetFromFolderEval_v5(
            data_dir=self.opt.validate_data,
            patch_size=self.opt.test_patch_size, 
            upscale_factor=self.opt.up_ratio
            )
        return data.DataLoader(
            self.val_set, 
            batch_size=opt.test_batchSize, 
            shuffle=False, 
            num_workers=opt.threads
            )



class LitAutoEncoder(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.model = DNNCS(dim=64)
        if opt.checkpoint is not None:
            pretrained_model = torch.load(opt.checkpoint, map_location=lambda storage, loc: storage)
            checkpoint = {
                k[6:]: v
                for k, v in pretrained_model['state_dict'].items()
            }
            self.model.load_state_dict(checkpoint, strict=False)
            print('==============successfully loaded pretrained SR model')


        self.lr = opt.lr
        self.automatic_optimization = True
        self.log_dir = opt.log_dir
        self.num_training_samples = opt.num_training_samples
        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()
        self.save_pool = []
        self.validate_loss = 0
        self.validate_lossLP = 0
        self.loss_fn = torch.nn.L1Loss()
        self.mse_fn = torch.nn.MSELoss()

    def cal_mass(self, SR, HR):
        HR = HR.reshape(HR.shape[0], 3, -1)
        SR = SR.reshape(SR.shape[0], 3, -1)
        sum_hr = torch.mean(HR, dim=2)
        sum_sr = torch.mean(SR, dim=2)
        loss_mass = self.mse_fn(sum_sr, sum_hr) 
        return loss_mass
    
    def Lp_loss(self, SR, HR, mask, p=2):
        mask = mask + 1e-6
        SR = SR * mask
        HR = HR * mask
        num_examples = SR.size()[0]
        total_loss = []
        for i in range(3):
            x = SR[:, i:i+1, :, :]
            y = HR[:, i:i+1, :, :]
            diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1), p, 1)

            out = torch.mean(diff_norms/y_norms)
            total_loss.append(out)
        total_loss = torch.stack(total_loss, dim=-1)

        data_loss = torch.mean(total_loss)
        
        return data_loss
    
    def deriv(self, SR1, SR2, SR3, HR1, HR2, HR3):
        SR = torch.cat((SR1, SR2, SR3), dim=0)
        HR = torch.cat((HR1, HR2, HR3), dim=0)
        sr_deriv1 = torch.gradient(SR, dim=1)[0]
        hr_deriv1 = torch.gradient(HR, dim=1)[0]
        sr_deriv2 = torch.gradient(SR, dim=2)[0]
        hr_deriv2 = torch.gradient(HR, dim=2)[0]
        sr_deriv3 = torch.gradient(SR, dim=3)[0]
        hr_deriv3 = torch.gradient(HR, dim=3)[0]
        loss = F.mse_loss(sr_deriv1, hr_deriv1) + F.mse_loss(sr_deriv2, hr_deriv2) + F.mse_loss(sr_deriv3, hr_deriv3)

        return loss
    
    def training_step(self, batch, batch_idx):
        LR1, LR2, HR1, HR2, HR3, mask, coord = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        SR1, SR2, SR3 = self.model(LR1, LR2, coord, mask)

        loss1 = self.loss_fn(SR1*mask, HR1*mask)
        loss2 = self.loss_fn(SR2*mask, HR2*mask)
        loss3 = self.loss_fn(SR3*mask, HR3*mask)
        loss_mse = loss1 + loss2 + loss3

        loss1_mass = self.cal_mass(SR1*mask, HR1*mask)
        loss2_mass = self.cal_mass(SR2*mask, HR2*mask)
        loss3_mass = self.cal_mass(SR3*mask, HR3*mask)
        loss_mass = loss1_mass + loss2_mass + loss3_mass

        loss_lp1 = self.Lp_loss(SR1, HR1, mask)
        loss_lp2 = self.Lp_loss(SR2, HR2, mask)
        loss_lp3 = self.Lp_loss(SR3, HR3, mask)
        loss_lp = loss_lp1 + loss_lp2 + loss_lp3

        loss_deriv = self.deriv(SR1, SR2, SR3, HR1, HR2, HR3)

        loss = 10 * loss_mse + loss_mass + loss_lp + 200 * loss_deriv
        # log metrics to wandb
        self.log('train_loss', loss, prog_bar=True)
        self.log('mse loss', loss_mse, prog_bar=True)
        self.log('mass loss', loss_mass, prog_bar=True)
        self.log('LP loss', loss_lp, prog_bar=True)
        self.log('deriv loss', loss_deriv, prog_bar=True)
        self.log('learning_rate', self.lr_schedulers().get_lr()[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        LR1, LR2, HR1, HR2, HR3, mask, coord = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        self.model.eval()
        batch_size, channels, img_height, img_width = LR1.size()
        lowres_patches1 = patchify_tensor(LR1, patch_size=self.opt.test_patch_size, overlap=self.opt.stride)
        lowres_patches2 = patchify_tensor(LR2, patch_size=self.opt.test_patch_size, overlap=self.opt.stride)
        lowres_coord = patchify_tensor(coord, patch_size=self.opt.test_patch_size, overlap=self.opt.stride)
        lowres_mask = patchify_tensor(mask, patch_size=self.opt.test_patch_size, overlap=self.opt.stride)

        with torch.no_grad():
            num_set = 1
            if lowres_patches1.shape[0] % 128 > 0:
                num_set = lowres_patches1.shape[0] // 128 + 1
            else:
                num_set = lowres_patches1.shape[0] // 128
            ps = self.opt.test_patch_size * self.opt.up_ratio
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
                SR1, SR2, SR3 = self.model(lr1, lr2, coord_patch, mask_patch)

                SR1_box[start:end, :, :, :] = SR1
                SR2_box[start:end, :, :, :] = SR2
                SR3_box[start:end, :, :, :] = SR3

            SR1, SR2, SR3 = recompose_tensor(SR1_box, SR2_box, SR3_box, self.opt.up_ratio * img_height, self.opt.up_ratio * img_width,
                                              overlap=self.opt.up_ratio * self.opt.stride)

        SR1 = SR1 * mask
        SR2 = SR2 * mask
        SR3 = SR3 * mask
        HR1 = HR1 * mask
        HR2 = HR2 * mask
        HR3 = HR3 * mask
        loss1 = F.mse_loss(SR1, HR1)
        loss2 = F.mse_loss(SR2, HR2)
        loss3 = F.mse_loss(SR3, HR3)
        loss_lp1 = self.Lp_loss(SR1, HR1, mask)
        loss_lp2 = self.Lp_loss(SR2, HR2, mask)
        loss_lp3 = self.Lp_loss(SR3, HR3, mask)
        loss_lp = loss_lp1 + loss_lp2 + loss_lp3
        loss = loss1 + loss2 + loss3
        self.validate_loss = self.validate_loss + loss
        self.validate_lossLP = self.validate_lossLP + loss_lp
        self.save_pool.append((HR1, HR2, HR3, SR1, SR2, SR3))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[self.opt.nEpochs//4, self.opt.nEpochs//2],
            gamma=0.5
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]
    
    def on_validation_epoch_end(self):

        (HR1, HR2, HR3, SR1, SR2, SR3) = self.save_pool[torch.randint(len(self.save_pool), (1,))]

        self.save_pool = []  # empty

        count = 0
        if (len(self.save_pool)) > 1:
            idx = 0
            count = len(self.save_pool)
        else:
            idx = torch.randint(HR1.shape[0], (1,))
            count = HR1.shape[0]

        com_pc = torch.cat((HR1[idx].cpu(), HR2[idx].cpu(), HR3[idx].cpu(), SR1[idx].cpu(), SR2[idx].cpu(), SR3[idx].cpu()), dim=0)
        
        # b, c, h, w = com_pc.shape
        # images = com_pc.permute(2, 0, 3, 1).reshape(h, b*w, c).clamp(0, 1)
        # u, v, eta = images.chunk(3, dim=2)
        # images = torch.cat((u, v, eta), dim=0)
        # images = images.numpy() * 255.0
        # save_path = '%s/epoch_%04d.jpg' % (self.log_dir, self.current_epoch+1)
        # images = images.astype(np.uint8)
        # heatmap = cv2.applyColorMap(images, cv2.COLORMAP_JET)
        # cv2.imwrite(save_path, heatmap)

        grid = torchvision.utils.make_grid(com_pc.permute(0, 1, 3, 2))
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        # log metrics to wandb
        loss = self.validate_loss / count
        loss_LP = self.validate_lossLP / count
        self.log("validation_loss", loss, sync_dist=True, prog_bar=True)
        self.log("validation_loss_LP", loss_LP, sync_dist=True, prog_bar=True)
        self.validate_loss = 0
        self.validate_lossLP = 0

        return loss


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Overall upsampling parameters
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--of_checkpoint", type=str, default='/data/flownet/FlowNet2-C_checkpoint.pth.tar')
    parser.add_argument("--log_dir", type=str, default="output")
    parser.add_argument('--patch_size', type=int, default=64, help='input LR patch size')
    parser.add_argument('--test_patch_size', type=int, default=64, help='input LR patch size')
    parser.add_argument('--stride', type=int, default=8, help='overlapping size')
    parser.add_argument("--up_ratio", type=int, default=1, help="upsampling factor")
    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    # training settings
    parser.add_argument('--train_data', type=str, default='/data/shallow_water/Train_v3')
    parser.add_argument('--validate_data', type=str, default='/data/shallow_water/Valid_v3')
    parser.add_argument('--aug', type=bool, default=True)
    parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
    parser.add_argument("--batchSize", type=int, default=32, help="training batch size") 
    parser.add_argument("--test_batchSize", type=int, default=1, help="testing batch size")
    parser.add_argument("--num_gpus", type=int, default=2, help="number of gpus to use")
    parser.add_argument("--strategy", type=str, default='ddp_find_unused_parameters_true', help="ddp_find_unused_parameters_true or ddp")
    parser.add_argument("--save_interval", type=int, default=5, help="checkpoint save interval")
    parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
    
    # ################ PREPARATIONS #################
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # setup data
    data_module = Datamodule(opt=opt)
    opt.num_training_samples = len(data_module.train_dataloader())
    # init the autoencoder
    autoencoder = LitAutoEncoder(opt)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer(max_epochs=opt.nEpochs, 
                         logger=tb_logger, 
                         devices = opt.num_gpus,
                         strategy = opt.strategy,
                         accelerator='gpu',
                         callbacks=[
                            ModelCheckpoint(
                                dirpath="ckpt", 
                                save_top_k=3, 
                                save_weights_only=True,
                                monitor="validation_loss",
                            )
                         ],
                         log_every_n_steps=5,
                         num_sanity_val_steps=0,
                         gradient_clip_algorithm="norm",
                         gradient_clip_val=1.0,
                         check_val_every_n_epoch=2,
                         )

    # train the model

    trainer.fit(autoencoder, data_module)