
import torch.optim
from tqdm import tqdm

import dataloader
import argparse
import model_GEN
import copy
from utils import ssim,AverageMeter,psnr
from utils.losses import *

def train(config):
    print("this is train_rgb ")
    print("gpu id is ", config.gpu_id)
    device = ('cuda:' + str(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    enhancer = model_GEN.GEN(config).to(device)

    train_dataset = dataloader.Traindata(config.lowlight_images_path_fivek)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    eval_dataset = dataloader.Valdata(config.lowlight_images_path_eval_fivek)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.val_batch_size, shuffle=True,
                                              num_workers=config.num_workers, pin_memory=True)

    optimizer=torch.optim.Adam(enhancer.parameters(), lr=config.lr,
                                                                         weight_decay=config.weight_decay)

    l_loss = nn.L1Loss().to(device)
    perceptual_loss = perception_loss().to(device)

    best_epoch_psnr = 0
    best_epoch_ssim = 0
    best_psnr = 0.
    best_ssim = 0.

    for epoch in range(config.num_epochs):
        enhancer.train()
        enhancer = enhancer.to(device)
        print("lr", optimizer.param_groups[0]['lr'])
        print("-------")
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for iteration, images in loop:
            img = images[0].to(device)
            gt = images[1].to(device)

            out = enhancer(img)

            loss_l = l_loss(out, gt)
            loss_per = perceptual_loss(out, gt) * 0.04
            loss = loss_l + loss_per

            optimizer.zero_grad()

            loss.backward()

            # torch.nn.utils.clip_grad_norm(enhancer.parameters(), config.grad_clip_norm)

            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{config.num_epochs}]')
            loop.set_postfix(loss=format(loss.item())
                             , loss_l=format(loss_l.item())
                             , loss_per=format(loss_per.item())
                             )
        enhancer.eval()
        epoch_psnr = AverageMeter.AverageMeter()
        epoch_ssim = AverageMeter.AverageMeter()

        loop_ = tqdm(enumerate(eval_loader), total=len(eval_loader))
        for iteration, images in loop_:
            img = images[0].to(device)
            gt = images[1].to(device)
            img_path = images[2]

            with torch.no_grad():
                out = enhancer(img)

            for i, pt in enumerate(img_path):
                if os.path.basename(pt) == 'a4512-09-05-19-at-18h43m31s-_MG_9546.jpg':
                    torchvision.utils.save_image(out[i], "data/temp/fivek_Epoch" + str(epoch) + ".jpg")

            epoch_psnr.update(psnr.calc_psnr(out, gt))
            epoch_ssim.update(ssim.ssim(out, gt))

            loop_.set_description(f'Epoch [{epoch}/{config.num_epochs}]')
            loop_.set_postfix(psnr=format(epoch_psnr.avg), ssim=format(epoch_ssim.avg))
        if epoch_ssim.avg > best_ssim:
            best_epoch_ssim = epoch
            best_ssim = epoch_ssim.avg
        if epoch_psnr.avg > best_psnr:
            best_epoch_psnr = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(enhancer.state_dict())

        print('best psnr epoch: {}, psnr: {:.2f}'.format(best_epoch_psnr, best_psnr))
        print('best ssim epoch: {}, ssim: {:.4f}'.format(best_epoch_ssim, best_ssim))

        torch.save(best_weights, os.path.join(config.snapshots_folder, 'best.pth'))

if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()

    #  training params
    # parser.add_argument('--lowlight_images_path_fivek', type=str, default="./data/train_data/fivek/train_lowsize/")
    # parser.add_argument('--lowlight_images_path_eval_fivek', type=str, default="./data/train_data/fivek/eval_lowsize/")
    parser.add_argument('--lowlight_images_path_fivek', type=str, default="/data/fivek/fivek_png/train/")
    parser.add_argument('--lowlight_images_path_eval_fivek', type=str, default="/data/fivek/fivek_png/eval/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=250)
    parser.add_argument('--snapshots_folder', type=str, default="./checkpoints/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="./checkpoints/best.pth")
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--low_img_size', type=int, default=128)

    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)