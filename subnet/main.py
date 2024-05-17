# -*- coding: utf8 -*-
import argparse
import datetime
import json
import logging
import os

import cv2
import torch.optim as optim
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from diffusers import LDMSuperResolutionPipeline
from pytorch_msssim import ms_ssim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.models.utils as utility
from dataset import DataSet, KodakDataSet
from drawkodak import drawplt
from subnet.dis_eva.dis_eva import calc_dis
from subnet.fid_eva.fid_eva import calc_fid
from subnet.src.models.CAM_net import *
from accelerate.logging import get_logger
from transformers import get_linear_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import get_constant_schedule

# def geti(lamb):
#     if lamb == 2048:
#         return 'H265L20'
#     elif lamb == 1024:
#         return 'H265L23'
#     elif lamb == 512:
#         return 'H265L26'
#     elif lamb == 256:
#         return 'H265L29'
#     else:
#         print("cannot find lambda : %d"%(lamb))
#         exit(0)

# ref_i_dir = geti(train_lambda)

def parse_argument():
    parser = argparse.ArgumentParser(description='FVC reimplement')
    parser.add_argument('-l', '--log', default='',
                        help='output training details')
    parser.add_argument('-p', '--pretrain', default='',
                        help='load pretrain model')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testuvg', action='store_true')
    parser.add_argument('--testvtl', action='store_true')
    parser.add_argument('--testmcl', action='store_true')
    parser.add_argument('--testauc', action='store_true')
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--allpick', action='store_true')
    parser.add_argument('--config', dest='config', required=True,
                        help='hyperparameter of Reid in json format')
    parser.add_argument('--from_scratch', default=False, action='store_true')
    parser.add_argument('--mse_loss-factor', type=float, default=1.0)
    parser.add_argument('--lps_loss-factor', type=float, default=0.05)
    parser.add_argument('--lmd-mode', type=str, choices=['fixed', 'random'], default="fixed",
                        help='Mode to set the lmd factor. Choose "fixed" to use a specific value or "random" to generate '
                             'it randomly within bounds.')
    parser.add_argument('--lmd-fixed_value', type=int, default=256,
                        help='Fixed value for lmd when mode is "fixed". This argument is required if mode is "fixed".')
    parser.add_argument('--lmd-lower_bound', type=int, default=8,
                        help='Lower bound for lmd when mode is "random". This argument is required if mode is "random".')
    parser.add_argument('--lmd-upper_bound', type=int, default=256,
                        help='Upper bound for lmd when mode is "random". This argument is required if mode is "random".')
    parser.add_argument('--test-interval', type=int, default=2000)
    parser.add_argument('--exp-name', type=str, default='exp')
    parser.add_argument('--batch-per-gpu', type=int, default=2)
    parser.add_argument('--test-dataset-path', type=str, default='data/Kodak24/kodak')
    parser.add_argument('--test-lmd', type=int, default=256)
    return parser.parse_args()


def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir, ifend, iftest, ifmsssim, msssim_lambda, ifout
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        # ref_i_dir = geti(train_lambda)
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
    if 'ifend' in config:
        ifend = config['ifend']
    if 'iftest' in config:
        iftest = config['iftest']
    if 'ifmsssim' in config:
        ifmsssim = config['ifmsssim']
    if 'msssim_lambda' in config:
        msssim_lambda = config['msssim_lambda']
    if 'ifout' in config:
        ifout = config['ifout']


def Var(x):
    return Variable(x.cuda())


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    input_tensor = input_tensor.to(torch.device('cpu'))

    input_tensor = input_tensor.squeeze()

    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


# def testuvg(global_step):
#     with torch.no_grad():
#         test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
#         net.eval()
#
#         sumbpp = 0
#         sumpsnr = 0
#         summsssim = 0
#         cnt = 0
#         train_lambda_tensor = Var(torch.tensor(train_lambda).view(1))
#         train_lambda_tensor_boudary = Var(torch.tensor(2048).view(1))
#
#         for batch_idx, input in enumerate(test_loader):
#
#             log = 'testing squeence: {} '.format(batch_idx)
#             print(log)
#
#             input_images = input
#             seqlen = input_images.size()[1]
#             cnt += seqlen
#             # print(seqlen)
#
#             for i in range(0, seqlen):
#                 image1 = input_images[:, i, :, :, :]
#                 # image2 = input_images[:, i+1, :, :, :]
#                 # image3 = input_images[:, i+2, :, :, :]
#                 # frame1, frame2, frame3 = Var(image1), Var(image2), Var(image3)
#                 frame1 = Var(image1)
#
#                 clipped_recon_image, feature_map, distortion, bpp_y, bpp_z, bpp = net(frame1, train_lambda_tensor,
#                                                                                       train_lambda_tensor_boudary)
#
#                 bpp_c = torch.mean(bpp).cpu().detach().numpy()
#                 psnr_c = torch.mean(10 * (torch.log(1. / distortion) / np.log(10))).cpu().detach().numpy()
#                 msssim_c = ms_ssim(clipped_recon_image.cpu().detach(), image1, data_range=1.0,
#                                    size_average=True).numpy()
#
#                 sumbpp += (bpp_c)
#                 sumpsnr += (psnr_c)
#                 summsssim += (msssim_c)
#
#             log = 'bpp: {:.4f}  psnr: {:.4f} ssim: {:.4f}'.format(sumbpp / cnt, sumpsnr / cnt, summsssim / cnt)
#             print(log)
#
#         log = "global step %d : " % (global_step) + "\n"
#         logger.info(log)
#         sumbpp /= cnt
#         sumpsnr /= cnt
#         summsssim /= cnt
#         log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (
#             sumbpp, sumpsnr, summsssim)
#         logger.info(log)
#         uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=True)


def testkodak(global_step, test_loader, net, logger):
    '''
    Test one model to one test dataset
    In this version, specific designed for Kodak
    '''
        # latents_dtype = next(net.parameters()).dtype
        # sigma = net.scheduler.init_noise_sigma
    net.eval()
    with torch.no_grad():
        sub_sumbpp = []
        sub_sumpsnr = []
        sub_summsssim = []
        sub_sumlpips = []
        sub_sumdis = []

        train_lambda_tensor = torch.tensor(train_lambda)
        print(f"Test Lambda set to {train_lambda}")
        cnt = test_loader.__len__()
        print(cnt)

        recon_path_768 = os.path.join("recon",args.exp_name,str(global_step),"kodak_recon_768")
        recon_path_512 = os.path.join("recon",args.exp_name,str(global_step),"kodak_recon_512")

        gt_path_768 = os.path.join(args.test_dataset_path, "images", "768x512")
        gt_path_512 = os.path.join(args.test_dataset_path, "images", "512x768")

        for num, input in enumerate(test_loader):
            frame = Var(input)
            height, width = frame.shape[-2:]

            latents_shape = (1, 3, height // 4, width // 4)
            latents = torch.randn(latents_shape, dtype=latents_dtype)
            latents = Var(latents * sigma)

            clipped_recon_image, _, distortion, lps_distortion, bpp = net(frame, latents, train_lambda_tensor, 2048)

            os.makedirs(recon_path_768, exist_ok=True)
            os.makedirs(recon_path_512, exist_ok=True)
            h, w = clipped_recon_image.shape[2], clipped_recon_image.shape[3]
            img_name = 'kodim' + str(len(test_loader)*accelerator.process_index + num + 1).zfill(3) + '_' + str(train_lambda) + '_recon.png'
            if w == 768:
                save_image_tensor2cv2(clipped_recon_image, os.path.join(recon_path_768, img_name))
            else:
                save_image_tensor2cv2(clipped_recon_image, os.path.join(recon_path_512, img_name))

            bpp_c = torch.tensor(bpp).to(accelerator.device)
            psnr_c = torch.tensor(10 * (torch.log(1. / distortion) / np.log(10))).to(accelerator.device)
            msssim_c = torch.tensor(ms_ssim(clipped_recon_image.cpu().detach(), frame.cpu().detach(), data_range=1.0,
                                size_average=True)).to(accelerator.device)
            lpips_c = torch.tensor(torch.mean(lps_distortion)).to(accelerator.device)
            dis = torch.tensor(calc_dis(frame, clipped_recon_image)).to(accelerator.device)
            sub_sumbpp.append(bpp_c)
            sub_sumpsnr.append(psnr_c)
            sub_summsssim.append(msssim_c)
            sub_sumlpips.append(lpips_c)
            sub_sumdis.append(dis) # Multithread, don't modify unless you know what are you doing
        sub_sumbpp = torch.mean(torch.stack(sub_sumbpp))
        sub_sumpsnr = torch.mean(torch.stack(sub_sumpsnr))
        sub_summsssim = torch.mean(torch.stack(sub_summsssim))
        sub_sumlpips = torch.mean(torch.stack(sub_sumlpips))
        sub_sumdis = torch.mean(torch.stack(sub_sumdis))
        
        sumbpp, sumpsnr, summsssim, sumlpips, sumdis = accelerator.gather_for_metrics((sub_sumbpp, sub_sumpsnr, sub_summsssim, sub_sumlpips, sub_sumdis))

        if accelerator.is_main_process:

            fid_768 = calc_fid(recon_path_768, gt_path_768)
            fid_512 = calc_fid(recon_path_512, gt_path_512)
            sumfid = (fid_512 * 6 + fid_768 * 18) /24

            log = "\n===== TST:global step %d : " % (global_step)
            logger.info(log)
            sumbpp = torch.mean(sumbpp).item()
            sumpsnr = torch.mean(sumpsnr).item()
            summsssim = torch.mean(summsssim).item()
            sumlpips = torch.mean(sumlpips).item()
            sumdis = torch.mean(sumdis).item()
            sumfid = sumfid
            log = f"Kodakdataset : average bpp : {sumbpp:.6f}, average psnr : {sumpsnr:.6f}, average msssim: {summsssim:.6f}\n, average lpips: {sumlpips:.6f}, average DIS: {sumdis:.6f}, average fid:{sumfid:6f}\n======"

            logger.info(log)
            drawplt(bpp=[sumbpp], lpips=[sumlpips], psnr=[sumpsnr], disit=[sumdis], fid=[sumfid], savepath=f"outfig/{args.exp_name}", step=global_step)
            # if not args.testuvg:
            #     tb_logger.add_scalar('kodak_bpp', sumbpp, global_step)
            #     tb_logger.add_scalar('kodak_psnr', sumpsnr, global_step)
            #     tb_logger.add_scalar('kodak_lpips', sumlpips, global_step)
            #     tb_logger.add_scalar('kodak_msssim', summsssim, global_step)
            #     tb_logger.add_scalar('kodak_dis', sumdis, global_step)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def run(model, batch_size, optimizer, lr, train_loader, global_step, logger, scheduler,test_loader,
        print_step=100, cal_step=10):



    cur_lr = lr
    model.train()
    print("Requires_grad parameters number: " + str(utility.count_network_parameters(model)))
    logger.info("Requires_grad parameters number: " + str(utility.count_network_parameters(model)))

    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    sumbpp = 0
    sum_lpips = 0
    sum_dis = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()
    try:
        for epoch in range(stepoch, tot_epoch):
            print("epoch", epoch)
            print(f"gloabl_step: {global_step}")
            logger.info(f"epoch {epoch} with len of train_loader: {len(train_loader)}")
            for batch_idx, input in tqdm(enumerate(train_loader)):
                global_step += 1
                bat_cnt += 1
                image1, image2, image3 = Variable(input[0]), Variable(input[1]), Variable(input[2])
                quant_noise_feature, quant_noise_z = Variable(input[3]), Variable(input[4])
                latents = Variable(input[5])
                if args.lmd_mode == 'fixed':
                    var_lambda = args.lmd_fixed_value
                elif args.lmd_mode == 'random':
                    var_lambda = np.random.randint(args.lmd_lower_bound, args.lmd_upper_bound)
                else:
                    raise ValueError(f"Invalid lambda mode: {args.lmd_mode}")
                clipped_recon_bimage, _, distortion, lpips_distortion, bpp = model(input_image=image2, latents=latents,
                                                                                lmd=var_lambda, lmd_boundary=2048,
                                                                                previous_frame=None, feature_frame=None,
                                                                                quant_noise_feature=quant_noise_feature,
                                                                                quant_noise_z=quant_noise_z)

                distortion, bpp, lpips_distortion = torch.mean(distortion), torch.mean(bpp), torch.mean(lpips_distortion)
                dis_loss = calc_dis(image2, clipped_recon_bimage)
                rd_loss = var_lambda * (args.mse_loss_factor * distortion + args.lps_loss_factor * lpips_distortion) + bpp

                optimizer.zero_grad()
                accelerator.backward(rd_loss)



                # clip_gradient(optimizer, 0.5)
                optimizer.step()
                scheduler.step()

                if global_step % cal_step == 0:
                    cal_cnt += 1
                    if distortion > 0:
                        psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
                        lpips = lpips_distortion.cpu().detach().numpy()
                    else:
                        psnr = 100

                    loss_ = rd_loss.cpu().detach().numpy()

                    sumloss += loss_
                    sumpsnr += psnr
                    sum_dis += dis_loss
                    sumbpp += bpp.cpu().detach()
                    sum_lpips += lpips

                if (batch_idx % print_step) == 0 and bat_cnt > 1:
                    tb_logger.add_scalar('lr', scheduler.get_last_lr(), global_step)
                    tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
                    tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
                    tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
                    tb_logger.add_scalar('lpips', sum_lpips / cal_cnt, global_step)
                    tb_logger.add_scalar('dis', sum_dis / cal_cnt, global_step)
                    t1 = datetime.datetime.now()
                    deltatime = t1 - t0
                    print(f'Train Epoch : {epoch:02} Lr: {scheduler.get_last_lr()} Global Step: {global_step}  Avgloss:{sumloss / cal_cnt:.6f} lr:{cur_lr} time:{(deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt}')
                    print(f'Details : psnr : {sumpsnr / cal_cnt:.2f} bpp : {sumbpp / cal_cnt:.6f} lpips : {sum_lpips / cal_cnt:.6f} dis : {sum_dis / cal_cnt:.6f}')
                    print(f"mse-factor: {args.mse_loss_factor}, lps-factor: {args.lps_loss_factor}")

                    bat_cnt = 0
                    cal_cnt = 0
                    sumbpp = sumloss = sumpsnr = sum_lpips = sum_dis = 0
                    t0 = t1

                if global_step % args.test_interval == 0:
                    logger.info(f"Saving at exp_name: {args.exp_name} at global step {global_step}")
                    save_model(model, global_step, args.exp_name)
                    save_path = os.path.join("snapshot_A", args.exp_name, f"iter{global_step}.model")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(output_dir=save_path)
                    testkodak(global_step, test_loader, model, logger)
            # logger.info(f"Saving at exp_name: {args.exp_name} at global step {global_step}")
        save_model(model, global_step, args.exp_name)
        save_path = os.path.join("snapshot_A", args.exp_name, f"iter{global_step}.model")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        accelerator.save_state(output_dir=save_path)
        testkodak(global_step, test_loader, model, logger)
        log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
        logger.info(log)
        return global_step
    except KeyboardInterrupt:
        logger.info(f"Training interrupted, saving at {global_step} step")
        save_model(model, global_step, args.exp_name)
        save_path = os.path.join("snapshot_A", args.exp_name, f"iter{global_step}.model")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        accelerator.save_state(output_dir=save_path)
        exit(0)


def main():


    state = AcceleratorState()
    num_devices, device_kind = state.num_processes, state.distributed_type
    print(f"num_devices: {num_devices}, device_kind: {device_kind}")

    torch.backends.cudnn.enabled = True
    gpu_num = torch.cuda.device_count()
    num_workers = gpu_num * 4


    # base_lr = 0.0001  # * gpu_num
    # train_lambda = 2048
    # print_step = 100
    # cal_step = 10
    # print_step = 10
    # warmup_step = 0  # // gpu_num
    # decay_interval = 1800000
    # lr_decay = 0.1
    # ifarr = 0
    # ifout = 0
    # recon_path = 'recon/recon.bin'
    # tb_logger = None
    # test_step = 10000  # // gpu_num

    tot_epoch = 100
    tot_step = 3000000

    logger = get_logger("VC3")
    file_handler = logging.FileHandler(os.path.join('logs',args.exp_name + '.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.logger.addHandler(file_handler)
    global_step = 0


    ##################################################################################
    pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages",
                                                      local_files_only=True)
    scheduler = pipe.scheduler

    scheduler.set_timesteps(28)
    vae = pipe.vqvae
    unet = pipe.unet
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    global latents_dtype
    latents_dtype = next(unet.parameters()).dtype
    global sigma
    sigma = scheduler.init_noise_sigma
    ##################################################################################


    parse_config(args.config)
    train_lambda = args.test_lmd

    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_path = os.path.join('events', args.exp_name, time_str)
    global tb_logger
    tb_logger = SummaryWriter(tb_path)
    gpu_per_batch = args.batch_per_gpu

    # logger = get_logger(__name__)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    # stdhandler = logging.StreamHandler()
    # stdhandler.setLevel(logging.INFO)
    # stdhandler.setFormatter(formatter)
    # logger.addHandler(stdhandler)
    # if args.log != '':
    #     filehandler = logging.FileHandler(args.log)
    #     filehandler.setLevel(logging.INFO)
    #     filehandler.setFormatter(formatter)
    #     logger.addHandler(filehandler)

    logger.setLevel(logging.INFO)
    logger.info("\n\n\n\n")
    logger.info("\n==================================\nCAM_I training")
    # logger.info(open(args.config).read())


    print("----------")
    print(f"args: {args}")
    logger.info(f"args: {args}")
    print("----------")

    model: CAM_net = CAM_net(vae, unet, scheduler)
    # for para in model.parameters():
    #     para.requires_grad = False
    model.unet.requires_grad_(True)

    cur_lr = 2.5e-5
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cur_lr)
    scheduler = get_constant_schedule(
        optimizer
    )

    train_dataset = DataSet(latents_dtype, sigma, "./data/vimeo_septuplet/test.txt")
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=num_workers, batch_size=gpu_per_batch,
                              pin_memory=True)
    test_dataset_I = KodakDataSet(os.path.join(args.test_dataset_path, "kodak"))
    test_loader = DataLoader(dataset=test_dataset_I, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)

    # train_loader, net, optimizer, scheduler = accelerator.prepare(train_loader, net, optimizer, scheduler)



    model, optimizer, scheduler,train_loader,test_loader = accelerator.prepare(model, optimizer, scheduler,train_loader,test_loader)
    if args.pretrain != '':
        accelerator.load_state(args.pretrain)
        p_str = str(args.pretrain)
        if p_str.find('iter') != -1 and p_str.find('.model') != -1:
            st = p_str.find('iter') + 4
            ed = p_str.find('.model', st)
            global_step = int(p_str[st:ed])  # return step
        else:
            global_step = 0
        logger.info(f"ACCELERATE: Loaded pretrain model from {args.pretrain}, global step: {global_step}")  




    if args.testuvg:
        testkodak(global_step, test_loader, model, logger)
        print('Tested Kodak, END')
        exit(0)



    global stepoch
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch * gpu_num))  # * gpu_num))

    log_exp = f'EXPERIMENT: {args.exp_name}'
    log_tb = f'TENSORBOARD: {tb_path}'
    log_lambda = f'TST_LAMBDA: {train_lambda}'
    log_lmd_mode = f'TRAIN_LMD_MODE: {args.lmd_mode}, FIXED_LMD: {args.lmd_fixed_value}, LOWER_LMD: {args.lmd_lower_bound}, UPPER_LMD: {args.lmd_upper_bound}'
    logger.info(log_exp)
    logger.info(log_lambda)
    logger.info(log_lmd_mode)
    logger.info(log_tb)



    global_step = run(global_step=global_step, model=model,
                          batch_size=gpu_per_batch, optimizer=optimizer, lr=cur_lr,
                          train_loader=train_loader, logger=logger, scheduler=scheduler, test_loader=test_loader)
    # logger.info(f"Saving at global step {global_step}")
    # save_model(model, global_step,args.exp_name)


if __name__ == "__main__":
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    print = accelerator.print
    args = parse_argument()
    main()
