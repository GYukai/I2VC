import os
from PIL import Image
from .DISTS_pt import prepare_image, DISTS
import torch

device = "cuda:0"
test_for_dis = DISTS().to(device)
def calc_dis(gt, recon):
    dis = test_for_dis(gt.to(device), recon.to(device), batch_average=True).item()
    return dis

# UVG = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]
# hevcb = ["BQTerrace","BasketballDrive","Cactus","Kimono1","ParkScene"]
# hevcc = ["BQMall","BasketballDrill","PartyScene","RaceHorses"]
# hevcd = ["BasketballPass","BlowingBubbles","BQSquare","RaceHorses"]
# hevce = ["FourPeople","Johnny","KristenAndSara"]
# hevcf = ["vidyo1","vidyo3","vidyo4"]
#
# all_dis0 = all_dis1 = all_dis2 = all_dis3 = 0
# dataset = hevcd
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# for path in dataset:
#     gt_path = '/opt/data/private/xcm/DVC/data/hevcD/images32/'+path+'/'
#     recon_path = '/opt/data/private/xcm/DVC/other_models/DCVC-DC/results/HEVCD-PSNR/'+ path
#     for folder in os.listdir(recon_path):
#         if folder.startswith("0_"):
#             recon_path0 = recon_path + ('/'+folder+'/')
#         if folder.startswith("1_"):
#             recon_path1 = recon_path + ('/'+folder+'/')
#         if folder.startswith("2_"):
#             recon_path2 = recon_path + ('/'+folder+'/')
#         if folder.startswith("3_"):
#             recon_path3 = recon_path + ('/'+folder+'/')
#
#     for n in range(0, 32):
#         gt_png = prepare_image(Image.open(gt_path + 'im' + str(n+1).zfill(3) + '.png').convert("RGB")).to(device)
#         recon_png = prepare_image(Image.open(recon_path0 + 'im' + str(n+1).zfill(5) + '.png').convert("RGB")).to(device)
#         dis = test_for_dis(gt_png, recon_png).item()
#         all_dis0 += dis
#
#     for n in range(0, 32):
#         gt_png = prepare_image(Image.open(gt_path + 'im' + str(n+1).zfill(3) + '.png').convert("RGB")).to(device)
#         recon_png = prepare_image(Image.open(recon_path1 + 'im' + str(n+1).zfill(5) + '.png').convert("RGB")).to(device)
#         dis = test_for_dis(gt_png, recon_png).item()
#         all_dis1 += dis
#
#     for n in range(0, 32):
#         gt_png = prepare_image(Image.open(gt_path + 'im' + str(n+1).zfill(3) + '.png').convert("RGB")).to(device)
#         recon_png = prepare_image(Image.open(recon_path2 + 'im' + str(n+1).zfill(5) + '.png').convert("RGB")).to(device)
#         dis = test_for_dis(gt_png, recon_png).item()
#         all_dis2 += dis
#
#     for n in range(0, 32):
#         gt_png = prepare_image(Image.open(gt_path + 'im' + str(n+1).zfill(3) + '.png').convert("RGB")).to(device)
#         recon_png = prepare_image(Image.open(recon_path3 + 'im' + str(n+1).zfill(5) + '.png').convert("RGB")).to(device)
#         dis = test_for_dis(gt_png, recon_png).item()
#         all_dis3 += dis
#
# print([round(all_dis0/(32*len(dataset)),5),round(all_dis1/(32*len(dataset)),5),round(all_dis2/(32*len(dataset)),5),round(all_dis3/(32*len(dataset)),5)])
