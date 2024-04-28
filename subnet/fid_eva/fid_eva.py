import os
from pytorch_fid import fid_score
from PIL import Image
import torchvision
import torchvision.transforms as transforms

def calc_fid(img1_path, img2_path):
    fid = fid_score.calculate_fid_given_paths([img1_path, img2_path],32,"cuda:0",2048)
    return fid

# UVG = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]
# hevcb = ["BQTerrace","BasketballDrive","Cactus","Kimono1","ParkScene"]
# hevcc = ["BQMall","BasketballDrill","PartyScene","RaceHorses"]
# hevcd = ["BasketballPass","BlowingBubbles","BQSquare","RaceHorses"]
# hevce = ["FourPeople","Johnny","KristenAndSara"]
# hevcf = ["vidyo1","vidyo3","vidyo4"]
#
# all_fid0 = all_fid1 = all_fid2 = all_fid3 = 0
# dataset = hevcf
#
# for path in dataset:
#     gt_path = '/opt/data/private/xcm/DVC/data/hevcF/images32/'+path+'/'
#     recon_path = '/opt/data/private/xcm/DVC/other_models/DCVC-DC/results/HEVCF-PSNR/'+ path
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
#     fid = fid_score.calculate_fid_given_paths([gt_path, recon_path0],32,"cuda:0",2048)
#     all_fid0 += fid
#
#     fid = fid_score.calculate_fid_given_paths([gt_path, recon_path1],32,"cuda:0",2048)
#     all_fid1 += fid
#
#     fid = fid_score.calculate_fid_given_paths([gt_path, recon_path2],32,"cuda:0",2048)
#     all_fid2 += fid
#
#     fid = fid_score.calculate_fid_given_paths([gt_path, recon_path3],32,"cuda:0",2048)
#     all_fid3 += fid
#
# print([round(all_fid0/(len(dataset)),3),round(all_fid1/(len(dataset)),3),round(all_fid2/(len(dataset)),3),round(all_fid3/(len(dataset)),3)])