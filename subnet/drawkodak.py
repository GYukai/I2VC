import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2

def kodakdrawplt(lbpp, lpsnr, llpips, global_step, la='Ours', testfull=False):
    prefix = 'performance'
    if testfull:
        prefix = 'fullpreformance'
    LineWidth = 2
    test, = plt.plot(lbpp, lpsnr, marker='x', color='black', linewidth=LineWidth, label=la)

    bpp, psnr = [0.1152, 0.1857, 0.3018, 0.4690, 0.6864, 0.9669], [27.1100, 28.6800, 30.6200, 32.5550, 34.5800, 36.7200]
    bmshj2018, = plt.plot(bpp, psnr, "-o", linewidth=LineWidth, label='ICLR2018')

    bpp, psnr = [0.1156, 0.1745, 0.2690, 0.4270, 0.5953, 0.8000], [28.4310, 29.7659, 31.3200, 33.3630, 34.9494, 36.4444]
    Cheng, = plt.plot(bpp, psnr, "-o", color="goldenrod", linewidth=LineWidth, label='Cheng')

    bpp, psnr = [0.1142, 0.2439, 0.4747, 0.7186, 0.9802], [28.684, 31.2922, 34.2399, 36.16, 37.9119]
    EVC, = plt.plot(bpp, psnr, "-o", color="darkblue", linewidth=LineWidth, label='EVC')

    bpp, psnr = [0.3000, 0.4000, 0.5000, 0.8000], [31.9800, 33.2400, 34.3500, 36.8300]
    VTM, = plt.plot(bpp, psnr, "g--v", linewidth=LineWidth, label='VTM')

    bpp, psnr = [0.025113, 0.031415, 0.044679, 0.064834, 0.090035,  0.223581, 0.301998, 0.417876, 0.555947], [23.910667, 24.612929, 25.648502, 26.672548, 27.452418, 30.479342, 31.693313, 33.015954, 34.076672]
    ours, = plt.plot(bpp, psnr, "-*", color="red", linewidth=LineWidth, label='CAM_baseline (Proposed)')

    bpp, psnr = [0.025113, 0.031415, 0.044679, 0.064834, 0.090035,  0.223581, 0.301998, 0.417876, 0.555947], [23.820, 24.463, 25.399, 26.2576, 26.8758, 27.762, 27.829, 27.823, 27.814]
    cam, = plt.plot(bpp, psnr, "-*", color="cyan", linewidth=LineWidth, label='CAM_ldm (Proposed)')
    
    savepathpsnr = prefix + '/Kodak_psnr_' + str(global_step) + '.png'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    plt.legend(handles=[VTM, EVC, bmshj2018, Cheng, ours, cam, test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('Kodak dataset')
    plt.savefig(savepathpsnr)
    plt.clf()

# ----------------------------------------LPIPS-------------------------------------------------
    test, = plt.plot(lbpp, llpips, marker='x', color='black', linewidth=LineWidth, label=la)

    bpp, lpips = [0.025113, 0.031415, 0.044679, 0.064834, 0.090035, 0.222], [0.4341, 0.3948, 0.3367, 0.2828, 0.2411, 0.174]
    ours, = plt.plot(bpp, lpips, "-*", color="red", linewidth=LineWidth, label='CAM_baseline (Proposed)')
    # 0.301998, 0.417876, 0.555947,  0.223581]
    # 0.1289, 0.0985, 0.0781, 0.1638]
    # 0.1307, 0.1065, 0.0904, 0.1608

    bpp, lpips = [0.025113, 0.031415, 0.044679, 0.064834, 0.090035], [0.4247, 0.3843, 0.3248, 0.2709, 0.2297]
    cam, = plt.plot(bpp, lpips, "-*", color="cyan", linewidth=LineWidth, label='CAM_ldm (Proposed)')
   
    savepathlpips = prefix + '/' + 'Kodak_lpips_' + str(global_step) + '.png'
    plt.legend(handles=[ours, cam, test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('LPIPS')
    plt.title('Kodak dataset')
    plt.savefig(savepathlpips)
    plt.clf()


if __name__ == '__main__':
    labelname = 'ours'
    kodakdrawplt([], [], [], 1373002, la=labelname, testfull=True)