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


def drawplt(bpp, psnr=None, lpips=None, fid=None, disit=None, savepath='./outfig/', step=0):

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    _metrics = [psnr, lpips, fid, disit]
    for i in range(len(_metrics)):
        if _metrics[i] is None:
            _metrics[i] = [0] * len(bpp)
    psnr, lpips, fid, disit = _metrics[0], _metrics[1], _metrics[2], _metrics[3]

    LineWidth = 4
    Markersize = 10
    savepathdis = os.path.join(savepath, f"Kodak_dis_{step}.png")
    savepathlps = os.path.join(savepath, f"Kodak_lps_{step}.png")
    savepathfid = os.path.join(savepath, f"Kodak_fid_{step}.png")
    savepathpsnr = os.path.join(savepath, f"Kodak_psnr_{step}.png")
    plt.rcParams.update({'font.size': 40})
    plt.rcParams['figure.figsize'] = (20, 16)

    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(LineWidth)
    ax.spines['left'].set_linewidth(LineWidth)
    ax.spines['top'].set_linewidth(LineWidth)
    ax.spines['right'].set_linewidth(LineWidth)

    VTM_AI_bpp, VTM_AI_l, VTM_AI_f, VTM_AI_d = [0.21456739637586805, 0.43711937798394096, 0.7968571980794272,
                                                1.3344311184353295], [0.14843, 0.08558, 0.04943, 0.0296], [67.912,
                                                                                                           36.305,
                                                                                                           18.392,
                                                                                                           9.647], [
        0.10001, 0.05628, 0.03188, 0.019]
    HiFiC_bpp, HiFiC_l, HiFiC_f, HiFiC_d = [0.091782187, 0.259258993, 0.43848822], [0.067125002, 0.035261946,
                                                                                    0.014439106], [54.84925477,
                                                                                                   37.32461606,
                                                                                                   24.95266454], [
        0.091348196, 0.066313835, 0.050604216]
    CDC_bpp, CDC_l, CDC_f, CDC_d = [0.182853851, 0.284156075, 0.505863008], [0.07744836, 0.053649945, 0.025284657], [
        31.84179438, 21.83418834, 10.73098895], [0.051428877, 0.035374757, 0.017132249]
    DGML_bpp, DGML_l, DGML_f, DGML_d = [0.05739, 0.12722, 0.2221, 0.38176, 0.57369, 0.80485], [0.30366, 0.23635,
                                                                                               0.17535, 0.12158,
                                                                                               0.07584, 0.04535], [
        187.939, 155.856, 124.136, 87.337, 61.54, 39.34], [0.214679, 0.179829, 0.149417, 0.11219, 0.082779, 0.060482]
    VC3_AI_bpp, VC3_AI_l, VC3_AI_f, VC3_AI_d = bpp, lpips, fid, disit

    # ------------------------------lps----------------------------------------------------

    VTM_AI_lps, = plt.plot(VTM_AI_bpp, VTM_AI_l, "g--o", markersize=Markersize, linewidth=LineWidth,
                           label='VTM-19.0 (AI default)')
    DGML_lps, = plt.plot(DGML_bpp, DGML_l, "-o", color="teal", markersize=Markersize, linewidth=LineWidth,
                         label='DGML (CVPR\'20)')
    HiFiC_lps, = plt.plot(HiFiC_bpp, HiFiC_l, "-o", color="darkorchid", markersize=Markersize, linewidth=LineWidth,
                          label='HiFiC (NeurIPS\'20)')
    CDC_lps, = plt.plot(CDC_bpp, CDC_l, "-o", color="tan", markersize=Markersize, linewidth=LineWidth,
                        label='CDC (NeurIPS\'23)')
    VC3_AI_lps, = plt.plot(VC3_AI_bpp, VC3_AI_l, "-o", color="red", markersize=Markersize * 1.5, linewidth=LineWidth,
                           label='VC\u00b3-AI (Proposed)')

    labelss = plt.legend(handles=[VTM_AI_lps, DGML_lps, HiFiC_lps, CDC_lps, VC3_AI_lps], prop={'size': 35},
                         loc=1).get_texts()
    plt.grid()
    [label.set_fontname('Times New Roman') for label in labelss]
    plt.xlabel('Bpp', fontproperties='Times New Roman', fontsize=50)
    plt.ylabel('LPIPS', fontproperties='Times New Roman', fontsize=50)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.title('Kodak', fontproperties='Times New Roman', fontsize=50)
    plt.savefig(savepathlps, bbox_inches='tight', pad_inches=0.05)
    plt.clf()

    # ------------------------------fid----------------------------------------------------

    plt.rcParams.update({'font.size': 40})
    plt.rcParams['figure.figsize'] = (20, 16)

    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(LineWidth)
    ax.spines['left'].set_linewidth(LineWidth)
    ax.spines['top'].set_linewidth(LineWidth)
    ax.spines['right'].set_linewidth(LineWidth)

    VTM_AI_fid, = plt.plot(VTM_AI_bpp, VTM_AI_f, "g--o", markersize=Markersize, linewidth=LineWidth,
                           label='VTM-19.0 (AI default)')
    DGML_fid, = plt.plot(DGML_bpp, DGML_f, "-o", color="teal", markersize=Markersize, linewidth=LineWidth,
                         label='DGML (CVPR\'20)')
    HiFiC_fid, = plt.plot(HiFiC_bpp, HiFiC_f, "-o", color="darkorchid", markersize=Markersize, linewidth=LineWidth,
                          label='HiFiC (NeurIPS\'20)')
    CDC_fid, = plt.plot(CDC_bpp, CDC_f, "-o", color="tan", markersize=Markersize, linewidth=LineWidth,
                        label='CDC (NeurIPS\'23)')
    VC3_AI_fid, = plt.plot(VC3_AI_bpp, VC3_AI_f, "-o", color="red", markersize=Markersize * 1.5, linewidth=LineWidth,
                           label='VC\u00b3-AI (Proposed)')

    labelss = plt.legend(handles=[VTM_AI_fid, DGML_fid, HiFiC_fid, CDC_fid, VC3_AI_fid], prop={'size': 35},
                         loc=1).get_texts()
    plt.grid()
    [label.set_fontname('Times New Roman') for label in labelss]
    plt.xlabel('Bpp', fontproperties='Times New Roman', fontsize=50)
    plt.ylabel('FID', fontproperties='Times New Roman', fontsize=50)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.title('Kodak', fontproperties='Times New Roman', fontsize=50)
    plt.savefig(savepathfid, bbox_inches='tight', pad_inches=0.05)
    plt.clf()

    # ------------------------------dis----------------------------------------------------

    plt.rcParams.update({'font.size': 40})
    plt.rcParams['figure.figsize'] = (20, 16)

    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(LineWidth)
    ax.spines['left'].set_linewidth(LineWidth)
    ax.spines['top'].set_linewidth(LineWidth)
    ax.spines['right'].set_linewidth(LineWidth)

    VTM_AI_dis, = plt.plot(VTM_AI_bpp, VTM_AI_d, "g--o", markersize=Markersize, linewidth=LineWidth,
                           label='VTM-19.0 (AI default)')
    DGML_dis, = plt.plot(DGML_bpp, DGML_d, "-o", color="teal", markersize=Markersize, linewidth=LineWidth,
                         label='DGML (CVPR\'20)')
    HiFiC_dis, = plt.plot(HiFiC_bpp, HiFiC_d, "-o", color="darkorchid", markersize=Markersize, linewidth=LineWidth,
                          label='HiFiC (NeurIPS\'20)')
    CDC_dis, = plt.plot(CDC_bpp, CDC_d, "-o", color="tan", markersize=Markersize, linewidth=LineWidth,
                        label='CDC (NeurIPS\'23)')
    VC3_AI_dis, = plt.plot(VC3_AI_bpp, VC3_AI_d, "-o", color="red", markersize=Markersize * 1.5, linewidth=LineWidth,
                           label='VC\u00b3-AI (Proposed)')

    labelss = plt.legend(handles=[VTM_AI_dis, DGML_dis, HiFiC_dis, CDC_dis, VC3_AI_dis], prop={'size': 35},
                         loc=1).get_texts()
    plt.grid()
    [label.set_fontname('Times New Roman') for label in labelss]
    plt.xlabel('Bpp', fontproperties='Times New Roman', fontsize=50)
    plt.ylabel('DISTS', fontproperties='Times New Roman', fontsize=50)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.title('Kodak', fontproperties='Times New Roman', fontsize=50)
    plt.savefig(savepathdis, bbox_inches='tight', pad_inches=0.05)
    plt.clf()


if __name__ == '__main__':
    labelname = 'ours'
    kodakdrawplt([], [], [], 1373002, la=labelname, testfull=True)