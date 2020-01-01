import cv2
import numpy as np
from CreateMask import DrawMask
import os
import math as m
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from utils import Mask_Histogram_Util,HistogramUtil,Preconditioned_pd_algo
import pdb
import argparse

if __name__ == '__main__':

    if not os.path.exists('./../results'): os.mkdir('./../results')
    parser=argparse.ArgumentParser()
    parser.add_argument("im_name", help="Name of Image to be segmented",type=str)
    parser.add_argument("--labels",help='Parts to be segmented',default=2,type=int)
    parser.add_argument("--manual_label", help="Set Masks Manually", default=1,type=int)
    args = parser.parse_args()
    ############### set values here ################
    ImageName=args.im_name   
    nb_label=args.labels          # nb_label= 2,3 respectively or as per choice
    ################################################
    nb_bin=6  
    manual_input_histograms=args.manual_label
    nb_label=max(2,nb_label)

    mask_inst=Mask_Histogram_Util(ImageName,nb_label)
    if manual_input_histograms:
    	mask_inst.save_manual_masks()
    Mask=mask_inst.read_masks()

    inst_hist=HistogramUtil(mask_inst.Image,Mask,nb_label)
    inst_hist.make_histogram()
    inst_hist.normalize_ref_histogram()
    inst_hist.key_to_bin()

    u0=Preconditioned_pd_algo(inst_hist)

    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(mask_inst.Image[...,::-1])
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(u0-1,'gray')
    plt.axis('off')
        
    u0N=cv2.normalize(1-u0,None,0,255,cv2.NORM_MINMAX)
    contours, hierarchy = cv2.findContours(np.copy(u0N),cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    c=cv2.drawContours(np.copy(mask_inst.Image), contours, -1, (255, 0, 0), 2)
    plt.subplot(3,1,3)
    plt.imshow(c[...,::-1])
    plt.axis('off')
    plt.savefig('./../results/'+ImageName[:-4]+'_result.png')
    plt.show()