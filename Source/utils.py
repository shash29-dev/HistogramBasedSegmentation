
import cv2
import numpy as np
from CreateMask import DrawMask
import os
import math as m
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix


class Mask_Histogram_Util():
    def __init__(self,im_name,nb_label):
        print('Reading Image...')
        self.nb_label=max(2,nb_label)
        self.ImageF=os.path.abspath(os.path.join(os.getcwd(),"..",'Images/'+im_name))
        self.Image=cv2.imread(self.ImageF)
        self.nx,self.ny,_=self.Image.shape
        self.MaskF=os.path.abspath(os.path.join(os.getcwd(),"..",'Masks/'))

    def save_manual_masks(self):
        print('Give Manual Masks for reference histogram...')
        print('Select Mask---> Right Click to Finish----> Hit Enter')
        for file in os.listdir(self.MaskF):
            os.remove(self.MaskF+'/'+file)
        for k in range(self.nb_label):
            I=cv2.imread(self.ImageF)
            pd = DrawMask("Mask")
            image=pd.run(I.copy())
            fName='/Mask'+str(k)+'.png'
            cv2.imwrite(self.MaskF+fName,image)
        print('Mask saved !')

    def read_masks(self):
        print('Reading Masks..')
        ImageName=sorted(os.listdir(self.MaskF),reverse=True)
        Mask=np.zeros((self.nx,self.ny,self.nb_label))
        for i in range(len(ImageName)):
            print(ImageName[i])
            Mask[:,:,i]=cv2.imread(self.MaskF+'/'+ImageName[i],0)
        return Mask


class HistogramUtil():
    def __init__(self,Image,Mask,nb_label):
        self.Image=Image
        self.Mask=Mask
        self.nx,self.ny,_=self.Image.shape
        self.nb_bin=6
        self.nb_label=nb_label
 
    def make_histogram(self):
        print('Determining Histograms...',sep=' ', end='', flush=True)
        histoIm={}
        histo_ref={}
        for k in range(self.nb_label):
            histo_ref[k]={}
        for i in range(self.nx):
            for j in range(self.ny):
                bI=m.floor((self.Image[i,j,0]/256)*self.nb_bin)
                gI=m.floor((self.Image[i,j,1]/256)*self.nb_bin)
                rI=m.floor((self.Image[i,j,2]/256)*self.nb_bin)
                index=(rI,gI,bI)
                histoIm.setdefault(index,0)
                histoIm[index]=histoIm[index]+1
                for k in range(self.nb_label):
                    if self.Mask[i,j,k]>0:
                        histo_ref[k].setdefault(index,0)
                        histo_ref[k][index]=histo_ref[k][index]+1
        self.histo_ref=histo_ref
        self.histoIm=histoIm
        print('done')

    def normalize_ref_histogram(self):
        print('Normalizing Reference Histogram...',sep=' ', end='', flush=True)
        h_ref=self.histo_ref.copy()
        for k in range(self.nb_label):
            Total=sum(self.histo_ref[k].values())
            h_ref[k]={key: val/Total for key, val in h_ref[k].items()}
        self.h_ref=h_ref
        print('done')

    def DictoMat(self,keyToBin):
    # Converting Dictionary to np arrays in arranged fashion
        tmpHref={}
        for k in self.h_ref.keys():
            tmp=np.zeros((len(keyToBin),1))
            for key in self.h_ref[k].keys():
                val=keyToBin[key][1]
                tmp[val]=self.h_ref[k][key]
            tmpHref.setdefault(k,tmp)
        return tmpHref 

    def key_to_bin(self):
        print('Creating Bins and Preparing data...',sep=' ', end='', flush=True)
        c=0
        keyToBin1={}
        for i in range(self.nb_bin):
            for j in range(self.nb_bin):
                for k in range(self.nb_bin):
                    keyToBin1.setdefault((i,j,k),c)
                    c+=1
        keyToBin={}
        c=0
        for key in sorted(self.histoIm.keys()):
            keyToBin[key]=[keyToBin1[key],c]
            c+=1

        normeH=[]
        for key in sorted(self.histoIm.keys()):
            normeH.append([self.histoIm[key]])
        self.normeH=np.array(normeH)

        self.h_ref=self.DictoMat(keyToBin)
        # Creating Sparse Matrix
        H= dok_matrix((len(keyToBin),self.nx*self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                bI=m.floor((self.Image[i,j,0]/256)*self.nb_bin)
                gI=m.floor((self.Image[i,j,1]/256)*self.nb_bin)
                rI=m.floor((self.Image[i,j,2]/256)*self.nb_bin)
                index=(rI,gI,bI)
                val=keyToBin[index][1]
                H[val,(self.nx*j)+i]=1  
        totalBin=len(self.histoIm) 
        self.H=H
        self.totalBin=totalBin
        print('done')

def Preconditioned_pd_algo(inst_histo):

    ### utils funcs ###
    print('Optimization...')
    nx,ny,nb_label=inst_histo.nx,inst_histo.ny,inst_histo.nb_label
    totalBin=inst_histo.totalBin
    nb_pixel=inst_histo.nx*inst_histo.ny
    h_ref=inst_histo.h_ref
    normeH=inst_histo.normeH
    H=inst_histo.H

    def op_H_t(q,href):
        s=sum(href*q)
        return np.tile(s,[nx*ny,1])

    def op_H(utm,href):
        s=np.sum(utm)
        return href*s

    def simplexProj(x):
        [n,d]=x.shape   
        ind=np.argsort(x)[...,::-1]
        tmp=np.array(range(n))
        tmp=np.tile(tmp,[d,1]).T
        row=tmp.reshape([n*d,1],order='F')
        col=ind.reshape([n*d,1],order='F')
        xs=x[row,col].reshape([n,d],order='F')
        xtmp= (np.cumsum(xs,axis=1)-1)/np.cumsum(1/np.ones((n,d)),axis=1)       
        gsum=np.sum(xs>xtmp,axis=1)    
        tmp=np.array(range(n)) 
        a=xtmp[tmp,abs(gsum-1)]
        yt=np.zeros(x.shape)
        for i in range(x.shape[1]):
            yt[:,i]=x[:,i]-a
        yt[yt<0]=0
        return yt
    ####################

    p={}
    q={}
    sigmap={}
    sigmaq={}
    for k in range(nb_label):
        p[k]=np.zeros((totalBin,1))
        q[k]=np.zeros((totalBin,1))
        norme_Href=nb_pixel*h_ref[k]
        sigmaq[k]=1/(norme_Href+totalBin)
        sigmap[k]=1/(normeH+totalBin)

    tau=1/8
    sigmaz=1/4
    rho=6
    zx=np.zeros((nx,ny,nb_label));
    zy=np.zeros((nx,ny,nb_label));
    div=np.zeros((nx,ny,nb_label));
    u=np.ones((nx,ny,nb_label))/nb_label
    ut=np.ones((nx,ny,nb_label))/nb_label



    # Preconditined Primal Dual Algorithm
    for kk in range(10):
        print('\t Iteration \t {}...'.format(kk+1),sep=' ', end='', flush=True)
        zx[0:-1,:,:]=zx[0:-1,:,:]+sigmaz*(ut[1:,:,:]-ut[0:-1,:,:])
        zy[:,0:-1,:]=zy[:,0:-1,:]+sigmaz*(ut[:,1:,:]-ut[:,0:-1,:])
        normez=np.maximum(rho,np.sqrt(np.square(zx)+np.square(zy)))
        zx=rho*zx/normez
        zy=rho*zy/normez
        
        for k in range(nb_label):
            p[k]=p[k]+sigmap[k]*(H*ut[:,:,k].reshape([nx*ny,1],order='F'))
            q[k]=q[k]+sigmaq[k]*(op_H(ut[:,:,k],h_ref[k]))
            p[k]=np.divide((sigmaq[k]*p[k])-(sigmap[k]*q[k]),(sigmaq[k]+sigmap[k]))
            p[k]=np.divide(p[k],np.maximum(1,abs(p[k])))
            q[k]=-p[k]
        
        div[0,:,:]=zx[0,:,:] 
        div[1:-1,:,:]=zx[1:-1,:,:]-zx[0:-2,:,:]
        div[-1,:,:]=-zx[-2,:,:]
        
        div[:,0,:]=div[:,0,:]+zy[:,0,:] 
        div[:,1:-1,:]=div[:,1:-1,:]+ zy[:,1:-1,:]-zy[:,0:-2,:]
        div[:,-1,:]=div[:,-1,:]-zy[:,-2,:]
        ut=u
        
        for k in range(nb_label):
            tmp=tau*(H.T*p[k]+op_H_t(q[k],h_ref[k]))-div[:,:,k].reshape((nx*ny,1),order='F')
            u[:,:,k]=u[:,:,k]-tmp.reshape((nx,ny),order='F')
        if nb_label==2:
            u[:,:,0]=np.maximum(0,np.minimum(1,(u[:,:,0]+(1-u[:,:,1]))/2))
            u[:,:,1]=1-u[:,:,0]
        else:
            utmp=u.reshape([nx*ny,nb_label],order='F')
            utmp=simplexProj(utmp)
            u=utmp.reshape([nx,ny,nb_label],order='F')
         
        ut=(2*u)-ut
        u0=np.argmax(u,axis=2) 
        print('done')
    return u0
        
