
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pyfits
import pdb
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib import colors
import scipy.ndimage as ndimage



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def coopixfits(reg):
    haname="./cuts/"+reg+"/"+reg+"_ha_.fits"
    #iha_r,hdrha=pyfits.getdata(haname,0,header=True)
    hdu=pyfits.open(haname)
    hdr=hdu[0].header

    dx=hdr['NAXIS1']
    dy=hdr['NAXIS2']
    xini=abs(hdr['ltv1'])
    yini=abs(hdr['ltv2'])

    coords=[xini,yini,xini+dx,yini+dy]
    return coords
    #print coords

bxx=[]
byy=[]
blb=[]

#To read the table of stars in PIXELS and put it into a table [x,y,txt]
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def ini_stars_tbl():
    global bxx
    filename="./bonanos10/bon10_allclasses_PIX.reg"
    arch =open(filename,'r')
    [arch.readline() for i in xrange(3)]

    for s in np.arange(5324):
        row=arch.readline()
        tok=row.split('(')
        num=tok[1].split(')')
        coo=num[0].split(',')
        x=float(coo[0])
        y=float(coo[1])
        tx=num[1].split('{')
        txt=tx[1].split('}')
        lbl=txt[0]

        #print s,x,y,lbl
        #bst.append([s,x,y,lbl])
        bxx.append(x)
        byy.append(y)
        blb.append(lbl)
        #print bxx

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def extractinerg(lms):
    nax=np.array(bxx)
    nay=np.array(byy)
    nat=np.array(blb)
    
    #pdb.set_trace()

    subx=np.ma.masked_outside(bxx,lms[0],lms[2])
    suby=np.ma.masked_outside(byy,lms[1],lms[3])

    msk=np.ma.mask_or(subx.mask,suby.mask)
    
    sxx=np.ma.masked_where(msk==1,subx)
    syy=np.ma.masked_where(msk==1,suby)
    stt=np.ma.masked_where(msk==1,blb)

    out=[np.ma.compressed(sxx),
         np.ma.compressed(syy),
         np.ma.compressed(stt)]

    


    return out

    #    print nax
    #    print nay
    #    print nat
    #    print '-----------------------'
    #    print subx






#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_region(reg):
    #Reads point from individual region.
    print "REGION:",reg
    filename='./cuts/'+reg+'/'+reg+'_PIX_str.reg'
    arch=open(filename,'r')
    [arch.readline() for i in xrange(4)]
    res=[]
    for line in arch:
        if(line.find('polygon')>=0):
            cs=get_poly_coords(line)
            cna=np.array(cs,dtype=float)
            dim=cna.shape
            for row in np.arange(dim[0]):
                res.append((cna[row,0]-1,cna[row,1]-1))
            return res
            break
#> > > > >

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_center_points(points):
    ctxx=0
    ctyy=0
    c=0
    #nwctxx=0
    #nwctyy=0
    for p in points:
        ctxx=ctxx+p[0]
        ctyy=ctyy+p[1]
        #cx=p[0]*k
        #cy=p[1]*k
        #nwctxx=nwctxx+cx
        #nwctyy=nwctyy+cy
        c=c+1
    meds=(ctxx/c,ctyy/c)
    #nwmeds=(nwctxx/c,nwctyy/c)
    #medsdiff=(meds[0]-nwmeds[0],meds[1]-nwmeds[1])
    return meds
#> > > > >

#Returns the components of a polygon region.
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_poly_coords(line):
    if(line.find('polygon')>=0):
        tks0=line.split(')')
        tks1=tks0[0].split('(')
        tsk=tks1[1].split(',')
        N=len(tsk)
        clst=[]
        for k in np.arange(N/2):
            clst.append((tsk[k*2],tsk[k*2+1]))
        
        return clst
#>>>>>

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def make_all():
    #ini_stars_tbl()
    for k in np.arange(215):
        region='s{}'.format(k)
        dirname='./cuts/'+region
        if os.path.exists(dirname):
            #print region
            load_region(region,save=1)
            print dirname

rlst=[ 's17', 's20', 's67', 's91', 's98', 's107', 's161', 's51','s96',
       's6', 's33', 's34', 's44', 's169','s90','s80' ]
   
            
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def make_small():
    #ini_stars_tbl()

#    rlst=[ 's17', 's20', 's67', 's91', 's98', 's107', 's161', 
#           's6', 's33', 's34', 's44', 's169' ]

#    rlst=[ 's17', 's20', 's67', 's91', 's98', 's107', 's161', 's51','s96'
#           's6', 's33', 's34', 's44', 's169','s90','s80' ]
    
    #    for k in np.arange(len(rlst)):
    for k in rlst:
        region=k
        dirname='./cuts/'+region
        if os.path.exists(dirname):
            #print region
            load_region(region,save=1)
            print dirname
            
    
#>>>>>

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def dopages():
    g=open('./escimgs/sntwo/index_sntwo.html','w')
    g.write('<!DOCTYPE html>\n')
    g.write('<frameset cols="25%,75%">\n')
    g.write('<frame name="lside" src="./idx_sntwo.html"/>\n')
    g.write('<frame name="mainblock" src="./s17_sntwo.html"/>\n')
    g.write('<noframes></noframes></frameset>\n')
    g.write('</html>\n')
    g.close()

    
    f=open('./escimgs/sntwo/idx_sntwo.html','w')
    f.write('<HTML>')
    f.write('<body>')
    f.write('<table>')
    #<tr><td>thin</td> </tr>')
    c=0
    for reg in rlst:
        html_one(reg)
        
        if(c==0):
            f.write('<table>\n')
            f.write('<tr><td>thin</td> </tr>\n')
        if(c==9):
            f.write('</table>\n')    
            f.write('<table>\n')
            f.write('<tr><td>THICK</td> </tr>\n')
            
        f.write('<tr><td><a href="./'+reg+'_sntwo.html" target="mainblock">MCELS-'+reg+'</a></td></tr>\n')
        c=c+1
    f.write('</table>\n')
    f.write('</body>\n')
    f.write('</HTML>\n')
            

def html_one(reg):
    f=open('./escimgs/sntwo/'+reg+'_sntwo.html','w')

    f.write('<HTML>\n')
    f.write('<body>\n')
    f.write('<table border="1">\n')
    f.write('<tr><td><img src="../p12_smc_err/apj449869f24_'+reg[1:]+'_lr.gif" width="800" alt="xyz"></td></tr>\n')
    f.write('<tr><p>MCELS-'+reg+'</p><td><img src="./'+reg+'_sntwo.jpg" width="800" alt="xyz"></td></tr>\n')
    f.write('</table>\n')
    f.write('<body>\n')
    f.write('<HTML>\n')
    f.close()





#myBcmap= colors.ListedColormap(['white'])#, 'blue'])

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#To pre-process the image after sending it to the imshow
def clean_img(ax,img,kol='Greys_r'):
    num=np.ma.count(img)
    imin=img.min()
    imax=img.max()
    imea=img.mean()
    imed=np.ma.median(img)
    cent=(imax-imin)/100.0

    #print "*****************"
    print 'count:',num
    print "min,max:",imin,imax
    #lowlim=imin+cent*1.0
    #higlim=imin+cent*99.0
    #print "low,high:",lowlim,higlim
    #print "mean,median:",imea,imed

    dat=np.ma.compressed(img)
    
    ord=np.sort(dat)
    centdat=num/100.0
    #print '0,-1:',ord[0],ord[-1]
    #print "centdat:",centdat
    ilow=int(centdat*1.0)
    iup=int(centdat*99.0)
    vlow=ord[ilow]
    vup=ord[iup]
    print "ilow, iup:",vlow,vup


    #pdb.set_trace()
    ax.imshow(img,origin='lower',interpolation='none',cmap=kol,vmin=vlow,vmax=vup)
    #ax.imshow(img,origin='lower',interpolation='none',cmap=myBcmap,vmin=vlow,vmax=vup)

#>>>>>

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_bkstats(img):
    dim=img.shape
    lowlim_rg=np.median(img[0:dim[0]-1,-6:-1]) 
    lowlim_lf=np.median(img[0:dim[0]-1,0:6])
    lowlim=min([lowlim_rg,lowlim_lf])
    
    std_rg=np.std(img[0:dim[0]-1,-6:-1])
    std_lf=np.std(img[0:dim[0]-1,0:6])
    stdv=np.sqrt(min([std_rg,std_lf]))

    return [lowlim,stdv]
#>>>>>    


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def load_region(reg,save=0):
    points=get_region(reg)
    points_arr=np.array(points)
    center=get_center_points(points)
    ifx=reg

    s2name="./cuts/"+ifx+"/"+ifx+"_s2_.fits"
    o3name="./cuts/"+ifx+"/"+ifx+"_o3_.fits"
    haname="./cuts/"+ifx+"/"+ifx+"_ha_.fits"
    #soname="./sections/"+ifx+"/"+ifx+"_S2O3_.fits"

    is2_r,hdrs2=pyfits.getdata(s2name,0,header=True)
    io3_r,hdro3=pyfits.getdata(o3name,0,header=True)
    iha_r,hdrha=pyfits.getdata(haname,0,header=True)
    #iso_r,hdrha=pyfits.getdata(soname,0,header=True)


    #bkgnoise={'ha':12.96,
    #          'o3':0.45,
    #          's2':2.17}


    suave=1

    gx=2
    gy=2
    if(suave==1):
        is2 = ndimage.gaussian_filter(is2_r, sigma=(gx, gy), order=0)
        io3 = ndimage.gaussian_filter(io3_r, sigma=(gx, gy), order=0)
        iha = ndimage.gaussian_filter(iha_r, sigma=(gx, gy), order=0)
 #       iso = ndimage.gaussian_filter(iso_r, sigma=(gx, gy), order=0)
    else:
        is2 = is2_r
        io3 = io3_r
        iha = iha_r
  #      iso = iso_r

    bkgnoise={'ha':get_bkstats(iha),
                  'o3':get_bkstats(io3),
                  's2':get_bkstats(is2)}

    print "bkgnoise:", get_bkstats(iha_r),get_bkstats(io3_r),get_bkstats(is2_r)

    #is2 = is2_r
    #io3 = io3_r
    #iha = iha_r
    ##iso = iso_r

    dim=is2.shape
    width=dim[1]
    height=dim[0]
    img=Image.new('L',(width,height),0)
    ImageDraw.Draw(img).polygon(points,outline=1,fill=1)
    mask=np.array(img)

    pos_r=np.zeros_like(iha)
    for px in np.arange(dim[1]):
        for py in np.arange(dim[0]):
            pos_r[py,px]=np.sqrt(px*px+py*py)
    
    msk_i=np.ma.masked_where(mask==0,is2)
    is2_m=np.ma.masked_where(msk_i.mask==1,is2)
    io3_m=np.ma.masked_where(msk_i.mask==1,io3)
    iha_m=np.ma.masked_where(msk_i.mask==1,iha)
    pos=np.ma.masked_where(msk_i.mask==1,pos_r)
    #s2o3=is2_m/io3_m


    st=0
    o3_msk_a=np.ma.masked_where(io3_m<bkgnoise['o3'][st]*5.0,io3_m)
    s2_msk_a=np.ma.masked_where(o3_msk_a.mask==1,is2_m)
    #s2_msk=np.ma.masked_outside(s2_msk_a,bkgnoise['s2'][st]*0.0,bkgnoise['s2'][st]*3.0)
    s2_msk=np.ma.masked_where(s2_msk_a>bkgnoise['s2'][st]*2.0,s2_msk_a)
    o3_msk=np.ma.masked_where(s2_msk==1,io3_m)
    ha_msk=np.ma.masked_where(s2_msk==1,iha_m)
    s2o3_msk=s2_msk/o3_msk
    s2ha_msk=np.log10(s2_msk/ha_msk)
    o3ha_msk=np.log10(o3_msk/ha_msk)


    o3_msk_b=np.ma.masked_where(io3_m<bkgnoise['o3'][st]*5.0,io3_m)
    s2_msk_b=np.ma.masked_where(o3_msk_b.mask==1,is2_m)
    s2_msk_D=np.ma.masked_outside(s2_msk_b,bkgnoise['s2'][st]*2.0,bkgnoise['s2'][st]*4.0)
    o3_msk_D=np.ma.masked_where(s2_msk_D==1,io3_m)
    ha_msk_D=np.ma.masked_where(s2_msk_D==1,iha_m)
    s2o3_msk_D=s2_msk_D/o3_msk_D
    s2ha_msk_D=np.log10(s2_msk_D/ha_msk_D)
    o3ha_msk_D=np.log10(o3_msk_D/ha_msk_D)

    print "limha::",bkgnoise['ha']#[st]*25.0
    print "limo3::",bkgnoise['o3']#[st]*25.0
    print "lims2::",bkgnoise['s2']#[st]*25.0

    #ax=fig.add_subplot(111,projection='3d')

    #**create a list of points that have good OIII detection but not SII.
    #  Hence OIII should > 3-5 sigma but SII > 1-3.
    #I)

    s2ha=np.log10(is2_m/iha_m)
    o3ha=np.log10(io3_m/iha_m)
    s2o3=s2ha-o3ha
    ha=np.log10(iha_m)
    s2=np.log10(is2_m)
    o3=np.log10(io3_m)
    s2o3ha=np.log10((is2_m*io3_m)/iha_m)
    #pos=np.log10(pos_r)
    #s2o3=is2_m/io3_m
    #is2_m/io3_m
    lgpos=np.log10(pos)
    
    xx=s2ha
    yy=o3ha
    zz=s2o3
    ww=ha

    fi=np.log10((is2_m/io3_m)/iha_m)
    se=np.log10((is2_m/io3_m)*iha_m)

    x_lbl='log [SII]/H$\\alpha$'
    y_lbl='log [OIII]/H$\\alpha$'
    z_lbl='log [SII]/[OIII]'
    w_lbl='log H$\\alpha$'
    pos_lbl="pos"


    xrg=[-2.5,1.0]
    yrg=[-2.5,1.0]
    xtnts=[xrg[0],xrg[1],yrg[0],yrg[1]]

    noise_raw=np.load("./noise_ref_1001.npy")
    noise=noise_raw
    nmin=0.0#np.amin(noise)
    nmax=np.amax(noise)
    pcent=(nmax-nmin)/100.0
    nlow=pcent*1
    nup=pcent*95
    nlevs=np.linspace(start=nlow,stop=nup,num=5)


    distances=0
    twodim=0
    trid=0
    ddiag=0
    newrat=0
    dostresha=0
    makemap=0
    makeimgs=0
    haso=0
    signoise=1
    thestars=0

    #ini_stars_tbl()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(thestars==1):
        plyA=plt.Polygon(points, closed=True, fill=None, edgecolor='black',linewidth=1)
        plyB=plt.Polygon(points, closed=True, fill=None, edgecolor='black',linewidth=1)
        fig=plt.figure()
        fig.suptitle(reg)
        fx=2
        fy=1
        ax=fig.add_subplot(fy,fx,1)
        bx=fig.add_subplot(fy,fx,2)
        
        clean_img(ax,np.sqrt(ha))
        ax.add_patch(plyA)
        ax.set_xlabel("pix")
        ax.set_ylabel("pix")
        ax.set_title("H$\\alpha$")

        clean_img(bx,s2o3)
        bx.add_patch(plyB)
        bx.set_xlabel("pix")
        #bx.set_ylabel("pix")
        bx.set_title("s2/o3")

        coo=coopixfits(reg)
        outs=extractinerg(coo)

        '''
        print 'coo',coo
        print "out ------------\n",
        print outs[0]
        print outs[1]
        print outs[2]
        print "out ------------"
        '''

        qol='springgreen'
        qol='chartreuse'
        qolb='blue'
        r=5
        for q in np.arange(len(outs[0])):
            #pdb.set_trace()
            if(outs[2][q][0] in ['B','O']):
                #print outs[2][q][0]
                h=[]
                v=[]
                cx=outs[0][q]-coo[0]
                cy=outs[1][q]-coo[1]
                for a in np.arange(360):
                    angle=a*3.14/180.0
                    h.append(r*np.cos(angle)+cx)
                    v.append(r*np.sin(angle)+cy)
                ax.plot(h,v,color=qol)
                bx.plot(h,v,color=qolb)

            #ax.annotate(outs[2][q],xy=(cx-10,cy+10),xycoords='data',color=qolg,weight='bold',size=10)
                ax.annotate(outs[2][q],xy=(cx-10,cy+10),xycoords='data',color=qol,weight='bold',size=8.5)
                bx.annotate(outs[2][q],xy=(cx-10,cy+10),xycoords='data',color=qolb,weight='bold',size=8.5)
            
        if(save==1):
            fig.set_size_inches(10,6)
            plt.savefig("./escimgs/ions/"+reg+"_"+"ions.jpg")
            plt.close(fig)
        else:
            fig.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(signoise==1):

        plyA=plt.Polygon(points, closed=True, fill=None, edgecolor='black',linewidth=1)
        plyB=plt.Polygon(points, closed=True, fill=None, edgecolor='black',linewidth=1)
        plyC=plt.Polygon(points, closed=True, fill=None, edgecolor='black',linewidth=1)
        plyD=plt.Polygon(points, closed=True, fill=None, edgecolor='black',linewidth=1)

        fig=plt.figure()
        fig.suptitle(reg)
        fx=2
        fy=2
        ax=fig.add_subplot(fy,fx,1)
        bx=fig.add_subplot(fy,fx,2)
        cx=fig.add_subplot(fy,fx,3)
        dx=fig.add_subplot(fy,fx,4)
        myBcmap= colors.ListedColormap(['blue'])#, 'white'])
        myRedmap=colors.ListedColormap(['red'])#springgreen'])
        #bounds=[0,np.ma.max(ha_msk),9.e19]

        clean_img(ax,np.sqrt(iha_m))
        ##clean_img(ax,np.sqrt(ha_msk),kol='Blues_r')
        ax.imshow(np.sqrt(ha_msk),origin='lower',interpolation='none',\
                      cmap=myBcmap)
        ax.imshow(np.sqrt(ha_msk_D),origin='lower',interpolation='none',\
                      cmap=myRedmap)



        ###ax.imshow(np.sqrt(io3_m),origin='lower',interpolation='none',\
        ###              cmap='Blues_r')

        ax.add_patch(plyA)

        clean_img(bx,s2o3)
        bx.imshow(np.sqrt(ha_msk),origin='lower',interpolation='none',\
                      cmap=myBcmap)
        bx.imshow(np.sqrt(ha_msk_D),origin='lower',interpolation='none',\
                      cmap=myRedmap)

        #bx.imshow(np.sqrt(o3_msk_a),origin='lower',interpolation='none',\
        #              cmap=myBcmap)

        bx.add_patch(plyB)

        ax.set_xlabel("pix")
        ax.set_ylabel("pix")
        ax.set_title("H$\\alpha$")

        bx.set_xlabel("pix")
        #bx.set_ylabel("pix")
        bx.set_title("s2/o3")

        cx.scatter(s2ha,o3ha,marker=".",s=1)
        cx.scatter(s2ha_msk,o3ha_msk,marker=".",s=3,color='blue')
        cx.scatter(s2ha_msk_D,o3ha_msk_D,marker=".",s=3,color='red')

        cx.contour((noise),extent=xtnts,levels=nlevs,colors='green')
        cx.set_aspect('equal')
        cx.set_xlabel(x_lbl)
        cx.set_ylabel(y_lbl)

        cx.set_xlim(xrg)#-2.5,1.0)
        cx.set_ylim(yrg)#-2.5,1.0)
        

        dx.scatter(s2ha_msk,o3ha_msk,marker=".",s=3,color='blue')
        dx.scatter(s2ha_msk_D,o3ha_msk_D,marker=".",s=3,color='red')

        dx.contour((noise),extent=xtnts,levels=nlevs,colors='green')
        dx.set_aspect('equal')
        dx.set_xlabel(x_lbl)
        dx.set_xlim(xrg)#-2.5,1.0)
        dx.set_ylim(yrg)#-2.5,1.0)

        if(save==1):
            fig.set_size_inches(8,8)
            if(suave==0):
                #plt.savefig("./escimgs/sgn/"+reg+"_"+"sgn4.jpg")
                plt.savefig("./escimgs/sntwo/"+reg+"_"+"sntwo.jpg")
            if(suave==1):
                plt.savefig("./escimgs/sgns1/"+reg+"_"+"sgn4s1.jpg")
            plt.close(fig)
        else:
            fig.show()



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(haso==1):
        fig=plt.figure()
        fx=1
        fy=1
        ax=fig.add_subplot(fy,fx,1)
        ax.scatter(ww,zz,marker=".",s=1)
        ax.set_xlabel(w_lbl)
        ax.set_ylabel(z_lbl)
        if(save==1):
            fig.set_size_inches(6,6)
            plt.savefig("./escimgs/"+reg+"_"+"has2o30.jpg")
            plt.close(fig)
        else:
            fig.show()
       #plt.close(fig)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(makemap==1):
        fig=plt.figure()
        fx=1
        fy=1
        ax=fig.add_subplot(fy,fx,1)
        ax.scatter(xx,yy,marker=".",s=1)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(reg)

        ax.set_xlim(-2.5,1.0)
        ax.set_ylim(-2.5,1.0)

        if(save==1):
            fig.set_size_inches(6,6)
            plt.savefig("./escimgs/"+reg+"_"+"d0.jpg")
        else:
            fig.show()
        plt.close(fig)
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(makeimgs==1):
        fig=plt.figure()
        fx=2
        fy=1
        ax=fig.add_subplot(fy,fx,1)
        bx=fig.add_subplot(fy,fx,2)

        #ax.scatter(xx,yy,marker=".",s=1)
        #ax.imshow(np.log10(ha),origin='lower',interpolation='none')
        clean_img(ax,np.sqrt(ha))
        clean_img(bx,zz)

        #,extent=xtnts,vmin=conts_log[0],vmax=conts_log[-1])

        ax.set_xlabel("pix")
        ax.set_ylabel("pix")
        ax.set_title("H$\\alpha$")

        #ax.set_xlim(-2.5,1.0)
        #ax.set_ylim(-2.5,1.0)

        bx.set_xlabel("pix")
        bx.set_ylabel("pix")
        bx.set_title("s2/o3")


        if(save==1):
            fig.set_size_inches(8,6)
            plt.savefig("./escimgs/"+reg+"_"+"ihso0.jpg")
        else:
            fig.show()
        #plt.close(fig)



    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(dostresha==1):
        fig=plt.figure()
        fy=1
        fx=1   
        ax=fig.add_subplot(fy,fx,1)
        ax.scatter(ha,zz,marker=".",s=1)
        ax.set_xlim(-2.0,3.8)
        ax.set_xlim(-1.5,0.5)


        fig.show()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(newrat==1):
        fig=plt.figure()
        fy=2
        fx=1   
        fig.suptitle(reg)
        ax=fig.add_subplot(fy,fx,1)
        ax.scatter(pos,fi,marker=".",s=1)
        ax.set_xlabel(pos_lbl)
        ax.set_ylabel("fi")

        bx=fig.add_subplot(fy,fx,2)
        bx.scatter(pos,se,marker=".",s=1)
        bx.set_xlabel(pos_lbl)
        bx.set_ylabel("se")

        if(save==1):
            fig.set_size_inches(6,6)
            plt.savefig("./escimgs/"+reg+"_"+"fise0.jpg")
        else:
            fig.show()
        #plt.close(fig)

        #ax.set_xlabel(pos_lbl)
        #ax.set_ylabel("ha")
        #plt.show()


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(ddiag==1):
        fig=plt.figure()
        fy=1
        fx=1
        ax=fig.add_subplot(fy,fx,1)
        ax.scatter(s2ha,o3ha,marker=".",s=1)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(reg)
        if(save==1):
            fig.set_size_inches(6,6)
            plt.savefig("./escimgs/"+reg+"_"+"d0.jpg")
            plt.close(fig)
        else:
            fig.show()
        #plt.close(fig)


        plt.show()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(distances==1):
        fig=plt.figure()
        fy=3
        fx=2
        ax=fig.add_subplot(fy,fx,1)
        ax.scatter(pos,ha,marker=".",s=1)
        #ax.set_xlabel(pos_lbl)
        ax.set_ylabel("ha")

        bx=fig.add_subplot(fy,fx,3)
        bx.scatter(pos,s2,marker=".",s=1)
        bx.set_ylabel("s2")

        cx=fig.add_subplot(fy,fx,5)
        cx.scatter(pos,o3,marker=".",s=1)
        cx.set_ylabel("o3")

        dx=fig.add_subplot(fy,fx,2)
        dx.scatter(pos,s2ha,marker=".",s=1)
        dx.set_ylabel("s2/ha")

        ex=fig.add_subplot(fy,fx,4)
        ex.scatter(pos,o3ha,marker=".",s=1)
        ex.set_ylabel("o3/ha")

        fx=fig.add_subplot(fy,fx,6)
        fx.scatter(pos,zz,marker=".",s=1)
        fx.set_ylabel("s2/o3")


        plt.show()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if(twodim==1):
        fig=plt.figure()
        ax=fig.add_subplot(221)
        ax.scatter(zz,yy,marker=".",s=1)
        ax.set_xlabel(z_lbl)
        ax.set_ylabel(y_lbl)

        bx=fig.add_subplot(222)
        bx.scatter(zz,xx,marker=".",s=1)
        bx.set_xlabel(z_lbl)
        bx.set_ylabel(x_lbl)

        cx=fig.add_subplot(223)
        cx.scatter(xx,yy,marker=".",s=1)
        cx.set_xlabel(x_lbl)
        cx.set_ylabel(y_lbl)

        dx=fig.add_subplot(224)
        dx.scatter(zz,ww,marker=".",s=1)
        dx.set_xlabel(z_lbl)
        dx.set_ylabel(w_lbl)
        
        plt.show()
        
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (trid==1):

    #yy=s2o3
    #xx=s2o3ha
        
    #xx=ha
    #yy=s2o3
    #zz=np.log10(pos)#o3ha
        
    
        fig=plt.figure()
    #ax=plt.subplot(1,1,1)
    #ax.imshow(is2_m,origin='lower',interpolation='none')
        
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(xx,yy,zz,marker=",",s=1)

    #ax.set_xlim(2.0,4.0)
        ax.set_ylim(-1.0,0.5)
    #ax.set_zlim(-1.0,0.0)

    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('zz')

    #plt.show()



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#def call_rdmodels():
#taus=[    '0.1',    '0.3',    '0.5',    '1.0',    '5.0']

taus=['5.0', '1.0', '0.5', '0.3', '0.1',]

#pre="../../nebumods/ii_set/smc/"
pre="/u/Seneca1/jlopezhe/utring/nebumods/ii_set/smc/"
pos="_.ems.dat"
mod=[("i_50000_50.00_1.00_1.00_0.20_",""),
     ("i_30488_47.00_1.00_1.00_0.20_","")]
fname=pre+mod[0][0]

sel={'a':['s2Tha' ,'o3Thb'],
     'aa':['s2Tha','o357ha'],
     'b':['radx','ha'],
     'c':['radx','s2T'],
     'd':['radx','o357'],
     'e':['radx','s2Tha'],
    'f':['radx','o357ha'],
     'gg':['s2Tha','o3Tha'],
     'hh':['o2Thb','o3Thb'],
     'ii':['o3To2T','o3Thb'],
     'jj':['o2To3T','o2Thb'],
     'kk':['o3To2T','o2Thb'],
     'll':['s2Tha','s2To357']
}

lnsty=["-","--","-.",":","-","--","-."]


atx={'s2Tha':'[SII]/H$\alpha$',
     'o3Thb':'[OIII]5007,4959/H$\beta$',
     'o357ha':'[OIII]5007/H$\beta$',
     'radx':'r$_x$',
     'rady':'r$_y$',
     'ha':'H$\alpha$',
     's2T':'[SII]6717,6731',
     'o3Tha':'[OIII]5007,4959/H$\\alpha$',
     'o3To2T':'[OIII]5007,4959/H$\\beta$',
     'o2Thb':'[OII]3727/H$\\beta$'}



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def test_sb99():
    pfx="../../nebumods/sb99/output36725/"
    md=[("i_50000_50.00_1.00_1.00_0.20_0.1_.ems.dat",""),
         ("i_50000_50.00_1.00_1.00_0.20_0.5_.ems.dat",""),
         ("i_50000_50.00_1.00_1.00_0.20_5.0_.ems.dat","")]


    fig=plt.figure()
    ax=fig.add_subplot(111)

    for m in md:
        nm=pfx+m[0]
        print nm
        #d2_oplt_mod(nm,ax,sel['aa'][0],sel['aa'][1],col='g',od=0)
        d2_oplt_mod(nm,ax,sel['kk'][0],sel['kk'][1],col='g',od=0)

    fig.show()


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def test_sulfabund():
    pfx="../../nebumods/ii_set/absul/"
    md=[("i_50000_50.00_1.0_2.00_5.57_5.0_.ems.dat",""),
        ("i_50000_50.00_1.0_2.00_5.50_5.0_.ems.dat",""),
        ("i_50000_50.00_1.0_2.00_5.30_5.0_.ems.dat",""),
        ("i_50000_50.00_1.0_2.00_5.10_5.0_.ems.dat",""),
        ("i_44616_50.00_1.0_2.00_5.57_5.0_.ems.dat",""),
        ("i_44616_50.00_1.0_2.00_5.50_5.0_.ems.dat",""),
        ("i_44616_50.00_1.0_2.00_5.30_5.0_.ems.dat",""),
        ("i_44616_50.00_1.0_2.00_5.10_5.0_.ems.dat",""),
        ("i_40062_50.00_1.0_2.00_5.57_5.0_.ems.dat",""),
        ("i_40062_50.00_1.0_2.00_5.50_5.0_.ems.dat",""),
        ("i_40062_50.00_1.0_2.00_5.30_5.0_.ems.dat",""),
        ("i_40062_50.00_1.0_2.00_5.10_5.0_.ems.dat","")]


    fig=plt.figure()
    ax=fig.add_subplot(111)

    c=0
    for m in md[0:4]:
        nm=pfx+m[0]
        print nm
        d2_oplt_mod(nm,ax,sel['aa'][0],sel['aa'][1],col='g',od=0,lsty=lnsty[c])
        c=c+1

    c=0
    for m in md[4:8]:
        nm=pfx+m[0]
        print nm
        d2_oplt_mod(nm,ax,sel['aa'][0],sel['aa'][1],col='b',od=0,lsty=lnsty[c])
        c=c+1

    c=0
    for m in md[8:12]:
        nm=pfx+m[0]
        print nm
        d2_oplt_mod(nm,ax,sel['aa'][0],sel['aa'][1],col='r',od=0,lsty=lnsty[c])
        c=c+1




    ax.set_xlim(-2.5,1.0)
    ax.set_ylim(-2.5,1.0)

    fig.show()





#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def bidim():
    '''
    sel={'a':['s2Tha' ,'o3Thb'],
         'aa':['s2Tha','o357ha'],
         'b':['radx','ha'],
         'c':['radx','s2T'],
         'd':['radx','o357'],
         'e':['radx','s2Tha'],
         'f':['radx','o357ha'],
         'gg':['s2Tha','o3Tha'],
         'hh':['o2Thb','o3Thb'],
         'ii':['o3To2T','o3Thb'],
         'jj':['o2To3T','o2Thb']
    }
    '''
    #cual=sel[comb]
    #xxr=cual[0]#'s2Tha' 
    #yyr=cual[1]#'o3Thb'
    fname=pre+mod[0][0]#+taus[-1]+pos

    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #d2_oplt_mod(fname,ax,xxr,yyr)
    #d2_oplt_mod(fname,ax,sel['b'][0],sel['b'][1],col='r')
    #d2_oplt_mod(fname,ax,sel['c'][0],sel['c'][1],col='g')
    #d2_oplt_mod(fname,ax,sel['d'][0],sel['d'][1],col='b')

    #d2_oplt_mod(fname,ax,sel['a'][0],sel['a'][1])
    #d2_oplt_mod(fname,ax,sel['aa'][0],sel['aa'][1],col='g',od=1)
    #d2_oplt_mod(fname,ax,sel['e'][0],sel['e'][1],col='g')
    #d2_oplt_mod(fname,ax,sel['gg'][0],sel['gg'][1],col='b')
    #d2_oplt_mod(fname,ax,sel['hh'][0],sel['hh'][1],col='b',od=1)
    #d2_oplt_mod(fname,ax,sel['ii'][0],sel['ii'][1],col='b',od=1)
    d2_oplt_mod(fname,ax,sel['jj'][0],sel['jj'][1],col='b',od=3)
    #d2_oplt_mod(fname,ax,sel['ll'][0],sel['ll'][1],col='g',od=3)

    fig.show()


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def tridim():
    sel={'a':['s2Tha' ,'o3Thb'],
         #'aa':['s2Tha','o357ha','s2To357'],
         'aa':['s2Tha','o357ha','ha'],
         'b':['radx','ha'],
         'c':['radx','s2T'],
         'd':['radx','o357'],
         'e':['radx','s2Tha'],
         'f':['radx','o357ha'],
         'gg':['s2Tha','o3Tha'],
         'hh':['o2Thb','o3Thb'],
         'ii':['o3To2T','o3Thb'],
         'jj':['o2To3T','o2Thb']
    }

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    fname=pre+mod[0][0]#+taus[-1]+pos
    
    D3_oplot_model(fname,ax,sel['aa'][0],sel['aa'][1],sel['aa'][1],col='g',od=1)
    ax.set_xlim(-2.5,-0.5)
    ax.set_zlim(-0.01,0.5)
    fig.show()
    
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def d2_oplt_mod(fname,ax,xxr,yyr,col='k',od=0,lsty="-"):
    #dat=read_model(fname,xxr,yyr)
    if (od==3):
        ta=np.arange(len(taus))
    elif (od==1):
        ta=[0]#taus[:1]
    elif (od==0):
        ta=''
    
    if(len(ta)>0):
        for t in ta:
            fnm=fname+taus[t]+pos
            print "thefilename:",fnm
            dat=read_model(fnm,xxr,yyr)
            ax.plot(dat[0],dat[1],'b',color=col,linestyle=lsty)
            ax.plot(dat[0][0],dat[1][0],'ko')
    elif(len(ta)==0):
        dat=read_model(fname,xxr,yyr)
        ax.plot(dat[0],dat[1],'b',color=col,linestyle=lsty)
        ax.plot(dat[0][0],dat[1][0],'ko')
    ax.set_xlabel(atx[xxr])
    ax.set_ylabel(atx[yyr])



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def D3_oplot_model(fname,ax,xxr,yyr,zzr,col='k',od=0):
    if (od==1):
        ta=taus
    else:
        ta=taus[:1]

    for t in ta:
        fnm=fname+t+pos
        dat=read_model(fnm,xxr,yyr,zzr)
        ax.plot(dat[0],dat[1],dat[2],color=col)
        #ax.plot(dat[0][0],dat[1][0],dat[2][0],'ko')

    
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#returns the lines associated with a given model. 
def read_model(fname,xx,yy,zz="",log=1):
    dat=np.loadtxt(fname,skiprows=1)
    i=1
    names={'radx'   :'depth_xx',
           'oIIT'   :'TOTL_3727A',
           'oIII49' :'O_3_4959A',
           'oIII57' :'O_3_5007A',
           'sII17'  :'S_II_6716A',
           'sII31'  :'S_II_6731A',
           'sIIT'   :'S_2_6720A',
           'hb'     :'H_1_4861A',
           'ha'     :'H_1_6563A',
           "heII86" :'HE_2_4686A'}

    lineID=['depth_xx','depth_yy',
            'TOTL_3727A',
            'O_II_3726A',	
            'O_II_3729A',	
            'H_1_6563A',	
            'H_1_4861A',	
            'HE_1_5876A',	
            'HE_2_4686A',	
            'TOTL_4363A',	
            'O_3_5007A',	
            'O_3_4959A',	
            'S_II_6716A',	
            'S_II_6731A',	
            'S_2_6720A',	
            'NE_3_3869A',	
            'AR_3_7135A',
            'O_1_6300A',	
            'N_2_6548A',	
            'N_2_6584A',	
            'O_II_7323A',	
            'O_II_7332A',	
            'S_3_9069A',	
            'S_3_6312A',
            'UNK']
    
    radx=dat[i:,lineID.index('depth_xx')]
    o3T=dat[i:,lineID.index('O_3_5007A')]+dat[i:,lineID.index('O_3_4959A')*0.407]
    ha=dat[i:,lineID.index('H_1_6563A')]
    hb=dat[i:,lineID.index('H_1_4861A')]
    s2T=dat[i:,lineID.index('S_2_6720A')]
    o2T=dat[i:,lineID.index('TOTL_3727A')]
    o357=dat[i:,lineID.index('O_3_5007A')]

    #pdb.set_trace()
    
    rat={'radx':radx,
         'o3Tha':o3T/ha,
         'o2Tha':o2T/ha,
         'o3Thb':o3T/hb,
         'o2Thb':o2T/hb,
         'o2To3T':o2T/o3T,
         'o3To2T':o3T/o2T,
         's2Tha':s2T/ha,
         'o357ha':o357/ha,
         's2To357':s2T/o357,
         's2To3T':s2T/o3T,
         'o3T':o3T,
         'o357':o357,
         's2T':s2T,
         'ha':ha,
         'hb':hb,
         'o2T':o2T}

    mnha=np.mean(ha[0:20])
    limha=mnha*0.01
    valid=np.ma.masked_where(ha<limha,ha)
    comp=np.ma.compressed(valid)
    print mnha,limha
    #ixlim=np.argmax(valid)
    print "len",len(ha),len(comp)
    ixlim=-1
    ixlim=len(comp)
    
    xxdat=rat[xx]
    yydat=rat[yy]
    if(log==0):
        xxdat=rat[xx]
        yydat=rat[yy]
    else:
        xxdat=np.log10(rat[xx])
        yydat=np.log10(rat[yy])

    #pdb.set_trace()
    
    
    if (len(zz)>1):
        if(log==0):
            zzdat=rat[zz]
        else:
            zzdat=np.log10(rat[zz])
        return [xxdat[:ixlim],yydat[:ixlim],zzdat[:ixlim]]
        #return [xxdat[:],yydat[:],zzdat[:]]

    else:
        return [xxdat[:ixlim],yydat[:ixlim]]
        #return [xxdat[:],yydat[:]]

'''
    xstr=xx.split('/')
    ystr=yy.split('/')
    zstr=zz.split('/')

    xnum=xstr[0]
    xden=xstr[1]
    ynum=ystr[0]
    yden=ystr[1]

    if(len(zstr[0]>0)):
        znum=ystr[0]
        zden=ystr[1]


    req=[xx,yy,zz]

    for r in req:
        if (r=='oIIIT'):
            i01=lineID.index('O_3_5007A')
            i02=lineID.index('O_3_4959A')
            p1=dat[i:,i01]
            p2=dat[i:,i02]
            px=pi+p2
        else:
 '''           



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def oplot_model(fname,axe,xxrt,yyrt,ann=0,label="",get_data=0,klr='black'):
    dat=np.loadtxt(fname,skiprows=1) 
    
    i=1
    O2TOT=dat[i:,2]
    OII=dat[i:,3]+dat[i:,4]
    OII73=dat[i:,21]+dat[i:,22]
    OIII=dat[i:,10]
    SII=dat[i:,12]+dat[i:,13]
    Ha=dat[i:,5]
    Hbeta=dat[i:,6]

    NII=dat[i:,20]#+dat[i:,19]
    OIII_45=dat[i:,10]+dat[i:,11]
    #Rad=dat[i:,0]
    
    NII=dat[i:,20]#+dat[i:,19]
    OIII_45=dat[i:,10]+dat[i:,11]
    Rad=dat[i:,0]

    razon={"o3ha":  OIII/Ha,
           "o2o3":  OII/OIII,
           "o23ha": (OIII+OII)/Ha,
           "o2ha":  OII/Ha,
           "s2ha":  SII/Ha,
           "s2o3":  SII/OIII,
           "s2o2":  SII/OII,
           "o3o2":  OIII/OII,
           "o3hb": OIII/Hbeta,
           "o2hb":  OII/Hbeta,
           "n2ha":  NII/Ha,
           "n2o2":  NII/OII,
           "r23":  (OII+OIII_45)/Hbeta,
           "rad": Rad,
           "o2totha":O2TOT/Ha,
           "o273ha":OII73/Ha,
           "ha":Ha,
           "o3":OIII,
           "s2":SII}

    xxdata=np.log10(razon[xxrt])
    yydata=np.log10(razon[yyrt])

    if(get_data==0):
        f=fname.split('/')[-1]
        txt=""
        if (ann==1):
            tks=f.split('_')
        #txt="T*="+tks[1]+" Q="+tks[2]+" Ne="+tks[4]+" Z="+tks[5]+"$\odot$"
            txt=label
            axe.plot(xxdata,yydata,label=txt,color=klr)
    else:
        return np.array([xxdata,yydata])

#>>>>>


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def test_models():
#    mod=[("c_50000_50.00_01_1.00_0.2_10.0_.ems.dat","O3*  50kK Q50.0 0.3Z"),
         #("c_44616_49.63_01_1.00_0.2_10.0_.ems.dat","O3.0 44kK Q49.0 0.3Z"),
         #("c_40062_48.96_01_1.00_0.2_10.0_.ems.dat","O5.5 40kK Q48.4 0.3Z"),
         #("c_34419_48.29_01_1.00_0.2_10.0_.ems.dat","O7.5 34kK Q48.2 0.3Z"),
#         ("c_30488_47.56_01_1.00_0.2_10.0_.ems.dat","O9.5 30kK Q47.5 0.3Z")]
#    pre='../../nebumods/esb_emsdat/'

    pre="../../nebumods/ii_set/smc/"
    pos="_.ems.dat"
    
    mod=[("i_50000_50.00_1.00_1.00_0.20_",""),
         ("i_30488_47.00_1.00_1.00_0.20_","")]


    taus=["0.1","0.3","0.5","1.0","5.0"]
    #taus=["5.0"]

    fxx=1
    fyy=1
    pxxlim=(-2.5,1.0)
    pyylim=(-2.5,1.0)

    xxrt_nm="rad"
    yaa_nm="ha"
    ybb_nm="s2"
    ycc_nm="o3"
    ydd_nm="s2ha"
    yee_nm="o3ha"
    yff_nm="s2o3"

    #yfour=""
    ax=plt.subplot(fyy,fxx,1)
    #bx=plt.subplot(fyy,fxx,2)
    #cx=plt.subplot(fyy,fxx,5)

#    dx=plt.subplot(fyy,fxx,2)
#    ex=plt.subplot(fyy,fxx,4)
#    fx=plt.subplot(fyy,fxx,6)

    for star in mod:
        '''
        ha=oplot_model(pre+star[0],ax,xxrt_nm,yaa_nm,get_data=1)
        s2=oplot_model(pre+star[0],ax,xxrt_nm,ybb_nm,get_data=1)
        o3=oplot_model(pre+star[0],ax,xxrt_nm,ycc_nm,get_data=1)
        s2ha=oplot_model(pre+star[0],ax,xxrt_nm,ydd_nm,get_data=1)
        o3ha=oplot_model(pre+star[0],ax,xxrt_nm,yee_nm,get_data=1)
        s2o3=oplot_model(pre+star[0],ax,xxrt_nm,yff_nm,get_data=1)

        #s2_o3=s2s2[1]-o3o3[1]
        #rs=haha[0,-1]
        #print rs,mod[0]

        #ax.plot(np.power(m[0],10),np.power(m[1],10))
        #m=[m_r[0]*1e17,m_r[1]*1e17]
        ax.plot(ha[0],ha[1],",-")#,s=1)
        bx.plot(s2[0],s2[1],",-")#,s=1)
        cx.plot(o3[0],o3[1],",-")#,s=1)
        dx.plot(s2ha[0],s2ha[1],",-")
        ex.plot(o3ha[0],o3ha[1],",-")
        fx.plot(s2o3[0],s2o3[1],",-")
        '''
        '''
        ha=oplot_model(pre+star[0],ax,xxrt_nm,yaa_nm,get_data=1)
        s2o3=oplot_model(pre+star[0],ax,xxrt_nm,yff_nm,get_data=1)
        fst=s2o3[1]-ha[1]
        sec=s2o3[1]+ha[1]
        ax.plot(ha[0],fst,",-")#,s=1)
        bx.plot(ha[0],sec,",-")
        '''
        
        for ta in taus:
            fnm=pre+star[0]+ta+pos
            ha=oplot_model(fnm,ax,xxrt_nm,yaa_nm,get_data=1)
            #        s2=oplot_model(pre+star[0],ax,xxrt_nm,ybb_nm,get_data=1)
            #        o3=oplot_model(pre+star[0],ax,xxrt_nm,ycc_nm,get_data=1)
            s2o3=oplot_model(fnm,ax,xxrt_nm,yff_nm,get_data=1)
        #        tre=(s2[1]+o3[1])-ha[1]
            ax.plot(ha[1],s2o3[1],",-")
    ax.set_xlabel("log(Ha)")
    ax.set_ylabel("log(s2/o3)")


    plt.show()



#>>>>>



#def tensinx():

    


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#def models()
