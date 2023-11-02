from matplotlib import lines
import scipy
import cmath
import math
from math import pi, sin, cos, sqrt, exp
import matplotlib

import numpy as np
import progressbar

import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.path import Path
from matplotlib.patches import PathPatch
'''
# int
'''
a = 2.46               #A
b = 4*pi/(sqrt(3)*a)    #1/A, dot(a,b)=sqrt(3)a*b/2 = 2pi
S = sqrt(3)/2 * a**2    #A^2
a1 = np.array([a*sqrt(3)/2,  -a/2])
a2 = np.array([a*sqrt(3)/2,  a/2])
l_list=np.array([a1,a2])

def rotate(in_list,theta):
    # rotate k theta in raduis
    T_rotate=np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    out_list=[]
    for i in range(2):
        k = in_list[i]
        kr = np.matmul(T_rotate,k)
        out_list.append( kr )
    out_list = np.array(out_list)
    return out_list

def rotate_2D(in_list,theta):
    # rotate k theta in raduis
    T_rotate=np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    kr = np.matmul(T_rotate, in_list)
    out_list = np.array(kr)
    return out_list

def rotate_3D(in_list,theta):
    # rotate k theta in raduis
    T_rotate=np.array([[cos(theta),-sin(theta), 0],[sin(theta),cos(theta), 0], [0, 0, 1]])
    kr = np.matmul(T_rotate, in_list)
    out_list = np.array(kr)
    return out_list

def strian(in_list,theta,ex,ey):
    # give strain ex in diraction theta
    # and strain ey in diraction theta+pi/2
    T_rotate=np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    T_rotate_back=np.array([[cos(-theta),-sin(-theta)],[sin(-theta),cos(-theta)]])
    T_strian = np.array([[1/(1+ex),0],[0,1/(1+ey)]])
    T_sum = np.matmul(T_rotate,np.matmul(T_strian,T_rotate_back))
    out_list=[]
    for i in range(2):
        k = in_list[i]
        kr = np.matmul(T_sum,k)
        out_list.append( kr )
    out_list = np.array(out_list)
    return out_list

def gen_revise_vectors(input_vectors):
    """
    Generate reciprocal lattice vectors from real-space lattice vectors.
    or Generate real-space lattice vectors from reciprocal lattice vectors.

    NOTE: Here we evaluate reciprocal lattice vectors via
        dot_product(a_i, b_j) = 2 * pi * delta_{ij}
    The formulae based on cross-products are not robust in some cases.

    :param input_vectors: (3, 3) float64 array
        Cartesian coordinates of input lattice vectors
    :return: output_vectors: (3, 3) float64 array
        Cartesian coordinates of output lattice vectors.
    """
    output_vectors = np.zeros((2, 2))
    product = 2 * pi * np.eye(2)
    for i in range(2):
        output_vectors[i] = np.linalg.solve(input_vectors, product[i])
    return output_vectors

def cart2frac(k_base, k_in):
    """
    Convert k_in to combination of k_base.

    :param k_base: (2, 2) float64 array
        Cartesian coordinates of base k vectors
    :param k_in: (num_coord, 2) float64 array
        Cartesian coordinates to convert
    :return: fractional_coordinates: (num_coord, 3) float64 array
        fractional coordinates in basis of lattice vectors
    """
    fractional_coordinates = np.zeros(k_in.shape)
    conversion_matrix = np.linalg.inv(k_base.T)
    for i, row in enumerate(k_in):
        fractional_coordinates[i] = np.matmul(conversion_matrix, row.T)
    return fractional_coordinates

def plot_reciprocal(k_top,k_bottom,k_moire,k_list):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    #   Gamma vector
    ax.arrow(0, 0, k_top[0,0]-k_list[0,0], k_top[0,1]-k_list[0,1], head_width=0.01, head_length=0.01,fc='r', ec='r',label="dK_top")
    ax.arrow(0, 0, k_top[1,0]-k_list[1,0], k_top[1,1]-k_list[1,1], head_width=0.01, head_length=0.01,fc='r', ec='r')
    ax.arrow(0, 0, k_bottom[0,0]-k_list[0,0], k_bottom[0,1]-k_list[0,1], head_width=0.01, head_length=0.01,fc='b', ec='b',label="dK_bottom")
    ax.arrow(0, 0, k_bottom[1,0]-k_list[1,0], k_bottom[1,1]-k_list[1,1], head_width=0.01, head_length=0.01,fc='b', ec='b')
    ax.arrow(0, 0, k_moire[0,0], k_moire[0,1], head_width=0.01, head_length=0.01,fc='k', ec='k',label="K_Moire")
    ax.arrow(0, 0, k_moire[1,0], k_moire[1,1], head_width=0.01, head_length=0.01,fc='k', ec='k')
    
    plt.text(0.5, 0.35,'k_top=\n'+str(cart2frac(k_moire, k_top))+"\n*k_moire", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
    plt.text(0.5, 0.15, 'k_bottom=\n'+str(cart2frac(k_moire, k_bottom))+"\n*k_moire", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
    plt.legend()
    dk1=np.linalg.norm(k_moire[0,:])
    dk2=np.linalg.norm(k_moire[1,:])
    kmax=sqrt(3)*max(dk1,dk2)
    plt.xlim((-kmax, kmax)) 
    plt.ylim((-kmax, kmax)) 
    plt.xlabel("kx (1/A)")
    plt.ylabel("ky (1/A)")
    plt.savefig(fname='reciprocal_space.png',dpi=600)
    plt.close

def plot_lattice(l_top,l_bottom,l_moire):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    #   Gamma vector
    ax.arrow(0, 0, l_top[0,0], l_top[0,1], head_width=0.01, head_length=0.01,fc='r', ec='r',label="l_top")
    ax.arrow(0, 0, l_top[1,0], l_top[1,1], head_width=0.01, head_length=0.01,fc='r', ec='r')
    ax.arrow(0, 0, l_bottom[0,0], l_bottom[0,1], head_width=0.01, head_length=0.01,fc='b', ec='b',label="l_bottom")
    ax.arrow(0, 0, l_bottom[1,0], l_bottom[1,1], head_width=0.01, head_length=0.01,fc='b', ec='b')
    ax.arrow(0, 0, l_moire[0,0], l_moire[0,1], head_width=0.01, head_length=0.01,fc='k', ec='k',label="l_Moire")
    ax.arrow(0, 0, l_moire[1,0], l_moire[1,1], head_width=0.01, head_length=0.01,fc='k', ec='k')
    
    plt.text(0.5, 0.35,'L_moire=\n'+str(cart2frac(l_top, l_moire))+"\n*L_top", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
    plt.text(0.5, 0.15, 'L_moire=\n'+str(cart2frac(l_bottom, l_moire))+"\n*L_bottom", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
    plt.legend()
    L1=np.linalg.norm(l_moire[0,:])
    L2=np.linalg.norm(l_moire[1,:])
    theta= np.arccos(np.inner(l_moire[0,:],l_moire[1,:]) /(L1*L2))
    lmax=max(L1,L2)
    plt.xlim((-lmax, lmax)) 
    plt.ylim((-lmax, lmax))
    plt.xlabel("x (A)")
    plt.ylabel("y (A)") 
    plt.title("L1: %3.3f A  L2:% 3.3f A\ntheta:%3.3f °" %(L1,L2, 180*theta/pi))
    plt.savefig(fname='real_space.png',dpi=600)
    plt.close

def normalization_lattice(l_top,l_bottom,l_moire):

    #print('old_L_moire=\n'+str(cart2frac(l_top, l_moire))+"\n*L_top")
    #print('old_L_moire=\n'+str(cart2frac(l_bottom, l_moire))+"\n*L_bottom")

    Matrixtop=cart2frac(l_top, l_moire)
    Matrixbottom=cart2frac(l_bottom, l_moire)
    topmin =min(np.min((np.abs(Matrixtop))),1)
    bottomin=min(np.min((np.abs(Matrixbottom))),1)
    l_moire_new = l_moire/topmin
    for i in range(0,2):
        for j in range(0,2):
            Matrixtop[i,j] = int(Matrixtop[i,j]/topmin)
            Matrixbottom[i,j] = int(Matrixbottom[i,j]/bottomin)
    l_top_new = np.matmul(np.linalg.inv(Matrixtop),l_moire_new)
    l_bottom_new = np.matmul(np.linalg.inv(Matrixbottom),l_moire_new)

    #print('L_moire=\n'+str(cart2frac(l_top_new, l_moire_new))+"\n*L_top")
    #print('L_moire=\n'+str(cart2frac(l_bottom_new, l_moire_new))+"\n*L_bottom")
    return l_top_new,l_bottom_new,l_moire_new

class tBG():
    def __init__(self):
        """
        initialize all required parameters
        """
        # def hopping constants
        self.a0 =  1.42            #: [A] C-C bond length
        self.a  =  2.46            #: [A] C-C cell length
        self.d0 =  3.35             #: [A] interlayer distance normal:3.35
 
    def Lattice(self,l_top,l_bottom,l_moire,topx,topy,botx,boty,zstrain,twist):
        """
        bilayerGraphene-hBN lattice.
        orbital_coords: orbital_coords[2i] is sub-lattice A,
                        orbital_coords[2i+1] is sub-lattice B,
                        and spin down part is a copy of spin up. 
        """
        L = np.linalg.norm(l_moire[0,:])            #: [A] Moire period length     
        print ('Moire period/nm=',L)

        def judge_and_swap(l_moire):
            L1 = np.array([l_moire[0,0], l_moire[0,1], 0])
            L2 = np.array([l_moire[1,0], l_moire[1,1], 0])
            cross_a1_a2 = np.cross(L1,L2)
            if cross_a1_a2[2] > 0: #z value of cross product of a1, a2
                #print("a2 on the anti-clockwise direction of a2, no swapping needed")
                return l_moire
            if cross_a1_a2[2] < 0:
                #print("a2 on the clockwise direction of a1, now swap a1 and a2")
                l_moire[0,0], l_moire[1,0] = l_moire[1,0], l_moire[0,0]
                l_moire[0,1], l_moire[1,1] = l_moire[1,1], l_moire[0,1]
                return l_moire
        
        l_moire = judge_and_swap(l_moire)
        #print("Swapped l_moire matrix is\n",l_moire)
        
        a1= np.array([l_moire[0,0], l_moire[0,1], 0])              #: [A] Supercell vector
        a2= np.array([l_moire[1,0], l_moire[1,1], 0])              #: [A] Supercell vector
        a3= np.array([0, 0, 6*self.d0])      #: [A] Supercell vector 
        
        theta_old_1 = np.arctan2(a1[1], a1[0])#angle with respect to x-axis of original a1 vector
        theta_old_2 = np.arctan2(a2[1], a2[0]) #angle with respect to x-axis of original a2 vector
        
        #print("the old theta of a1 and a2 w.r.t. x-axis is", theta_old_1, theta_old_2)
        
        theta_rotate = -theta_old_1
        #print("The rotate theta is", theta_rotate)
        
        l_moire = rotate(l_moire, theta_rotate)
        
        a1= np.array([l_moire[0,0], l_moire[0,1], 0])              #: [A] Supercell vector
        a2= np.array([l_moire[1,0], l_moire[1,1], 0])   
        self.vectors = [a1, a2, a3]
        #print('lattice_vectors is ',self.vectors)
        
        r1 = l_bottom[0,:]   #: [nm] layer1 vector
        r2 = l_bottom[1,:]   #: [nm] layer1 vector
        r1p = l_top[0,:]     #: [nm] layer3 vector
        r2p = l_top[1,:]     #: [nm] layer3 vector
        
       #print(r1,r2,r1p,r2p)
        
        r1 = rotate_2D(r1, theta_rotate)
        r2 = rotate_2D(r2, theta_rotate)
        r1p = rotate_2D(r1p, theta_rotate)
        r2p = rotate_2D(r2p, theta_rotate)
        #print(r1,r2,r1p,r2p)
        
        l_bottom[0,:] = r1
        l_bottom[1,:] = r2
        l_top[0,:] = r1p
        l_top[1,:] = r2p
        
        delta =np.append((r1 +r2 )/3,0)     #: [nm] layer1 AB vector
        deltap=np.append((r1p+r2p)/3,0)     #: [nm] layer2 AB vector
        height=np.array([0, 0, self.d0])    #: [nm] height vector
        #print(r1,r2,delta)
        

        orbital_coords = []
        L1= l_moire[0,:]
        L2= l_moire[1,:]
        SL= np.cross(L1,L2)
        #print(SL,L1,L2)
        
        def judge(orb, L1, L2):
            """
            private method:
            move the bottom-layer/layer1,2 i,j atom location
              or the    top-layer/layer3,4 j,i atom location
            """
            if (0<=np.cross(L1,orb)<SL) and (0<=np.cross(orb,L2)<SL):
                return True
            else:
                #print(np.cross(L1,orb)/SL,np.cross(orb,L2)/SL)
                return False
        bar = progressbar.ProgressBar(max_value=1600)
        print("Making TBG sample.")
        for  i in range(-800,800):
            for j in range(-800,800):
                rr1=i*r1 +j*r2
                if judge(rr1, L1, L2):
                    ra1 = np.append(rr1,-self.d0/2)
                    rb1=ra1+delta
                    orbital_coords.append(ra1) # A1
                    orbital_coords.append(rb1) # B1

                rr3=i*r1p+j*r2p
                if judge(rr3, L1, L2):
                    ra3 = np.append(rr3,self.d0/2)
                    rb3=ra3+deltap
                    orbital_coords.append(ra3) # A3
                    orbital_coords.append(rb3) # B3
            bar.update(i+800)
        #print("Done1!")

        #-Near Origin=ABBC-
        #--A3--B3--
        #--A1--B1--
        self.n_atom = len(orbital_coords)
        #print(self.n_atom)
        #print(len(orbital_coords))
        #fw = open("strain_TBG_"+str(int(round(L/10,0)))+"nm.data",'w+')
        fw = open(f"./TBG_top_x_{topx}_topy_{topy}_botx_{botx}_boty_{boty}_zstrain_{zstrain}_twist_{twist}/Strain.data",'w+')
        fw.write(("#L/Angstroms = %.4f" %(float(L))))
        fw.write(("\n%d atoms" %(int(len(orbital_coords)))))
        fw.write(("\n4 atom types"))
        fw.write(("\n"))

        fw.write(("\n%10.8f %10.8f   xlo xhi" %(float(0.),float(L1[0]))))
        fw.write(("\n%10.8f %10.8f   ylo yhi" %(float(0.),float(L2[1]))))
        fw.write(("\n%10.8f %10.8f   zlo zhi" %(float(-a3[2]),float(a3[2]))))
        fw.write(("\n"))

        fw.write(("\n %10.7f %10.7f %10.7f  xy xz yz" %(float(L2[0]),float(0.),float(0.))))
        fw.write(("\n"))
        
        fw.write(("\nAtoms"))
        fw.write(("\n"))
        #print("Done2!")
        for i in range(self.n_atom):
            coord = orbital_coords[i]
            if coord[2] >0:
                type =2
            else:
                type =0
            fw.write(('\n%6d %4d      %12.6f %12.6f %12.6f' 
                    %(int(i+1),
                    int(type+i%2+1),
                    float(coord[0]), 
                    float(coord[1]), 
                    float(coord[2])) 
                    ))
        #print("Done3!")
    def Sample(self,l_top,l_bottom,l_moire, topx,topy,botx,boty,zstrain,twist):
        """
        Calculate and plot tBG sample 
        """        
        self.Lattice(l_top,l_bottom,l_moire, topx,topy,botx,boty,zstrain,twist)

def make_sample(l_top,l_bottom,l_moire,topx,topy,botx,boty,zstrain,twist):
    tBG_sample = tBG()
    tBG_sample.d0 = zstrain
    tBG_sample.Sample(l_top,l_bottom,l_moire, topx,topy,botx,boty,zstrain,twist)
    return  tBG_sample

def make_strain_twisted_sample(ktop_x, ktop_y, kbot_x, kbot_y, twist_angle,z_strain):
    
    t = twist_angle*pi/180
    print("twist angle: %3.3f °"%(t*180/pi))

    k_list   = gen_revise_vectors(l_list)
    #k_top    = rotate(strian(k_list,0,x*t,y*t),t)
    #k_bottom = strian(k_list,0,z*t,0)
    k_top    = rotate(strian(k_list, 0, ktop_x, ktop_y),t)
    #k_top = strian(k_list,0, ktop_x, k_top_y)
    k_bottom = strian(k_list,0, kbot_x, kbot_y)
    k_moire  = k_top-k_bottom
    plot_reciprocal(k_top,k_bottom,k_moire,k_list)
 
    l_top = gen_revise_vectors(k_top)
    l_bottom = gen_revise_vectors(k_bottom)
    l_moire = gen_revise_vectors(k_moire)
    print("l_top is:",l_top)
    print("l_bot is:",l_bottom)
    l_top_new,l_bottom_new,l_moire_new=normalization_lattice(l_top,l_bottom,l_moire)
    print("l_top_new is:",l_top_new)
    print("l_bot_new is:",l_bottom_new)
    plot_lattice(l_top_new,l_bottom_new,l_moire_new)

    make_sample(l_top_new,l_bottom_new,l_moire_new, ktop_x,ktop_y,kbot_x,kbot_y,z_strain,twist_angle)