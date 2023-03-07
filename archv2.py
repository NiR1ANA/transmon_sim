#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from qutip import *
from qutip.qip import *
from tqdm.auto import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp


# In[ ]:


class tmon_fn:
    #init variables
    def __init__(self, N, Nlevel):
        #Energy scale: Ej = 20Ghz +- 10Ghz, Ec = 250MHz, T = (5-20)MHz 
        #ratio = 200/2.5/.1 
        #therefore, 1 in sim = 100MHz
        self.N = N
        self.Nlevel = Nlevel
        self.Ej =  np.absolute(np.random.normal(200,20, N))
        self.Ec = np.absolute(np.random.normal(2.5,0.2, N))
        
    
    def op_na(self):
        na = []
        Neye = []
        for i in  range(self.N):
            Neye.append(qeye(self.Nlevel))
        for i in range(self.N):
            temp = Neye.copy()
            temp[i] = num(self.Nlevel)
            na.append(tensor(temp))
        return na
    
    def op_a(self):
        Neye = []
        a = []
        for i in  range(self.N):
            Neye.append(qeye(self.Nlevel))
        for i in range(self.N):
            temp = Neye.copy()
            temp[i] = destroy(self.Nlevel)
            a.append(tensor(temp))
        return a
    
    def op_nid(self):
        #n dim id operator
        Neye = []
        for i in  range(self.N):
            Neye.append(qeye(self.Nlevel))
        nid = tensor(Neye)
        return nid
    
    def H_c(self):
        #constant part of hamiltonian
        N = self.N
        na = self.op_na()
        #initialize v ,t
        v = np.zeros(N)
        for i in range(N):
            v[i] = math.sqrt(8*self.Ej[i]*self.Ec[i])
        
        H1 = 0
        H2 = 0
        for i in range(N):
            H1 = H1 + na[i]*v[i]
            H2 = H2 + self.Ec[i]*(na[i]*(na[i]+1))
            
        H2 = -0.5*H2        
        return H1 + H2
    
    def H_int(self,T,RWA = False,t_matrix = 0, nn2 = False ):
        #t_matrix for custom matrices if required, otherwise nncoupling as usual
        N = self.N
        Ec = self.Ec
        Ej = self.Ej
        a = self.op_a()
        
        if t_matrix == 0:
            t = np.zeros((N,N))
            for i in range(N-1):
                t[i][i+1] = (T/(4*(2*((Ec[i]*Ec[i+1])**1/float(2)))**(1/float(2))))*((Ej[i]*Ej[i+1])**(1/float(4)))
            
            if nn2:
                for i in range(N-2):
                    t[i][i+2] = (T/(4*(2*((Ec[i]*Ec[i+2])**1/float(2)))**(1/float(2))))*((Ej[i]*Ej[i+2])**(1/float(4)))
            
        else:
            t = t_matrix
        
        H3 = 0
        if RWA:
            for i in range(N):
                for j in range(N):
                    H3 = H3 + t[i][j]*(a[i]*a[j].dag()+a[i].dag()*a[j])
            
        if not RWA:
            for i in range(N):
                for j in range(N):
                    H3 = H3 + t[i][j]*(a[i] + a[i].dag())*(a[j].dag()+a[j])
        return H3
    
    
    def t_matrix_setup(self, arr=0, T = 0):
        #arr is array of qubit index to be coupled
        N = self.N
        Ec = self.Ec
        Ej = self.Ej
        
        if arr == 0:
            arr = np.arange(N)
        else:
            arr = np.sort(arr)
            
        t = np.zeros((N,N))
        for i in range(len(arr)-1):
            t[arr[i]][arr[i+1]] = (T/(4*(2*((Ec[arr[i]]*Ec[arr[i+1]])**1/float(2)))**(1/float(2))))*((Ej[arr[i]]*Ej[arr[i+1]])**(1/float(4)))
            
        return t
    
    def H_arch_int(self,Tx = 0,Tz = 0, RWA = False, x_arr = 0, z_arr = 0 ):
        #t_matrix for custom matrices if required, otherwise nncoupling as usual
        N = self.N
        Ec = self.Ec
        Ej = self.Ej
        a = self.op_a()
        
        tx = self.t_matrix_setup(arr = x_arr, T = Tx )
        tz = self.t_matrix_setup(arr = z_arr, T = Tz)
            
        
        
        H3 = 0
        for i in range(N):
            for j in range(N):
                if tx[i][j] !=0: #a simple check to eliminate unnecessary calculations
                    H3 = H3 + tx[i][j]*(a[i] + a[i].dag())*(a[j].dag()+a[j])
        
        for i in range(N):
            for j in range(N):
                if tz[i][j] !=0:
                    H3 = H3 + tz[i][j]*(a[i]*a[i].dag()*a[j]*a[j].dag())
        
        '''
        H3 = 0
        if RWA:
            for i in range(N):
                for j in range(N):
                    H3 = H3 + t[i][j]*(a[i]*a[j].dag()+a[i].dag()*a[j])
            
        if not RWA:
            for i in range(N):
                for j in range(N):
              
              H3 = H3 + t[i][j]*(a[i] + a[i].dag())*(a[j].dag()+a[j])
        
        '''
        return H3
# In[ ]:


class evol_fns:
    def __init__(self, tarch,tlist,RWA=False,tolerance = 0.1):
        self.tarch = tarch
        self.tlist = tlist
        self.RWA = RWA
        self.tolerance = tolerance
        
    def simple_spectrum_fn(self):
        spectrum_arr = []
        cspectrum_arr = []
        na = self.tarch.op_na()
        tol = self.tolerance
        
        for t in tqdm(self.tlist):
            H3 = self.tarch.H_int(t,RWA = self.RWA)
            H = self.tarch.H_c() + H3
            eigen_vals, eigen_vecs = H.eigenstates()
            spectrum_arr.append([t,eigen_vals,eigen_vecs])
        
        return spectrum_arr
    
    def spectrum_fn(self):
        spectrum_arr = []
        cspectrum_arr = []
        na = self.tarch.op_na()
        tol = self.tolerance
        
        for t in tqdm(self.tlist):
            H3 = self.tarch.H_int(t,RWA = self.RWA)
            H = self.tarch.H_c() + H3
            eigen_vals, eigen_vecs = H.eigenstates()
            spectrum_arr.append([t,eigen_vals,eigen_vecs])
            #cstate check
            c_val = []
            c_vec = []
            labels = []
            for i in range(len(eigen_vals)):
                #first check if already found max number of cstates
                if len(c_val) > 2**len(na):
                    break
                c = 1
                blabel = []
                vec = eigen_vecs[i]
                for nai in na:
                    temp = expect(nai,vec)
                    if not (math.isclose(temp, 1, abs_tol=tol) or math.isclose(temp, 0, abs_tol=tol)):
                        c = 0
                        break
                    blabel.append(temp)
                    
                if c==1 :
                    c_val.append(eigen_vals[i])
                    c_vec.append(eigen_vecs[i])
                    labels.append(blabel)
            
            cspectrum_arr.append([t,c_val,c_vec,labels])
        return spectrum_arr, cspectrum_arr
    
    def sparse_spectrum_fn(self,nstates):
        #nstates is number of eigenvectors to find
        spectrum_arr = []
        na = self.tarch.op_na()
        
        for t in tqdm(self.tlist):
            H1 = self.tarch.H_c() + self.tarch.H_int(t,RWA = self.RWA)
            #print("loading matrix")
            eigen_vals, eigen_vecs = sp.sparse.linalg.eigsh(H1.data, k=nstates,sigma=0)
            spectrum_arr.append([t,eigen_vals,eigen_vecs])
        
        return spectrum_arr
    
    def sparse_cspectrum_fn(self,nstates):
        #nstates is number of eigenvectors to find
        #need to change nai operator to truncated hilbertspace operator
        spectrum_arr = []
        cspectrum_arr = []
        na = self.tarch.op_na()
        tol = self.tolerance
        
        for t in tqdm(self.tlist):
            H3 = self.tarch.H_int(t,RWA = self.RWA)
            H = self.tarch.H_c() + H3
            #print("loading matrix")
            eigen_vals, eigen_vecs = sp.sparse.linalg.eigsh(H.data, k=nstates,sigma=0)
            spectrum_arr.append([t,eigen_vals,eigen_vecs])
        
            #cstate check
            c_val = []
            c_vec = []
            labels = []
            for i in range(len(eigen_vals)):
                #first check if already found max number of cstates
                #if len(c_val) > 2**len(na):
                 #   break
                c = 1
                blabel = []
                vec = eigen_vecs[i]
                for nai in na:
                    temp = expect(nai,vec)
                    if not (math.isclose(temp, 1, abs_tol=tol) and math.isclose(temp, 0, abs_tol=tol)):
                        c = 0
                        break
                    blabel.append(temp)
                    
                if c==1 :
                    c_val.append(eigen_vals[i])
                    c_vec.append(eigen_vecs[i])
                    labels.append(blabel)
            cspectrum_arr.append([t,c_val,c_vec,labels])
        
        return spectrum_arr, cspectrum_arr
        
    def range_spectrum(self,ymin,ymax):
        spectrum_arr = []
        na = self.tarch.op_na()
        tol = self.tolerance
        
        for t in tqdm(self.tlist):
            H3 = self.tarch.H_int(t,RWA = self.RWA)
            H = self.tarch.H_c() + H3
            eigen_vals, eigen_vecs = H.eigenstates()
            #select states only in band..............
            '''
            for i in range(len(eigen_vals)):
                temp = eigen_vals[i]
                if (temp<ymin) or (temp > ymax):
                    np.delete(eigen_vals,i)
                    np.delete(eigen_vecs,i)
            '''
            e_val = []
            e_vec = []
            for i in range(len(eigen_vals)):
                temp = eigen_vals[i]
                if (temp>ymin) and (temp< ymax):
                    e_val.append(eigen_vals[i])
                    e_vec.append(eigen_vecs[i])
            #........................................       
            spectrum_arr.append([t,e_val,e_vec])
        return spectrum_arr
        


# In[ ]:


class plot_spectrum:    
    def __init__(self, N ,Nlevel,spectrum_arr):
            self.title = str(str(N) + " tranmons " + str(Nlevel)+ " level systems")
            self.x_label = str("Coupling strength, 1 = 100MHz")
            self.y_label = str("eigenvalues, 1 = 100MHz")
            self.size = 10
            self.c = 'b'
            self.x = 0
            self.y = 0
            self.label = 0
            self.spectrum_arr = spectrum_arr
            self.spectrum_data()

    def spectrum_data(self):
        #spectrum data ->matplotlib format
        x = []
        y = []
        label = []
        fig = 0
                
        if len(self.spectrum_arr[0]) == 3:
            for i in self.spectrum_arr:
                for e in i[1]:
                    x.append(i[0])
                    y.append(e)
            self.x = x
            self.y = y
        elif len(self.spectrum_arr[0]) == 4:
            for i in self.spectrum_arr:
                for e,l in zip(i[1],i[3]):
                    x.append(i[0])
                    y.append(e)
                    label.append(l)
            self.x = x
            self.y = y
            self.label = label
    
    
    '''
    #ignore indivisual labels for now
    def label_t(self,t=0):
        label_t_arr = []
        label_t_y = []
        for x_temp,y_temp,l in zip(self.x,self.y,self.label):
            if x_temp == t:
                label_t_arr.append(l)
                label_t_y.append(y_temp)
        return label_t_arr , label_t_y
    '''
    def view_fig(self, y_min = -1, y_max = -1, use_label=False, title = -1):
        self.fig = plt.figure(figsize=(self.size,self.size))
        if not((y_min ==-1 or y_max ==-1) or (y_min > y_max) ):
            plt.ylim(y_min, y_max)
        plt.scatter(self.x,self.y,s = 0.5, color = self.c)
        '''
        if use_label:
            label_t_arr , label_t_y = self.label_t()
            for l,y in zip(label_t_arr , label_t_y):
                plt.text(0.0,y,str(l))
        '''
        if title == -1:
            title = self.title
        plt.title(title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        

