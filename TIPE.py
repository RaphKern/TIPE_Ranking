# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 15:09:12 2016

@author: Raphaël
"""

N=10

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def match_alea(n):
    i,j=rd.randint(0,n),rd.randint(0,n)
    while i==j:
        i=rd.randint(0,n)
    return i,j

def majniveau(En,match,matprob,k):
    joueur1=match[0]
    joueur2=match[1]
    p=matprob[joueur1][joueur2]
    if p<rd.random():
        gagnant=(1,0)
    else:
        gagnant=(0,1)
    d=float((En[joueur1]-En[joueur2]))/400
    pd=1./(1+(10**(-d)))
    En[joueur1]+=k*(gagnant[0]-pd)
    En[joueur2]+=k*(gagnant[1]-(1-pd))
    return En

def matproba(nivreel):
    n=len(nivreel)    
    M=np.zeros((n,n))
    for i in range(n):
        M[i][i]=0.5
    for i in range(n-1):
        for j in range(i+1,n):
            D=float((nivreel[j]-nivreel[i]))/400
            p=1./(1+(10**(-D)))
            M[i][j]=p
            M[j][i]=1-p
    return M

def simulation(nivreel,N,k):
    n=len(nivreel)
    m=np.mean(nivreel)
    E=[m for i in range(n)]
    matprob=matproba(nivreel)
    for i in range(N):
        match=match_alea(n)
        E=majniveau(E,match,matprob,k)    
    return E
    
def ecart_type(l1,l2):
    n=len(l1)
    res=0
    for i in range(n):
        res+=(l1[i]-l2[i])**2
    return np.sqrt(res/n)

def test(nivreel,Lk,j):
    X=[(float(i)/10.) for i in range(20,10*j)]
    for i in range(len(Lk)):    
        Y=[ecart_type(nivreel,(simulation(nivreel,int(10**k),Lk[i]))) for k in X]
        plt.plot(X,Y)
    plt.xlabel('Nombre de matchs')
    plt.ylabel('Ecart type')
    plt.show()
  
def test2(nivreel,k,j):
    X=[10**(float(i)/2.) for i in range(4,2*j+1)]
    Y=[ecart_type(nivreel,(simulation(nivreel,int(j),k))) for j in X]
    print(X)
    print()
    print(Y)
    




