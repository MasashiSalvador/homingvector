#coding: utf-8
import numpy as np
import pylab as pl
from math import *

def line3Test(x,L,theta):
    tiltTheta = theta - pi/2
    ret = tan(tiltTheta)*(x-L*cos(theta))+L*sin(theta)
    return ret
def isOnline2dVerticalToTheta(x,y,L,theta):
    eps = 0.00001
    tiltTheta = theta - pi/2
    val = tan(tiltTheta)*(x-L*cos(theta))+L*sin(theta) - y
    if(abs(val) < eps):
        ret = True
    else:
        ret = False
    return ret
def countEps(ind,L,eps):
    count = 0
    ref = L[ind]
    for Lin in L:
        if(abs(Lin - ref) < eps):
            count = count + 1
    count = count - 1
    return count

def howManymultipleInList(L):
    eps = 0.00001
    multiples = []
    for i in range(0,len(L)):
        multiples.append(countEps(i,L,eps))
    numMultipled = max(multiples)
    return numMultipled
    
         

def generateFourPoints(a1,a2,ang,MAX):
    ret = []
    A1s = [a1,-MAX+a1]
    A2s = [a2, -MAX+a2]
    for A1 in A1s:
        for A2 in A2s:
            X = A1
            Y = yOnLine(X,A2,ang)
            #Y = tan(ang+pi/2)*(A1-A2*cos(ang)) + A2*sin(ang)
            ret.append(np.array((X,Y)))
    return ret


def generateRotationMat(theta3):
    ret = np.matrix(((cos(-theta3),-sin(-theta3)),(sin(-theta3),cos(-theta3))))
    return ret

def projectPointToAxis(p,theta3):
    ret = 0
    X = p[0]
    Y = p[1]
    r = (X**2 + Y**2)**0.5
    angle = atan(Y/X)
    diff = abs(angle - theta3)
    if(diff <= pi/2):
        ret = r*cos(diff)
    else:
        ret = -r*cos(pi-diff)
    return ret

def yOnLine(x,R,theta):
    ret = tan(theta+pi/2)*(x-R*cos(theta))+R*sin(theta)
    return ret

if __name__ == "__main__":
    #MAX = 200
    #length = 2*MAX*10
    #a1 = 30
    #theta1 = 0
    #a2 = 25
    #theta2 = 60 * pi/180
    #a3 = 51
    #theta3 = 0.5*(pi+theta2)
    #A1s = [a1,-MAX+a1]
    #A2s = [a2,-MAX+a2]
    #A3s = [a3,-a3]
    #thetas = [theta2,theta2,theta3,theta3+pi]
    #As = A2s
    #pl.clf()
    #for A in A1s:
    #    xs = [A for i  in range(0,length*2)]
    #    ys = np.arange(-MAX*2,MAX*2,0.1)
    #    pl.plot(xs,ys,'.b')
    #    pl.hold(True)

    #xs = np.arange(-MAX*2,MAX*2,0.1)
    #for A in A2s:
    #    ys = [yOnLine(x,A,thetas[As.index(A)]) for x in xs]
    #    pl.plot(xs,ys,'.b')
    #ps = generateFourPoints(a1,a2,theta2,MAX)
    #for p in ps:
    #    pl.plot(p[0],p[1],'*k')
    #    pl.hold(True)
    #pl.show()
    ##################################################
    MaxA1 = 40
    MaxA2 = 40
    theta1= 0
    theta2s = np.arange(50,70,1)
    theta2s = theta2s * pi/180
    a1s = np.arange(20,MaxA1,1)
    a2s = np.arange(20,MaxA2,1)
    numss = []
    counter = 0
    for t in theta2s:
        t3 = pi/2 + t/2
        nums = []
        for a1 in a1s:
            for a2 in a2s:
                sols = []
                vecs = generateFourPoints(a1,a2,t,MaxA2)
                for vec in vecs:
                    sols.append(projectPointToAxis(vec,t3))
                hs = sols
                hss = [h% MaxA1 for h in hs]
                nums.append(howManymultipleInList(hs))
        numss.append(nums)
    yaxis = np.arange(len(a1s)**2+1)
    thetaAxis = np.arange(len(theta2s)+1)
    Xs,Ys = pl.meshgrid(yaxis,thetaAxis)
    pl.pcolor(Xs,Ys,numss)
    pl.colorbar()

    pl.show()
        
                
            
   
   
   
   
   

