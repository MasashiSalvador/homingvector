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
    MAX = 200
##    length = 2*MAX*10
##    a1 = 30
##    theta1 = 0
##    a2 = 25
##    theta2 = 40 * pi/180
##    a3 = 51
##    theta3 = 0.5*(pi+theta2)
##    A1s = [a1,-MAX+a1]
##    A2s = [a2,-MAX+a2]
##    A3s = [a3,-a3]
##    thetas = [theta2,theta2,theta3,theta3+pi]
##    As = A2s
##    pl.clf()
##    for A in A1s:
##        xs = [A for i  in range(0,length*2)]
##        ys = np.arange(-MAX*2,MAX*2,0.1)
##        pl.plot(xs,ys,'.b')
##        pl.hold(True)
##
##    xs = np.arange(-MAX*2,MAX*2,0.1)
##    for A in A2s:
##        ys = [yOnLine(x,A,thetas[As.index(A)]) for x in xs]
##        pl.plot(xs,ys,'.b')
##    
##    ps = generateFourPoints(a1,a2,theta2,MAX)
##    cor3rdaxis = []
##    for p in ps:
##        cor3rdaxis.append(projectPointToAxis(p,theta3))
    #pl.show()
    ##################################################
    THETA_MIN = 20
    THETA_MAX = 80
    theta2s = [ang * pi/180 for ang in range(THETA_MIN,THETA_MAX)]
    A1s = np.arange(1,180,1)
    A2s = np.arange(1,180,1)
    ratios = []
    for theta2 in theta2s:
        theta3 = 0.5*(pi+theta2)
        count = 0
        for a1 in A1s:
            for a2 in A2s:
                ps = generateFourPoints(a1,a2,theta2,MAX)
                cor3rdaxis = []
                for p in ps:
                    cor3rdaxis.append(projectPointToAxis(p,theta3))
                cs = [c % MAX for c in cor3rdaxis]
                if(howManymultipleInList(cs) == 0):
                    count = count + 1
        ratios.append((1.0*count)/(1.0*len(A1s)*len(A2s)))
    pl.plot(range(THETA_MIN,THETA_MAX),ratios,'.b')
    pl.title("false ratio to each orientation")
    pl.xlabel("orientation(deg)")
    pl.ylabel("false ratio")
    pl.plot()
    pl.show()
            
            
   
   
   
   
   

