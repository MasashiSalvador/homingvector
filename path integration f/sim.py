import numpy as np
import math

#general settings
totalstep = 20
Ninput = 51
Nmem = 51
mode = "debug" #debug or release
b0 = 0.5
v = 1.0
sigma = np.pi/8
mu = np.pi/2
mu_v = np.pi/32
#funcitons
def retWmem(i,j):
   return(2*np.cos(2*np.pi*(i*1.0/Nmem-j*1.0/Nmem)))

def k0velo(v):
   if mode == "debug":
      return 1.0
   else:
      return(0)

def retWinput(i,j):
   return(-k0velo(v)*np.cos(2*np.pi*(i*1.0/Ninput - j*1.0/Nmem)))

def fsig(x):
   if(x < 0):
      return(0)
   elif(x >= 0 and x <= 2*b0):
      return(x)
   else:
      return 2*b0
def tuningcurve(x):
   return(1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2)))

#main
class Neuron:
   def __init__(self,index):
      self.fr = 0
      self.index = 0
      self.coordinate = index*2*np.pi/Ninput

class NeuronCluster:
   def __init__(self,num):
      self.N = num
      self.type = type
      #setting of compass direction
      self.sigma = np.pi/8
      self.mu = np.pi/2
      self.mu_v = np.pi/32
      self.neurons = [Neuron(i) for i in range(0,self.N)]
      self.neuronsPrevious = [Neuron(i) for i in range(0,self.N)]
      self.neuronsInputlayer = [Neuron(i) for i in range(0,self.N)]
      self.Wmem = np.zeros((self.N,self.N))
      self.Winput = np.zeros((self.N,self.N))
      for i in range(0,self.N):
         for j in range(0,self.N):
            self.Wmem[i,j] = retWmem(i,j)
            self.Winput[i,j] = retWinput(i,j)

   def renewMu(self):
      self.mu += self.mu_v
      self.mu = self.mu % (np.pi*2)
   def tuningcurve(self,x):
      return(1/(self.sigma*np.sqrt(2*np.pi))*np.exp(-(x-self.mu)**2/(2*self.sigma**2)))
   def setInputlayer(self):
      self.renewMu()
      for i in range(0,self.N):
         self.neuronsInputlayer[i].fr = self.tuningcurve(self.neuronsInputlayer[i].coordinate)
      
   def savePrevious(self):
      self.neuronsPrevious = self.neurons
   def printFiringRate(self):
      strs = [str(self.neuronsPrevious[i].fr) + " " for i in range(0,self.N)]
      return(strs)
   def printFiringRate2(self):
      for i in range(0,self.N):
         print str(self.neuronsPrevious[i].fr) + " ",
      print
   def printCompass(self):
      for i in range(0,self.N):
         print str(self.neuronsInputlayer[i].fr) + " ",
      print

   def proceed(self):
      self.savePrevious()
      self.setInputlayer()
      tmp = 0
      for i in range(0,self.N):
         for j in range(0,self.N):
            tmp += self.Wmem[i,j]*self.neuronsPrevious[j].fr + self.Winput[i,j]*self.neuronsInputlayer[i].fr+b0
            self.neurons[i].fr = fsig(tmp)
            tmp =0
            
if __name__ == "__main__":
    print "start"
    itr = 0
    neuCs = NeuronCluster(Nmem)
    while (itr < totalstep):
       neuCs.proceed()
       neuCs.printFiringRate2()
       itr += 1
    print "end"

