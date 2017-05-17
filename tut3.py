#Sanele Lionel Khanyile
#210522213

﻿import numpy as np

class N_Body:

	def __init__(self,mass = 0,x_pos = 0, y_pos = 0):
		self.mass = mass
		self.x_pos = x_pos
		self.y_pos = y_pos

	var = {'no_of_particles': 100, 'G': 6.67*10**-11}

	def pot_E(self,m,x,y):
		r = np.zeros(len(x))
		E_top = np.zeros(len(r))
		f = np.ones(len(r))
		E = np.zeros(len(r))
		k = ()
		for k in range(0,len(m)):
			for i in range(0,len(x)):
				r[i] = np.sqrt((x[k]-x[i])**2 + (y[k]-y[i])**2)
				if r[i] != 0:
					f[i] = r[i]
			for i in range(0,len(x)):
				E_top[i] = N_Body.var['G']*m[k]*m[i]
			for i in range(0,len(x)):
				if r[i] != 0:
					E[i] = E_top[i]/r[i]
			print (E.sum())
            

if __name__=="__main__":

	s = N_Body.var['no_of_particles']
	x = np.random.randint(1,10,size=s)
	y = np.random.randint(1,10,size=s)
	m = np.random.randint(1,4,size=s)
	test = N_Body(m,x,y)

	energy = test.pot_E(test.mass,test.x_pos,test.y_pos)

	def SoftenedForce_E(self,m,x,y):
		r = np.zeros(len(x))
		E_top = np.zeros(len(r))
		f = np.ones(len(r))
		E = np.zeros(len(r))
		k = ()
		for k in range(0,len(m)):
			for i in range(0,len(x)):

				r[i] = (((x[k]-x[i])**2 + (y[k]-y[i])**2 +  0.9)**1.5/np.sqrt(x[k]-x[i])**2 + (y[k]-y[i])**2) 
				if r[i] != 0:
					f[i] = r[i]
			for i in range(0,len(x)):
				E_top[i] = N_Body.var['G']*m[k]*m[i]
			for i in range(0,len(x)):
				if r[i] != 0:
					E[i] = E_top[i]/r[i]
			print (E.sum())
            

if __name__=="__main__":

	s = N_Body.var['no_of_particles']
	x = np.random.randint(1,10,size=s)
	y = np.random.randint(1,10,size=s)
	m = np.random.randint(1,4,size=s)
	test = N_Body(m,x,y)
	
n= 101

tfin= 2*np.pi

dt= tfin/(n-1)

s= np.arange(n)

y= np.sinc(dt*s)

fy= np.fft.fft(y)

wps= np.linspace(0,2*np.pi,n+1)[:-1]

basis= 1.0/n*np.exp(1.0j * wps * s[:,np.newaxis])

recon_y= np.dot(basis,fy)

yerr= np.max(np.abs(y-recon_y))
print('yerr:',yerr)

lin_fy= np.linalg.solve(basis,y)

fyerr= np.max(np.abs(fy-lin_fy))
print('fyerr',fyerr)

def simulate_lorentzian(x,a=2.5,b=1,c=5):
    dat=a/(b+(x-c)**2)
    dat+=np.random.randn(x.size)
    return dat

def get_trial_offset(sigs):
    return sigs*np.random.randn(sigs.size)

class Lorentzian:
    def __init__(self,x,a=2.5,b=1,c=5,offset=0):
        self.x=x
        self.y=simulate_lorentzian(x,a,b,c)+offset
        self.err=np.ones(x.size)
        self.a=a
        self.b=b
        self.c=c
        self.offset=offset

    def get_chisq(self,vec):
        a=vec[0]
        b=vec[1]
        c=vec[2]
        off=vec[3]

        pred=off+a/(b+(self.x-c)**2)
        chisq=np.sum(  (self.y-pred)**2/self.err**2)
        return chisq

def run_mcmc(data,start_pos,nstep,scale=None):
    nparam=start_pos.size
    params=np.zeros([nstep,nparam+1])
    params[0,0:-1]=start_pos
    cur_chisq=data.get_chisq(start_pos)
    cur_pos=start_pos.copy()
    if scale==None:
        scale=np.ones(nparam)
    for i in range(1,nstep):
        new_pos=cur_pos+get_trial_offset(scale)
        new_chisq=data.get_chisq(new_pos)
        if new_chisq<cur_chisq:
            accept=True
        else:
            delt=new_chisq-cur_chisq
            prob=np.exp(-0.5*delt)
            if np.random.rand()<prob:
                accept=True
            else:
                accept=False
        if accept: 
            cur_pos=new_pos
            cur_chisq=new_chisq
        params[i,0:-1]=cur_pos
        params[i,-1]=cur_chisq
    return params


if __name__=='__main__':
    
    x=np.arange(-5,5,0.01)
    dat=Lorentzian(x,b=2.5)

    guess=np.array([0.3,1.2,0.3,-0.2])
    scale=np.array([0.1,0.1,0.1,0.1])
    nstep=100000
    chain=run_mcmc(dat,guess,nstep,scale)
    nn=np.round(0.2*nstep)
    chain=chain[nn:,:]
    
    param_true=np.array([dat.a,dat.b,dat.c,dat.offset])
    for i in range(0,param_true.size):
        val=np.mean(chain[:,i])
        scat=np.std(chain[:,i])
print ([param_true[i],val,scat])
