#!/usr/bin/env python
# coding: utf-8

# # Density Functional Theory python examples
# 

# I'll add more info soon, for now here it is, shortly:
# 
# This follows the assignments for lectures of Thomas Arias, lectures available on youtube. You'll find info and an embedded video of the first lecture in the DFT Quantum Dot page on the Computational Physics Blog: https://compphys.go.ro/dft-for-a-quantum-dot/
# 
# The following links are available on the page mentioned above, but I'll repeat them here:
# 
# Assignments are described here: https://drive.google.com/drive/folders/0B8lnMKudhQYMd2NWTkV1akpYSTQ?hl=en
# 
# A paper that describes the formalism: https://arxiv.org/abs/cond-mat/9909130
# 
# For now this is work in progress, I'll add more to it (including comments), hopefully it will cover at least as much as the https://github.com/aromanro/DFTQuantumDot project.

# ## The Poisson Equation
# 
# This corresponds to the first assigment and it covers what the https://github.com/aromanro/Poisson project covers (described here: https://compphys.go.ro/solving-poisson-equation/).

# In[1]:


import math as m
import numpy as np
import scipy as sp
import scipy.linalg as splalg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def CreateIndices(S):
    ms = np.arange(np.prod(S))
    m1 = np.remainder(ms, S[0])
    m2 = np.remainder(np.floor_divide(ms, S[0]), S[1])
    m3 = np.remainder(np.floor_divide(ms, S[0] * S[1]), S[2])
    M = np.asarray([m1, m2, m3]).transpose()    
    
    n1 = np.array([x - (x > S[0]/2) * S[0] for x in m1])
    n2 = np.array([x - (x > S[1]/2) * S[1] for x in m2])
    n3 = np.array([x - (x > S[2]/2) * S[2] for x in m3])
    N = np.asarray([n1, n2, n3]).transpose()
    
    return M, N


# In[3]:


S = np.array([32, 32, 32])

M, N = CreateIndices(S)


# In[4]:


R=np.diag([6, 6, 6])


# In[5]:


r = M @ splalg.inv(np.diag(S)) @ np.transpose(R)
G = 2 * m.pi * N @ splalg.inv(R)
G2 = np.sum(G * G, axis=1)
G2 = np.reshape(G2, (G2.size, 1))


# In[6]:


cellCenter = np.sum(R, axis = 1) / 2


# In[7]:


vecsFromCenter = r - np.ones((np.prod(S), 1)) * cellCenter
dr = np.sqrt(np.sum(vecsFromCenter * vecsFromCenter, 1))


# In[8]:


def Gaussian(r, sigma = 0.5):
    twosigma2 = 2. * sigma * sigma
    return np.exp(-r*r/twosigma2) / np.power(np.sqrt(m.pi * twosigma2), 3)    


# In[9]:


sigma1 = 0.75
sigma2 = 0.5
g1 = Gaussian(dr, sigma1)
g2 = Gaussian(dr, sigma2)
n = g2 - g1
n = np.reshape(n, (n.size, 1))


# In[10]:


print ("Normalization check on g1: ", np.sum(g1) * splalg.det(R) / np.prod(S))


# In[11]:


print ("Normalization check on g2: ", np.sum(g2) * splalg.det(R) / np.prod(S))


# In[12]:


print ("Total charge check: ", np.sum(n) * splalg.det(R) / np.prod(S))


# In[13]:


def fft3(dat, N, s):
    if s == 1:
        result = np.reshape(np.fft.ifftn(np.reshape(dat, (N[0], N[1], N[2]), order='F')) * np.prod(N), dat.shape, order='F')        
    else:
        result = np.reshape(np.fft.fftn(np.reshape(dat, (N[0], N[1], N[2]), order='F')), dat.shape, order='F')
    
    return result


# In[14]:


def cI(inp):
    return fft3(inp, S, 1)


# In[15]:


def cJ(inp):
    return 1. / np.prod(S) * fft3(inp, S, -1)


# In[16]:


def O(inp):
    return splalg.det(R) * inp


# In[17]:


def L(inp):    
    return -splalg.det(R) * G2 * inp


# In[18]:


def Linv(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp
        
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    result = -1. / splalg.det(R) * vals / np.reshape(G2, vals.shape)
    result[0] = 0
    np.seterr(**old_settings)
    return result    


# In[19]:


def Poisson(inp):
    if inp.ndim == 1:
        n = np.reshape(inp, (inp.size, 1))
    else:
        n = inp
        
    return cI(Linv(-4. * m.pi * O(cJ(n))))


# In[20]:


phi = Poisson(n)


# In[21]:


Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Uanal=((1./sigma1+1./sigma2)/2.- np.sqrt(2.) / np.sqrt(sigma1*sigma1 + sigma2*sigma2))/np.sqrt(m.pi)
print('Numeric, analytic Coulomb energy:', Unum[0,0], Uanal)


# In[22]:


def Plot(dat):

    if dat.ndim != 3:
        dat = np.reshape(dat, S, order='F')
    
    fig=plt.figure(figsize=(35, 25))

    x = np.arange(0, S[1])
    y = np.arange(0, S[2])
    xs, ys = np.meshgrid(x, y, indexing='ij')

    toplot1 = dat[int(S[0]/2),:,:]
        
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(xs, ys, toplot1, cmap='viridis', edgecolor='none')

    x = np.arange(0, S[0])
    xs, ys = np.meshgrid(x, y, indexing='ij')

    toplot2 = dat[:,int(S[1]/2),:]
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(xs, ys, toplot2, cmap='viridis', edgecolor='none')

    y = np.arange(0, S[1])
    xs, ys = np.meshgrid(x, y, indexing='ij')
    toplot3 = dat[:,:,int(S[2]/2)]
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(xs, ys, toplot3, cmap='viridis', edgecolor='none')

    plt.tight_layout()
    plt.show()


# In[23]:


Plot(n)


# In[24]:


Plot(np.real(phi))


# In[25]:


X = np.asarray([[0, 0, 0], [1.75, 0, 0]])
Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)
Z = 1


# In[26]:


sigma1 = 0.25
g1 = Z * Gaussian(dr, sigma1)


# In[27]:


n = cI(cJ(g1) * Sf)
n = np.real(n)


# In[28]:


Plot(n)


# In[29]:


Uself = Z*Z/(2.*m.sqrt(m.pi))*(1./sigma1)*np.size(X,0)


# In[30]:


phi = Poisson(n)


# In[31]:


Plot(np.real(phi))


# In[32]:


Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))[0]
print('Ewald energy:', Unum - Uself)
print('Unum:', Unum)
print('Uself:', Uself)


# ## Assignment 2: Implement variational solution to Schrodinger's equation and Kohn-Sham equations using steepest descents

# First, let's repeat initialization similarly as above but with different params:

# In[33]:


S = np.array([20, 25, 30])
M, N = CreateIndices(S)

r = M @ splalg.inv(np.diag(S)) @ np.transpose(R)
G = 2. * m.pi * N @ splalg.inv(R)
G2 = np.sum(G * G, axis=1)
G2 = np.reshape(G2, (G2.size, 1))

vecsFromCenter = r - np.ones((np.prod(S), 1)) * cellCenter
dr2 = np.sum(vecsFromCenter * vecsFromCenter, 1)
dr = np.sqrt(dr2)


# In[34]:


def cI(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp
        
    out = np.zeros(vals.shape, dtype = "complex_")    
    for col in range(np.size(vals, 1)):
        out[:,col] = fft3(vals[:,col], S, 1)
    
    return out


# In[35]:


def cJ(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp
    
    norm = 1. / np.prod(S)
    out = np.zeros(vals.shape, dtype = "complex_")
    
    for col in range(np.size(vals, 1)):
        out[:,col] = norm * fft3(vals[:,col], S, -1)
    
    return out


# In[36]:


def L(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp
    
    return -splalg.det(R) * (G2 @ np.ones((1, np.size(vals, 1)))) * vals


# In[37]:


def cIdag(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp

    out = np.zeros(vals.shape, dtype = "complex_")

    for col in range(np.size(vals, 1)):
        out[:,col] = fft3(vals[:,col], S, -1)
    
    return out


# In[38]:


def cJdag(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp
    
    norm = 1. / np.prod(S)
    out = np.zeros(vals.shape, dtype = "complex_")
    
    for col in range(np.size(vals, 1)):
        out[:,col] = norm * fft3(vals[:,col], S, 1)
    
    return out


# Test on what we computed already

# In[39]:


sigma1 = 0.75
sigma2 = 0.5
g1 = Gaussian(dr, sigma1)
g2 = Gaussian(dr, sigma2)
n = g2 - g1
n = np.reshape(n, (n.size, 1))

print ("Normalization check on g1: ", np.sum(g1) * splalg.det(R) / np.prod(S))
print ("Normalization check on g2: ", np.sum(g2) * splalg.det(R) / np.prod(S))
print ("Total charge check: ", np.sum(n) * splalg.det(R) / np.prod(S))

phi = Poisson(n)

Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Uanal=((1./sigma1+1./sigma2)/2.- np.sqrt(2.) / np.sqrt(sigma1*sigma1 + sigma2*sigma2))/np.sqrt(m.pi)
print('Numeric, analytic Coulomb energy:', Unum[0, 0], Uanal)


# #### Let's solve the Schrodinger equation

# In[40]:


V = 2. * dr2
V = np.reshape(V, (V.size, 1))


# In[41]:


Vdual=cJdag(O(cJ(V)))


# In[42]:


def diagouter(A, B):
    return np.sum(A*B.conjugate(),axis=1)


# In[43]:


def getE(W):
    U = W.transpose().conjugate() @ O(W)
    Uinv = splalg.inv(U)
    IW = cI(W)
    n = diagouter(IW @ Uinv, IW)
    E = np.real(-0.5 * np.sum(diagouter(L(W @ Uinv), W)) + Vdual.transpose().conjugate() @ n)
    return E


# In[44]:


def Diagprod(a, B):
    return (a @ np.ones((1, np.size(B, axis = 1)))) * B


# In[45]:


def H(W):
    return -0.5 * L(W) + cIdag(Diagprod(Vdual, cI(W)))


# In[46]:


def getgrad(W):
    Wadj = W.transpose().conjugate()
    OW = O(W)
    U = Wadj @ OW
    Uinv = splalg.inv(U)
    HW = H(W)  
    return (HW - (OW @ Uinv) @ (Wadj @ HW)) @ Uinv


# In[47]:


def orthogonalize(W):
    U = W.transpose().conjugate() @ O(W)
    return W @ splalg.inv(splalg.sqrtm(U))


# In[48]:


def sd(W, Nit):
    alfa = 0.00003
    
    for i in range(Nit):
        W = W - alfa * getgrad(W)
    
    return W


# In[49]:


def getPsi(W):
    Y = orthogonalize(W)
    mu = Y.transpose().conjugate() @ H(Y)
    
    epsilon, D = splalg.eig(mu)
    epsilon = np.real(epsilon)
    
    idx = epsilon.argsort()[::]   
    epsilon = epsilon[idx]
    D = D[:,idx]
    
    Psi = Y @ D
    
    return Psi, epsilon


# In[50]:


X = np.asarray([[0, 0, 0], [4, 0, 0]])
Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)

Z = 1
sigma1 = 0.25

g1 = Z * Gaussian(dr, sigma1)
g1 = np.reshape(g1, (g1.size, 1))
n = cI(cJ(g1) * Sf)
n = np.real(n)


# In[51]:


Ns = 4
np.random.seed(100)

W = np.random.randn(np.prod(S),Ns) + 1j * np.random.randn(np.prod(S),Ns)
W = orthogonalize(W)


# In[52]:


W = sd(W,600)


# In[53]:


Psi, epsilon = getPsi(W)


# In[54]:


print(epsilon)


# In[55]:


for i in range(4): 
    dat = cI(Psi[:,i])
    dat = np.real(dat.conjugate() * dat)
    print('State no:', i, 'energy value:', epsilon[i])
    Plot(dat)


# #### Now, DFT

# In[56]:


f = 2


# In[57]:


def getgrad(W):
    Wadj = W.transpose().conjugate()
    OW = O(W)
    U = Wadj @ OW
    Uinv = splalg.inv(U)
    HW = H(W)  
    return f * (HW - (OW @ Uinv) @ (Wadj @ HW)) @ Uinv


# In[58]:


def PoissonSolve(inp):
    if inp.ndim == 1:
        n = np.reshape(inp, (inp.size, 1))
    else:
        n = inp
        
    return -4. * m.pi * Linv(O(cJ(n)))


# In[59]:


def excVWN(n):
    X1 = 0.75*(3.0/(2.0*m.pi))**(2.0/3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = m.sqrt(4*c-b*b)
    X0 = x0*x0+b*x0+c

    rs=(4*m.pi/3*n)**(-1./3.)
  
    x = np.sqrt(rs)
    X = x*x+b*x+c

    return -X1/rs + A*(np.log(x * x / X) +2 * b / Q * np.arctan(Q/(2 * x+b)) - (b*x0)/X0*(np.log((x-x0)*(x-x0)/X)+2*(2*x0+b)/Q*np.arctan(Q/(2*x+b))))


# In[60]:


def getE(W):
    U = W.transpose().conjugate() @ O(W)
    Uinv = splalg.inv(U)
    IW = cI(W)
    
    n = f * diagouter(IW @ Uinv, IW)
    ndag = n.transpose().conjugate()
    
    Phi = PoissonSolve(n)
    exc = excVWN(n)
    
    PhiExc = 0.5 * Phi + cJ(exc)
    
    E = np.real(-f * 0.5 * np.sum(diagouter(L(W @ Uinv), W)) + Vdual.transpose().conjugate() @ n + ndag @ cJdag(O(PhiExc)))
    return E[0]


# In[61]:


def excpVWN(n):
    X1 = 0.75*(3.0/(2.0*m.pi))**(2.0/3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = m.sqrt(4.*c-b*b)
    X0 = x0*x0+b*x0+c

    rs=(4.*m.pi/3. * n)**(-1./3.)

    x=np.sqrt(rs)
    X=x*x+b*x+c

    dx=0.5/x

    return (-rs/(3.*n))* dx * (2.*X1/(rs*x)+A*(2./x-(2.*x+b)/X-4.*b/(Q*Q+(2.*x+b)*(2.*x+b))-(b*x0)/X0*(2./(x-x0)-(2.*x+b)/X-4.*(2.*x0+b)/(Q*Q+(2*x+b)*(2*x+b)))))


# In[62]:


def H(W):
    U = W.transpose().conjugate() @ O(W)
    Uinv = splalg.inv(U)
    IW = cI(W)

    n = f * diagouter(IW @ Uinv, IW)
    
    Phi = PoissonSolve(n)
  
    exc = excVWN(n)
    excp = excpVWN(n)
    
    PhiExc = Phi + cJ(exc)

    Veff = Vdual + cJdag(O(PhiExc)) + np.reshape(excp, (excp.size, 1)) * cJdag(O(cJ(n)))
    
    return -0.5 * L(W) + cIdag(Diagprod(Veff, IW))


# In[63]:


np.random.seed(100)

W = np.random.randn(np.prod(S),Ns) + 1j * np.random.randn(np.prod(S),Ns)
W = orthogonalize(W)


# In[64]:


W = sd(W,600)


# In[65]:


Psi, epsilon = getPsi(W)


# In[66]:


print(epsilon)


# In[67]:


print('Total energy:', getE(W))


# In[68]:


for i in range(4): 
    dat = cI(Psi[:,i])
    dat = np.real(dat.conjugate() * dat)
    print('State no:', i, 'energy value:', epsilon[i])
    Plot(dat)


# ## Assignment 3

# ### First part: Advanced techniques for numerical minimization

# In[69]:


R=np.diag([16, 16, 16])

S = np.array([64, 64, 64])
M, N = CreateIndices(S)

r = M @ splalg.inv(np.diag(S)) @ np.transpose(R)
G = 2. * m.pi * N @ splalg.inv(R)
G2 = np.sum(G * G, axis=1)
G2 = np.reshape(G2, (G2.size, 1))

cellCenter = np.sum(R, axis = 1) / 2

vecsFromCenter = r - np.ones((np.prod(S), 1)) * cellCenter
dr2 = np.sum(vecsFromCenter * vecsFromCenter, 1)
dr = np.sqrt(dr2)

Z = 1
f = 2

V = 2. * dr2
V = np.reshape(V, (V.size, 1))

Vdual=cJdag(O(cJ(V)))


# In[70]:


def sd(Win, Nit, fillE = True, alfa = 0.00003):
    W = Win
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        W = W - alfa * getgrad(W)
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[71]:


Ns = 10

np.random.seed(100)

W = np.random.randn(np.prod(S),Ns) + 1j * np.random.randn(np.prod(S),Ns)
W = orthogonalize(W)

W, Elist = sd(W, 30)


# In[72]:


W=orthogonalize(W)


# In[73]:


def Dot(a, b):
    return np.real(np.trace(a.transpose().conjugate() @ b))


# In[74]:


def lm(Win, Nit, fillE = True, alphat = 0.00003):
    W = Win
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        g = getgrad(W)
        if i > 0 and fillE:
            anglecos = Dot(g, d) / m.sqrt(Dot(g,g) * Dot(d,d))
            print("Anglecos:", anglecos)
        
        d = -g
    
        gt = getgrad(W + alphat * d)
    
        dotdif = Dot(g-gt, d)
        if np.abs(dotdif) < 1E-20:
            dotdif = 1E-20 * np.sign(dotdif)
    
        alpha = alphat * Dot(g,d) / dotdif
    
        W = W + alpha * d
    
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[75]:


Wlm, Elm = lm(W, 120)


# In[76]:


def K(inp):
    return inp / (1. + G2)


# The preconditioned line minimization is the same as the line minimization above, but with the direction changed:

# In[77]:


def pclm(Win, Nit, fillE = True, alphat = 0.00003):
    W = Win
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
        
    for i in range(Nit):
        g = getgrad(W)
        if i > 0 and fillE:
            anglecos = Dot(g, d) / m.sqrt(Dot(g,g) * Dot(d,d))
            print("Anglecos:", anglecos)
        
        d = -K(g)
    
        gt = getgrad(W + alphat * d)
    
        dotdif = Dot(g-gt, d)
        if np.abs(dotdif) < 1E-20:
            dotdif = 1E-20 * np.sign(dotdif)
    
        alpha = alphat * Dot(g,d) / dotdif
    
        W = W + alpha * d
    
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[78]:


Wpclm, Epclm = pclm(W, 120)


# In[79]:


def pccg(Win, Nit, cgform, fillE = True, alphat = 0.00003):
    W = Win
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
        
    for i in range(Nit):
        g = getgrad(W)
        theK = K(g)
        
        if i > 0:
            if fillE:
                anglecos = Dot(g, d) / m.sqrt(Dot(g,g) * Dot(d,d))
                cgtest = Dot(g, oldK) / m.sqrt(Dot(g, theK) * Dot(oldg, oldK)) 
                print("Anglecos:", anglecos, "cgtest:", cgtest)
        
            if cgform == 1: # Fletcher-Reeves
                beta = Dot(g, theK) / Dot(oldg, oldK)
            elif cgform == 2: # Polak-Ribiere
                beta = Dot(g-oldg, theK)/Dot(oldg, oldK)
            else: # Hestenes-Stiefel
                difg = g-oldg
                beta = Dot(difg, theK)/Dot(difg, d)        
        else:
            d = -theK
            beta = 0
            
        oldg = g
        oldK = theK
    
        d = -theK + beta * d
        
        gt = getgrad(W + alphat * d)
    
        dotdif = Dot(g-gt,d)
        if np.abs(dotdif) < 1E-20:
            dotdif = 1E-20 * np.sign(dotdif)
    
        alpha = alphat * Dot(g,d) / dotdif
    
        W = W + alpha * d
    
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[80]:


Wcg1, Ecg1 = pccg(W,120,1)


# In[81]:


Wcg2, Ecg2 = pccg(W,120,2)


# In[82]:


Wcg3, Ecg3 = pccg(W,120,3)


# In[83]:


plt.semilogy(range(Elm.size), Elm-43.33711477820)
plt.semilogy(range(Epclm.size), Epclm-43.33711477820)
plt.semilogy(range(Ecg1.size), Ecg1-43.33711477820)
plt.semilogy(range(Ecg2.size), Ecg3-43.33711477820)
plt.semilogy(range(Ecg3.size), Ecg3-43.33711477820)

plt.legend(['lm','pclm', 'pccg-FR', 'pccg-PR', 'pccg-HS'])

plt.show()


# #### The following gradient-descent derived methods are not in the lecture or assignments, 
# #### but they are important and can be useful not only in physics but also in other domains (like machine learning), so I added them here.
# #### They can be safely skipped!
# ##### They might be better suited in stochastic context (for example using batches), but there is no reason why they couldn't be tried here, too.
# ##### I intend to use them in a machine learning project and wanted to see them in action before implementing them in C++ in that project.
# ##### Here they don't perform as well as the methods above, but they seem to work.
# ##### They would benefit from some parameters tuning but since they are not intended for this, I will do that in the stochastic gradient descent context.

# In[84]:


def momentum(Win, Nit, fillE = True, alfa = 0.001, beta = 0.5):
    
    W = Win
    m = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        m = beta * m - alfa * getgrad(W)
        W = W + m
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[85]:


Wm, Em = momentum(W,120)


# In[86]:


def NesterovAccelerated(Win, Nit, fillE = True, alfa = 0.001, beta = 0.5):
    W = Win
    
    m = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        m = beta * m - alfa * getgrad(W + beta * m)
        W = W + m
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[87]:


Wn, En = NesterovAccelerated(W,120)


# In[88]:


def AdaGrad(Win, Nit, fillE = True, alfa = 0.001):
    W = Win
    
    s = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        g = getgrad(W)
        s = s + g * g
        W = W - alfa * g / np.sqrt(s + 0.0000001) 
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[89]:


Wa, Ea = AdaGrad(W,120)


# In[90]:


def RMSProp(Win, Nit, fillE = True, alfa = 0.001, beta = 0.5):
    W = Win
    
    s = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        g = getgrad(W)
        s = beta * s + (1. - beta) * g * g
        W = W - alfa * g / np.sqrt(s + 0.0000001) 
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[91]:


Wr, Er = RMSProp(W,120)


# In[92]:


def RMSPropNesterov(Win, Nit, fillE = True, alfa = 0.001, beta1 = 0.5, beta2 = 0.5):
    W = Win
    
    s = np.zeros(W.shape)
    m = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        g = getgrad(W + alfa * m)
        s = beta2 * s + (1. - beta2) * g * g
        m = beta1 * m - alfa * g / np.sqrt(s + 0.0000001) 
        W = W + m 
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[93]:


Wrn, Ern = RMSPropNesterov(W,120)


# In[94]:


def Adam(Win, Nit, fillE = True, alfa = 0.001, beta1 = 0.5, beta2 = 0.5):
    W = Win
    
    s = np.zeros(W.shape)
    m = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        g = getgrad(W)
        m = beta1 * m - (1. - beta1) * g
        s = beta2 * s + (1. - beta2) * g * g
        p = i + 1
        m = m / (1. - np.power(beta1, p))
        s = s / (1. - np.power(beta2, p))
        W = W + alfa * m / np.sqrt(s + 0.0000001) 
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[95]:


Wadam, Eadam = Adam(W, 120)


# In[96]:


def Nadam(Win, Nit, fillE = True, alfa = 0.001, beta1 = 0.5, beta2 = 0.5):
    W = Win
    
    s = np.zeros(W.shape)
    m = np.zeros(W.shape)
    
    if fillE:
        Elist = np.zeros(Nit)
    else:
        Elist = None
    
    for i in range(Nit):
        g = getgrad(W + alfa * m)
        m = beta1 * m - (1. - beta1) * g
        s = beta2 * s + (1. - beta2) * g * g
        p = i + 1
        m = m / (1. - np.power(beta1, p))
        s = s / (1. - np.power(beta2, p))
        W = W + alfa * m / np.sqrt(s + 0.0000001) 
        if fillE:
            E = getE(W)
            Elist[i] = E
            print("Niter:", i, "E:", E)
    
    return W, Elist


# In[97]:


Wnadam,Enadam = Nadam(W, 120)


# In[98]:


plt.semilogy(range(Em.size), Em-43.33711477820)
plt.semilogy(range(En.size), En-43.33711477820)
plt.semilogy(range(Ea.size), Ea-43.33711477820)
plt.semilogy(range(Er.size), Er-43.33711477820)
plt.semilogy(range(Ern.size), Ern-43.33711477820)
plt.semilogy(range(Eadam.size), Eadam-43.33711477820)
plt.semilogy(range(Enadam.size), Enadam-43.33711477820)

plt.legend(['Momentum','Nesterov','AdaGrad','RMSProp','RMSPropNesterov','Adam','Nadam'])

plt.show()


# #### The 'quantum dot':

# In[99]:


W=Wcg1
W=orthogonalize(W)
W, E = pccg(W,50,3, False)

Psi, epsilon = getPsi(W)

for i in range(Ns): 
    dat = cI(Psi[:,i])
    dat = np.real(dat.conjugate() * dat)
    print('State no:', i, 'energy value:', epsilon[i])
    Plot(dat)


# #### Atoms and molecules
# 
# Let's try for Hydrogen first:

# In[100]:


R=np.diag([16, 16, 16])

S = np.array([64, 64, 64])
M, N = CreateIndices(S)

r = M @ splalg.inv(np.diag(S)) @ np.transpose(R)
G = 2. * m.pi * N @ splalg.inv(R)
G2 = np.sum(G * G, axis=1)
G2 = np.reshape(G2, (G2.size, 1))

cellCenter = np.sum(R, axis = 1) / 2

vecsFromCenter = r - np.ones((np.prod(S), 1)) * cellCenter
dr2 = np.sum(vecsFromCenter * vecsFromCenter, 1)
dr = np.sqrt(dr2)

Z = 1
f = 1

X = np.asarray([[0, 0, 0]])

Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)
Sf = np.reshape(Sf, (Sf.size, 1))

sigma1 = 0.25
g1 = Z * Gaussian(dr, sigma1)

n = cI(cJ(g1) * Sf)
n = np.real(n)

phi = Poisson(n)

Uself = Z*Z/(2.*m.sqrt(m.pi))*(1./sigma1)*np.size(X,0)

Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Ewald = (Unum - Uself)[0,0]
print('Ewald energy:', Ewald)
print('Unum:', Unum[0,0])
print('Uself:', Uself)


# In[101]:


old_settings = np.seterr(divide='ignore', invalid='ignore')
Vps=-4.*m.pi*Z/G2
Vps[0]=0.
np.seterr(**old_settings)

Vps = np.reshape(Vps, (Vps.size, 1))
Vdual = cJdag(Vps * Sf)

Ns = 1

np.random.seed(100)

W = np.random.randn(np.prod(S),Ns) + 1j * np.random.randn(np.prod(S),Ns)
W = orthogonalize(W)

W, Elist = sd(W, 30, False)
W = orthogonalize(W)

W, Elist = pccg(W, 30, 1, False)

Psi, epsilon = getPsi(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])
    
print('\nTotal energy:', getE(W) + Ewald, "NIST value: -0.445671")


# Now the Hydrogen molecule:

# In[102]:


f = 2
X = np.asarray([[0, 0, 0], [1.5, 0, 0]])

Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)
Sf = np.reshape(Sf, (Sf.size, 1))

sigma1 = 0.25
g1 = Z * Gaussian(dr, sigma1)

n = cI(cJ(g1) * Sf)
n = np.real(n)

phi = Poisson(n)

Uself = Z*Z/(2.*m.sqrt(m.pi))*(1./sigma1)*np.size(X,0)

Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Ewald = (Unum - Uself)[0,0]
print('Ewald energy:', Ewald)
print('Unum:', Unum[0,0])
print('Uself:', Uself)


# In[103]:


old_settings = np.seterr(divide='ignore', invalid='ignore')
Vps=-4.*m.pi*Z/G2
Vps[0]=0.
np.seterr(**old_settings)

Vps = np.reshape(Vps, (Vps.size, 1))
Vdual = cJdag(Vps * Sf)

Ns = 1

np.random.seed(100)

W = np.random.randn(np.prod(S),Ns) + 1j * np.random.randn(np.prod(S),Ns)
W = orthogonalize(W)

W, Elist = sd(W, 30, False)
W = orthogonalize(W)

W, Elist = pccg(W, 30, 1, False)

Psi, epsilon = getPsi(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])
    
print('\nTotal energy:', getE(W) + Ewald, "Expected: -1.136")


# ## Assignment 4

# #### Minimal, isotropic spectral representation 

# In[104]:


R=np.diag([16, 16, 16])

S = np.array([64, 64, 64])
M, N = CreateIndices(S)

r = M @ splalg.inv(np.diag(S)) @ np.transpose(R)
G = 2. * m.pi * N @ splalg.inv(R)
G2 = np.sum(G * G, axis=1)
G2 = np.reshape(G2, (G2.size, 1))

cellCenter = np.sum(R, axis = 1) / 2

vecsFromCenter = r - np.ones((np.prod(S), 1)) * cellCenter
dr2 = np.sum(vecsFromCenter * vecsFromCenter, 1)
dr = np.sqrt(dr2)

Z = 1
f = 1

X = np.asarray([[0, 0, 0]])

Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)
Sf = np.reshape(Sf, (Sf.size, 1))

sigma1 = 0.25
g1 = Z * Gaussian(dr, sigma1)

n = cI(cJ(g1) * Sf)
n = np.real(n)

phi = Poisson(n)

Uself = Z*Z/(2.*m.sqrt(m.pi))*(1./sigma1)*np.size(X,0)

Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Ewald = (Unum - Uself)[0,0]
print('Ewald energy:', Ewald)
print('Unum:', Unum[0,0])
print('Uself:', Uself)


# In[105]:


eS = np.reshape(S / 2. + 0.5, (S.size, 1))
edges = np.nonzero(np.any(np.abs(M - np.ones((np.size(M, axis = 0), 1)) @ eS.transpose()) < 1, axis = 1))
G2mx = np.min(G2[edges])
active = np.nonzero(G2 < G2mx / 4.)
G2c = np.reshape(G2[active], (active[0].size, 1))
print("Compression:", G2.size / G2c.size, "Theoretical:", 1./(4.*m.pi*(1./4.)**3./3.))


# In[106]:


def L(inp):
    if inp.ndim == 1:
        out = np.reshape(inp, (inp.size, 1))
    else:
        out = inp
    
    if np.size(out, 0) == G2c.size:
        return -splalg.det(R) * (G2c @ np.ones((1, np.size(out, 1)))) * out        
    
    return -splalg.det(R) * (G2 @ np.ones((1, np.size(out, 1)))) * out


# In[107]:


def K(inp):
    if np.size(inp, 0) == G2c.size:
        return inp / (1. + G2c)
    
    return inp / (1. + G2)


# In[108]:


def cI(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
    else:
        vals = inp
    
    pS = np.prod(S)
    out = np.zeros((pS, np.size(vals, axis = 1)), dtype = "complex_")
    
    if np.size(vals, 0) == pS:
        for col in range(np.size(vals, 1)):
            out[:,col] = fft3(out[:,col], S, 1)
    else:
        for col in range(np.size(vals, 1)):
            full = np.zeros((pS, 1), dtype = "complex_")
            full[active] = vals[:,col]
            out[:,col] = fft3(full[:,col], S, 1)
    
    return out


# In[109]:


def cIdag(inp):
    if inp.ndim == 1:
        vals = np.reshape(inp, (inp.size, 1))
        cols = 1
    else:
        vals = inp
        cols = np.size(vals, 1)

    out = np.zeros((active[0].size, cols), dtype = "complex_")

    for col in range(cols):
        full = fft3(vals[:,col], S, -1)        
        out[:,col] = np.reshape(full,(full.size, 1))[active]
    
    return out


# In[110]:


old_settings = np.seterr(divide='ignore', invalid='ignore')
Vps=-4.*m.pi*Z/G2
Vps[0]=0.
np.seterr(**old_settings)

Vps = np.reshape(Vps, (Vps.size, 1))
Vdual = cJdag(Vps * Sf)

Ns = 1

np.random.seed(100)

W = np.random.randn(active[0].size,Ns) + 1j * np.random.randn(active[0].size,Ns)
W = orthogonalize(W)

W, Elist = sd(W, 30, False)
W = orthogonalize(W)

W, Elist = pccg(W, 30, 1, False)

Psi, epsilon = getPsi(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])
    
print('\nTotal energy:', getE(W) + Ewald, "NIST value: -0.445671")


# #### Full benefit from minimal representation:

# In[111]:


def getn(Psi, f):
    n = np.zeros((np.prod(S), 1))
    
    for col in range(np.size(Psi, axis = 1)):
        IPsi = cI(Psi[:,col])
        n[:,0] += np.reshape(f * np.real(IPsi.conjugate() * IPsi), n.size)
        
    return n


# In[112]:


def getE(W):
    W = orthogonalize(W)

    n = getn(W, f)
    PhiExc = 0.5 * PoissonSolve(n) + cJ(excVWN(n))
    
    E = -f * 0.5 * np.sum(diagouter(L(W), W)) + Vdual.transpose().conjugate() @ n + n.transpose() @ cJdag(O(PhiExc))
    
    return np.real(E[0, 0])


# In[113]:


def H(W):    
    W = orthogonalize(W)
    n = getn(W, f)    

    exc = excVWN(n)
    excp = excpVWN(n)
    
    Veff = Vdual + cJdag(O(PoissonSolve(n) + cJ(exc))) + np.reshape(excp, (excp.size, 1)) * cJdag(O(cJ(n)))
    
    out = -0.5 * L(W)
    
    for col in range(np.size(W, axis = 1)):
        out[:, col] += np.reshape(cIdag(Veff * cI(W[:,col])), np.size(out, axis = 0))        
        
    return out


# In[114]:


np.random.seed(100)

W = np.random.randn(active[0].size,Ns) + 1j * np.random.randn(active[0].size,Ns)
W = orthogonalize(W)

W, Elist = sd(W, 30, False)
W = orthogonalize(W)

W, Elist = pccg(W, 30, 1, False)

Psi, epsilon = getPsi(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])
    
print('\nTotal energy:', getE(W) + Ewald, "NIST value: -0.445671")


# #### Calculation of solid Ge

# In[115]:


a=5.66/0.52917721
R=a * np.diag(np.ones(3))

S = np.array([48, 48, 48])
M, N = CreateIndices(S)

r = M @ splalg.inv(np.diag(S)) @ np.transpose(R)
G = 2. * m.pi * N @ splalg.inv(R)
G2 = np.sum(G * G, axis=1)
G2 = np.reshape(G2, (G2.size, 1))

cellCenter = np.reshape(np.sum(R, axis = 1) / 2, (1, 3))

vecsFromCenter = r - np.ones((np.prod(S), 1)) * cellCenter
dr2 = np.sum(vecsFromCenter * vecsFromCenter, 1)
dr = np.sqrt(dr2)

Z = 4
f = 2

X = a * np.asarray([[0, 0, 0], [0.25, 0.25, 0.25], [0.00, 0.50, 0.50], [0.25, 0.75, 0.75], [0.50, 0.00, 0.50], [0.75, 0.25, 0.75], [0.50, 0.50, 0.00], [0.75, 0.75, 0.25]])

#Sf = np.sum(np.exp(-1j * G @ (X - cellCenter).transpose()), axis = 1)
Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)
Sf = np.reshape(Sf, (Sf.size, 1))

sigma1 = 0.25
g1 = Z * Gaussian(dr, sigma1)

n = cI(cJ(g1) * Sf)
n = np.real(n)

phi = Poisson(n)

Uself = Z*Z/(2.*m.sqrt(m.pi))*(1./sigma1)*np.size(X,0)

Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Ewald = (Unum - Uself)[0,0]
print('Ewald energy:', Ewald)
print('Unum:', Unum[0,0])
print('Uself:', Uself)


# In[116]:


eS = np.reshape(S / 2. + 0.5, (S.size, 1))
edges = np.nonzero(np.any(np.abs(M - np.ones((np.size(M, axis = 0), 1)) @ eS.transpose()) < 1, axis = 1))
G2mx = np.min(G2[edges])
active = np.nonzero(G2 < G2mx / 4.)
G2c = np.reshape(G2[active], (active[0].size, 1))
print("Compression:", G2.size / G2c.size, "Theoretical:", 1./(4.*m.pi*(1./4.)**3./3.))


# In[117]:


def PseudoGe(pos):
    rc = 1.052
    lam = 18.5

    if pos < 1E-10:
        P = 0
    else:
        P = - Z/pos * (1. - m.exp(-lam * pos)) / (1 + m.exp(-lam * (pos - rc)))
        
    return P


# In[118]:


#sz = np.size(r,axis = 0)
#V = np.zeros((sz, 1))
#for p in range(sz):
#    rd = r[p] - cellCenter
#    dist = m.sqrt(np.sum(rd**2, axis = 1))
#    V[p] += PseudoGe(dist)
    
#Vps = cJ(V)
#Vps[0] = 0

#Vdual = cJdag(O(Vps) * Sf)

# pseudopotential in fourier space taken from Arias assignment and translated to python from matlab code
# using the above pseudopotential gives the values 'wrong' (although the zero of energy does not matter, but I'll let this here to be comparable with values given in assignment)
lam=18.5
rc=1.052
Gm = np.sqrt(G2)

old_settings = np.seterr(divide='ignore', invalid='ignore')

Vps = -2. * m.pi * np.exp(-m.pi * Gm/lam) * np.cos(Gm * rc) * (Gm / lam) / (1. - np.exp(-2.*m.pi*Gm/lam))
for i in range(4):
    Vps += (-1)**i * np.exp(-lam * rc * i)/(1. + (i*lam/Gm)**2)
    
Vps = Vps * 4. * m.pi * Z / Gm**2 * (1.+ np.exp(-lam*rc)) - 4. * m.pi * Z/ Gm**2
np.seterr(**old_settings)

i = np.asarray([i for i in range(1, 5)])
Vps[0] = 4. * m.pi * Z * (1. + np.exp(-lam * rc))*(rc**2/2. + 1./lam**2 * (m.pi**2/6. + np.sum((-1)**i * np.exp(-lam*rc*i)/ i**2)))

Vps = np.reshape(Vps, (Vps.size,1))
Vdual = cJ(Vps * Sf)


# In[119]:


dat = np.reshape(np.real(Vdual), S)

fig=plt.figure(figsize=(35, 25))

x = np.arange(0, S[1])
y = np.arange(0, S[2])
xs, ys = np.meshgrid(x, y, indexing='ij')

toplot = dat[0,:,:]
        
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(xs, ys, toplot, cmap='viridis', edgecolor='none')
ax.title.set_text('Real Space')
plt.tight_layout()
plt.show()


# In[120]:


Ns = 16 # 8 atoms, Z = 4, 2 electrons / state

np.random.seed(100)

W = np.random.randn(active[0].size,Ns) + 1j * np.random.randn(active[0].size,Ns)
W = orthogonalize(W)

W, Elist = sd(W, 150, False)
W = orthogonalize(W)

W, Elist = pccg(W, 100, 1, False)


# In[121]:


Psi, epsilon = getPsi(W)
E = getE(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])

Etot = E + Ewald
print('\nTotal energy:', Etot)
print('Electronic energy:', E)
print('Energy/atom:', Etot / 8)
CE = abs(Etot / 8 + 3.7301)
CEeV = CE * 27.21138
print('Cohesive Energy:', CE, 'Hartree, eV:', CEeV, 'Experiment: 3.85', 'error:', abs(CEeV - 3.85) / 3.85 * 100.,'%')


# In[122]:


W = orthogonalize(W)
n = getn(W, f)
img = np.reshape(n,S)
img = img[0]
fig = plt.figure(figsize=(8,6))
fig.suptitle('100', fontsize=22)
plt.pcolormesh(img)
plt.show()


# In[123]:


img = n[np.nonzero(M[:,1] == M[:,2])]
# In the assignments, ppm function is given. That saves the matrix into a ppm file, but the structure of ppm is starting with the top of the image
# so it comes up flipped. This is done here to be the same as in ppm image displayed in the lecture and that can be obtained by doing the assignments in matlab/octave/scilab
img = np.flipud(np.reshape(img, (S[0],S[1]), order = 'F')) 
            
fig = plt.figure(figsize=(8,6))
fig.suptitle('110', fontsize=22)
plt.pcolormesh(img,cmap="plasma")
plt.show()


# #### Variable fillings and verification of the pseudopotential

# In[124]:


def getn(Psi, f):
    prodS = np.prod(S)
    n = np.zeros((prodS, 1))
    
    for col in range(np.size(Psi, axis = 1)):
        IPsi = cI(Psi[:,col])
        n[:,0] += np.reshape(f[col, 0] * np.real(IPsi.conjugate() * IPsi), prodS)
        
    return n


# In[125]:


def getE(W):
    W = orthogonalize(W)

    n = getn(W, f)
    PhiExc = 0.5 * PoissonSolve(n) + cJ(excVWN(n))
    
    E = -0.5 * np.trace(np.diagflat(f) @ (W.transpose().conjugate() @ L(W))) + Vdual.transpose().conjugate() @ n + n.transpose() @ cJdag(O(PhiExc))
        
    return np.real(E[0, 0])


# In[126]:


def Q(inp, U):
    mu, V = splalg.eig(U, check_finite = True)
    mu = np.reshape(mu, (mu.size,1))
    
    denom = np.sqrt(mu) @ np.ones((1, mu.size)) 
    denom = denom + denom.transpose().conjugate()
    
    Vadj = V.transpose().conjugate()

    return V @ ((Vadj @ inp @ V) / denom) @ Vadj


# In[127]:


def getgrad(W):
    Wadj = W.transpose().conjugate()
    OW = O(W)
    U = Wadj @ OW
    Uinv = splalg.inv(U)
    HW = H(W)  
    Umsqrt = splalg.sqrtm(Uinv)
    WadjHW = Wadj @ HW
    Htilde = Umsqrt @ WadjHW @ Umsqrt
    F = np.diagflat(f)
    
    Fconv = (Umsqrt @ F @ Umsqrt)
    
    first = (HW - (OW @ Uinv) @ WadjHW) @ Fconv
    
    second = OW @ (Umsqrt @ Q(Htilde @ F - F @ Htilde, U))

    return first + second


# #### Test the results against the previous results of Ge crystal

# In[128]:


f = np.asarray(2 * np.ones((Ns, 1)))

np.random.seed(100)

W = np.random.randn(active[0].size,Ns) + 1j * np.random.randn(active[0].size,Ns)
W = orthogonalize(W)

W, Elist = sd(W, 150, False)
W = orthogonalize(W)

W, Elist = pccg(W,100,1, False)


# In[129]:


Psi, epsilon = getPsi(W)
E = getE(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])

Etot = E + Ewald
print('\nTotal energy:', Etot)
print('Electronic energy:', E)
print('Energy/atom:', Etot / 8)
CE = abs(Etot / 8 + 3.7301)
CEeV = CE * 27.21138
print('Cohesive Energy:', CE, 'Hartree, eV:', CEeV, 'Experiment: 3.85', 'error:', abs(CEeV - 3.85) / 3.85 * 100.,'%')


# #### Isolated Ge atom

# In[130]:


Ns = 4
f = np.asarray([[2.], [2./3.], [2./3.], [2./3.]])

X = np.asarray([[0, 0, 0]])

Sf = np.sum(np.exp(-1j * G @ X.transpose()), axis = 1)
Sf = np.reshape(Sf, (Sf.size, 1))

sigma1 = 0.25
g1 = Z * Gaussian(dr, sigma1)

n = cI(cJ(g1) * Sf)
n = np.real(n)

phi = Poisson(n)

Uself = Z*Z/(2.*m.sqrt(m.pi))*(1./sigma1)*np.size(X,0)

Unum = 0.5 * np.real(cJ(phi).transpose().conjugate() @ O(cJ(n)))
Ewald = (Unum - Uself)[0,0]
print('Ewald energy:', Ewald)
print('Unum:', Unum[0,0])
print('Uself:', Uself)


# In[131]:


Vdual = cJ(Vps * Sf)


# In[132]:


np.random.seed(100)

W = np.random.randn(active[0].size,Ns) + 1j * np.random.randn(active[0].size,Ns)
W = orthogonalize(W)

W, Elist = sd(W, 600, False)
W = orthogonalize(W)

W, Elist = lm(W, 600, False)

# the preconditioned variants do not work so well here
W = orthogonalize(W)
W, Elist = pclm(W, 10, True)

#W, Elist = pccg(W, 10, 1, True)


# In[133]:


Psi, epsilon = getPsi(W)
E = getE(W)

for i in range(Ns):
    print('State:', i, 'Energy:', epsilon[i])

Etot = E + Ewald
print('\nTotal energy:', Etot)
print('Electronic energy:', E)
print('Energy dif beteen s and p orbitals:', epsilon[1] - epsilon[0], 'Expected (from NIST data): 0.276641')
