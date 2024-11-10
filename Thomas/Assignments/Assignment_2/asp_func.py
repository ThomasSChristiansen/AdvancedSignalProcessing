#-----------------------------------------------
# Bjarke Holm Jørgensen (AU664248)
# Aarhus Universitet (2020-2025)
# Course: Advance Signal Processing (Fall 2023)
#-----------------------------------------------

#Signal Processing Functions

def downsample(x, D, offset = None):
    k = offset
    if k == None:
        y = [x[i*D] for i in range(int(len(x)/D))]
    else:
        temp = [*x[k:],*x[:k]] #Wrap list: The 'offset' first values are appended to the end. 
        y = [temp[i*D] for i in range(int(len(temp)/D))]
    return y

def upsample(x, I):
    from numpy import append, array
    zeros = [0 for i in range(I-1)]
    y = []
    for i in range(len(x)):
        y = append(y, [x[i], *zeros])
        y = array(y).reshape(I*(i+1))
    return y

def convolve_1d(data, w, neg_start_idx = None): #If the window has a negative index, a padding of 0s can be added to the sequence. 
    from numpy import flip
    wf = flip(w)
    I = len(w)
    xdata = [*data] #,*[0 for i in range(I-1)] - Keep this if problems arise
    if neg_start_idx:
        x_idx = [*[0 for _ in range(neg_start_idx)], *xdata]
        y = [sum(x_idx[i:i+len(wf)] * wf) for i in range(len(x_idx)-len(wf)+1)]
    else:
        y = [sum(xdata[i:i+len(wf)] * wf) for i in range(len(xdata)-len(wf)+1)]
    return y

def lin_interp(data, I):
    from numpy import flip
    w = [] #Triangular convolution window
    for x in range(-I, I):
        w.append(1-abs(x)/I)
    wf = flip(w)
    y = convolve_1d(data, wf, I)
    return y

def delay(x, nSampDelay):
    from numpy import array
    if nSampDelay == 0:
        return array(x)
    zeros = [0 for _ in range(nSampDelay)]
    x = x[:-nSampDelay]
    y = [*zeros, *x]
    return array(y)

def zoh_interp(data, I): #Takes upsampled data and upsampling integer, I
    from numpy import ones
    w = ones((I))
    idx = I-1
    y = convolve_1d(data, w, idx)
    return y

def autocorrelation(signal, max_lag=None, half=False):
    from numpy import array, concatenate, linspace, flip
    if max_lag:
        L = max_lag
    else:
        L = len(signal)
    N = len(signal)
    x = array(signal)
    r = [] #ACR values
    
    for i in range(L+1): #For maximum amount of lag (N+1 as we want 0 lag also)
        r_l = 0
        for n in range(N-i):
            r_l += x[n] * x[n+i]
        r.append(r_l * 1/N)
    if half == False:    
        r = concatenate((flip(r), r[1:])) #Mirror the autocorrelation as it is always r[l]=r[-l]
        lag = linspace(-L,L,2*L+1)
    elif half == True:
        lag = linspace(0,L,L+1)
        
    return array(r), lag

def periodogram(data):
    from numpy import linspace, pi, exp
    w = linspace(0,pi,len(data))
    I = []
    for f in w:
        summ = 0
        for i in range(len(data)):
            summ += data[i]*exp(-1j*f*i)
        I.append(abs(summ)**2 / len(data))
    return w/pi, I


#----------PLOTS-----------#

def plotMagResponse(sys_num, sys_den, db = False, ax = None):
    from scipy.signal import freqz
    from numpy import pi, log10, concatenate, flip
    from matplotlib.pyplot import subplots, show
    w, H = freqz(sys_num, sys_den)
    w = concatenate((-flip(w[1:]), w)) / pi
    H = concatenate((flip(H[1:]), H))
    if ax is None:
        fig, ax = subplots(1, figsize=(8,6))
        if db == True:
            H = 10*log10(abs(H))
            ax.set_ylabel("$|H(e^{j\omega})|$ (dB)")
        else:
            H = abs(H)
            ax.set_ylabel("$|H(e^{j\omega})|$")
        ax.plot(w,H)
        ax.set_title("Magnitude Response of System")
        ax.set_xlabel("$\omega\ [x\cdot\pi]$")
        ax.grid()
        show()
    else:
        if db == True:
            H = 10*log10(abs(H))
            ax.plot(w,H)
        else:
            H = abs(H)
            ax.plot(w,H)
                    
def plotImpulseResponse(sys_num, sys_den, impulseLen, ax = None):
    from scipy.signal import lfilter
    from numpy import linspace
    from matplotlib.pyplot import subplots, show
    impulse = [1, *[0 for i in range(impulseLen)]] #Make sure to display all taps of the filter
    n = linspace(0,impulseLen,impulseLen+1)
    filtered = lfilter(sys_num,sys_den,impulse)
    if ax is None:
        fig, ax = subplots(1, figsize=(8,6))
        ax.stem(n, filtered)
        ax.set_title("Impulse Response of System")
        ax.set_xlabel("n")
        ax.set_ylabel("$h[n]$")
        ax.grid()
        show()
    else:
        ax.stem(n, filtered)
        show()

def plotPhaseResponse(sys_num, sys_den, ax = None):
    from scipy.signal import freqz
    from numpy import unwrap, angle, pi
    from matplotlib.pyplot import subplots, show
    w, H = freqz(sys_num, sys_den, whole=True)
    angles = unwrap(angle(H))
    if ax is None:
        fig, ax = subplots(1, figsize=(8,6))
        ax.plot(w/pi,angles)
        ax.set_title("Phase Response of System")
        ax.set_xlabel("$\omega\ [x\cdot\pi]$")
        ax.set_ylabel("Angle (Radians)")
        ax.grid()
        show()
    else:
        ax.plot(w/pi, angles)
                    
def pzplot(sys_num, sys_den, ax = None):#zeros and poles are each 1D arrays of complex numbers
    from scipy.signal import tf2zpk
    from numpy import linspace, cos, sin, pi
    from matplotlib.pyplot import subplots, show
    zs, ps, K = tf2zpk(sys_num, sys_den)
    if not len(zs): #Generate zeros in 0+0j
        zs = [0+0j for i in range(len(ps))]
    angle = linspace(0,2*pi, 150)
    radius = 1
    x = radius * cos(angle)
    y = radius * sin(angle)
    zs = [[zs[i].real, zs[i].imag] for i in range(len(zs))]
    ps = [[ps[i].real, ps[i].imag] for i in range(len(ps))]
    if ax is None:
        fig, ax = subplots(1, figsize = (8,8))
        ax.set_aspect("equal")
        ax.set_xlabel("Real Axis")
        ax.set_ylabel("Imaginary Axis")
        ax.plot(x,y, color='black')
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--')
        ax.grid(True)
        ax.locator_params(nbins=10) #Grid spacing
        ax.set_title("Pole-Zero Plot")
        for i in range(len(zs)):
            ax.scatter(zs[i][0], zs[i][1], s=60, facecolors='none', edgecolors='b', label="Zeros")
        for i in range(len(ps)):
            ax.scatter(ps[i][0], ps[i][1], s=60, c='r', marker='x', label="Poles")
        ax.legend()
        show()
    else:
        for i in range(len(zs)):
            ax.scatter(zs[i][0], zs[i][1], s=60, facecolors='none', edgecolors='b', label="Zeros")
        for i in range(len(ps)):
            ax.scatter(ps[i][0], ps[i][1], s=60, c='r', marker='x', label="Poles")
        ax.plot(x,y, color='black')
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--')
        ax.locator_params(nbins=10)