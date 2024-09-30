from pynwb import NWBFile



def add_photometry(nwbfile: NWBFile, metadata: dict):
    print("Adding photometry...")
    # get metadata for photometry from metadata file
    signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
    photometry_sampling_rate_in_hz = metadata["photometry"]["sampling_rate"]

    # TODO: extract photometry signals
# Jose:
# I copied over the code most relevant to the TODO from Python scripts, assuming MATLAB is already convered the binary data. 
    filepath =  'T:\\ACh Rats\\80B8CE6 (Ceecee)\\'
    date = '01122024-L'
    datepath = filepath + date + '\\'
    signals = scipy.io.loadmat(datepath+'signals.mat',matlab_compatible=True)

    # raw signals are saved as ndarrays 
    ref = signals['ref']
    sig1 = signals['sig1']
    sig2 = signals['sig2']

    SR = 10000 #sampling rate of photometry system
    Fs = 250 #downsample frequency 

    #downsample your raw data from 10kHz to 250Hz. 
    #Takes every 40th sample and stores it as sig1,sig2, or ref
    sig1 = sig1[0][::int(SR/Fs)]
    ref = ref[0][::int(SR/Fs)]
    sig2 = sig2[0][::int(SR/Fs)]

    sigs = pd.DataFrame()
    sigs['ref'] = ref
    sigs['green'] = sig1
    sigs['red'] = sig2

    # define reference and signal traces for analysis
    raw_reference = sigs['ref']
    raw_green = sigs['green']
    raw_red = sigs['red']

    smooth_window = int(Fs/30)
    # smooth signals and plot
    reference = np.array(raw_reference.rolling(window=smooth_window,min_periods=1).mean()).reshape(len(raw_reference),1) 
    # this finds the rolling average of the signal given the time window of 250Hz/30(30 second bins)
    # 'min_periods' sets the minimum number of observations required for a valid computation. 
        # 1 means if there's only one observation it will still compute the mean
        
    signal_green = np.array(raw_green.rolling(window=smooth_window,min_periods=1).mean()).reshape(len(raw_green),1)
    green_xvals = np.arange(0,len(signal_green))/Fs/60 # turns your time scale to mintes bc every 250th sample is one second.

    signal_red = np.array(raw_red.rolling(window=smooth_window,min_periods=1).mean()).reshape(len(raw_red),1)
    red_xvals = np.arange(0,len(signal_red))/Fs/60 # turns your time scale to mintes bc every 250th sample is one second.

    #find baseline for sig and ref

    from scipy.sparse import csc_matrix, eye, diags
    from scipy.sparse.linalg import spsolve

    def WhittakerSmooth(x,w,lambda_,differences=1):
        '''
        Penalized least squares algorithm for background fitting
        
        input
            x: input data (i.e. chromatogram of spectrum)
            w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
            differences: integer indicating the order of the difference of penalties
        
        output
            the fitted background vector
        '''
        X=np.matrix(x)
        m=X.size
        i=np.arange(0,m)
        E=eye(m,format='csc')
        D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
        W=diags(w,0,shape=(m,m))
        A=csc_matrix(W+(lambda_*D.T*D))
        B=csc_matrix(W*X.T)
        background=spsolve(A,B)
        return np.array(background)

    def airPLS(x, lambda_=100, porder=1, itermax=15):
        '''
        Adaptive iteratively reweighted penalized least squares for baseline fitting
        
        input
            x: input data (i.e. chromatogram of spectrum)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
            porder: adaptive iteratively reweighted penalized least squares for baseline fitting
        
        output
            the fitted background vector
        '''
        m=x.shape[0]
        w=np.ones(m)
        for i in range(1,itermax+1):
            z=WhittakerSmooth(x,w,lambda_, porder)
            d=x-z
            dssn=np.abs(d[d<0].sum())
            if(dssn<0.001*(abs(x)).sum() or i==itermax):
                if(i==itermax): print('WARING max iteration reached!')
                break
            w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
            w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
            w[0]=np.exp(i*(d[d<0]).max()/dssn) 
            w[-1]=w[0]
        return z

    lambd = 1e8
    porder = 1
    itermax = 50

    #smoothed background lines 
    ref_base=airPLS(raw_reference.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_reference),1)
    g_base=airPLS(raw_green.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_green),1)
    r_base=airPLS(raw_red.T,lambda_=lambd,porder=porder,itermax=itermax).reshape(len(raw_red),1)

    # subtract the respective moving airPLS baseline from the smoothed signal and reference 

    remove = 0
    reference = (reference[remove:] - ref_base[remove:])
    gsignal = (signal_green[remove:] - g_base[remove:])
    rsignal = (signal_red[remove:] - r_base[remove:])

    # standardize/ z-score signals and plot
    # Standardization assumes that your observations fit a Gaussian distribution (bell curve) 
    #with a well behaved mean and standard deviation.
    z_reference = (reference - np.median(reference)) / np.std(reference)
    gz_signal = (gsignal - np.median(gsignal)) / np.std(gsignal)
    rz_signal = (rsignal - np.median(rsignal)) / np.std(rsignal)

    # Lasso: Least Absolute Shrinkage and Selection Operator.
    # in simple terms: Lasso regression is like a detective that helps you find the simplest 
    #equation to predict something by focusing on the most important factors and ignoring the rest. 

    # Finds a balance between model simplicity and accuracy. 
    # It achieves this by adding a penalty term to the traditional linear regression model, 
    #which encourages sparse solutions where some coefficients are forced to be exactly zero. 

    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')

    lin.fit(z_reference, gz_signal)
    lin.fit(z_reference, rz_signal)
    z_reference_fitted = lin.predict(z_reference).reshape(len(z_reference),1)

    # deltaF / F is calculated here
    gzdFF = (gz_signal - z_reference_fitted)
    rzdFF = (rz_signal - z_reference_fitted)


    #Make dataframe with all data organized by sample number
    visits = np.unique(visits)
    a = np.tile(0,(len(gzdFF),22)) # each row is a 250Hz time stamp 
    data = np.full_like(a, np.nan, dtype=np.double) #make a sample number x variable number array of nans
    #fill in nans with behavioral data. 
    # columns == x,y,GRAB-ACh,dLight,port,rwd,roi
    # assigns values to columns that correspond to their signal
    data[:,0] = z_reference_fitted.T[0] # fitted z-scored reference
    data[:,1] = gzdFF.T[0] # green z-scored
    data[:,2] = rzdFF.T[0] # red z-scored
    data[:,3] = ref.T[0] # raw 405 reference (Should I add another column for0 'z_reference'?)
    data[:,4] = sig1 # raw green
    data[:,5] = sig2 # raw red 565

    sampledata = pd.DataFrame(data,columns = ['z-ref','green','red','ref','470','565'])

    #z-score and save signal as zscore
    gzscored = np.divide(np.subtract(sampledata.green,sampledata.green.mean()),sampledata.green.std())
    sampledata['green_z_scored'] = gzscored
    rzscored = np.divide(np.subtract(sampledata.red,sampledata.red.mean()),sampledata.red.std())
    sampledata['red_z_scored'] = rzscored 

    sampledata = reduce_mem_usage(sampledata)

    #save dataframe
    sampledata.to_csv(savepath+'sampleframe.csv')

    # Create ndx-fiber-photometry objects

    # TODO: extract nosepoke times

    # if photometry exists, it serves as the main clock, so we do not need to realign these timestamps

    # TODO: add to NWB file


    