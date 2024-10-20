from pynwb import NWBFile
import scipy.io  # for loading in signals.mat files
import pandas as pd  # essential functions
import numpy as np  # essential functions
from scipy.sparse import diags, eye, csc_matrix  # for creating sparse matrices
from scipy.sparse.linalg import spsolve  # for solving sparse linear systems
from sklearn.linear_model import Lasso  # for Lasso regression


def WhittakerSmooth(data, binary_mask, lambda_):
    """
    Penalized least squares algorithm for background fitting

    Wittaker smoothing is a method for fitting a smooth background to noisy data, often used in signal processing to remove baseline noise or trends while preserving the main features of the signal.

    - fits a smooth background to the input data ('data') by minimizaing the sum of squared differences between the data and the background.
    - uses a binary mask ('binary_mask') to identify the signal regions in the data (peaks and troughs) that are not part of the background calculation.
    - uses a penalty term ('lambda_') to control the smoothness of the background. Discourages rapid changes and enforces smoothness.

    input
        data: input data. 1D array that represents the signal data.
        bindary_mask: binary mask that defines which points are signals (peaks) and which are background. 0 = Peak / 1 = Background
        lambda_: parameter that can be adjusted by user. The larger lambda is, the smoother the resulting background. Careful: too large can lead to over smoothing.
    output
        the fitted background vector
    """
    data_matrix = np.matrix(data)  # matrix structure of data in order to use matrix operations
    data_size = data_matrix.size  # size of the data matrix
    ID_matrix = eye(
        data_size, format="csc"
    )  # creates an identity matrix the size of data_size in compressed sparse column (csc) format
    diff_matrix = ID_matrix[1:] - ID_matrix[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    diagonal_matrix = diags(
        binary_mask, 0, shape=(data_size, data_size)
    )  # creates a diagonal matrix with the binary mask values on the diagonal
    A = csc_matrix(
        diagonal_matrix + (lambda_ * diff_matrix.T * diff_matrix)
    )  # represents the combination of the diagonal matrix and the smoothness penalty term Lambda_
    B = csc_matrix(diagonal_matrix * data_matrix.T)  # represents the weighted data
    smoothed_baseline = spsolve(A, B)  # solves the linear system of equations to find the smoothed baseline

    return np.array(smoothed_baseline)


def airPLS(data, lambda_=100, penalty_order=1, max_iterations=15):
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    DOI : 10.1039/b922045c

    This function is used to fit a baseline to the input data x using adaptive iteratively reweighted penalized least squares.
    The baseline is fitted by minimizing the weighted sum of the squared differences between the input data and the baseline.
    The function uses a penalized least squares approach to fit the baseline, with a penalty term that encourages smoothness.
    The function iteratively adjusts the weights and the baseline until convergence is achieved.

    input
        data: input data (i.e. chromatogram of spectrum)
        lambda_: Parameter that can be adjusted by user. The larger lambda is, the smoother the resulting baseline.
        penalty_order: The order of the penalty for the differences in the baseline. Controls how strongly the smoothness constraint is enforced.
        max_iterations: Maximum number of iterations for the algorithm to adjust the weights and fit the baseline.

    output
        the fitted baseline vector
    """

    data_points = data.shape[0]  # number of data points
    weights = np.ones(data_points)  # initial weights set to 1. All points are initially treated equally.

    for i in range(1, max_iterations + 1):
        baseline = WhittakerSmooth(
            data, weights, lambda_, penalty_order
        )  # loop runs 'max_iterations' times to adjust the weights and fit the baseline
        delta = (
            data - baseline
        )  # difference between data and baseline to calculate residuals, how much each data point deviates from the baseline. delta > 0 == peak. delta < 0 == background.
        sum_of_neg_deltas = np.abs(delta[delta < 0].sum())  # how much data is below the baseline.

        if (
            sum_of_neg_deltas < 0.001 * (abs(data)).sum() or i == max_iterations
        ):  # convergence check: if sum_of_neg_deltas is less than 0.001 of the total data, or if the maximum number of iterations is reached, the loop breaks.
            if i == max_iterations:
                print("WARING max iteration reached!")
            break

        weights[delta >= 0] = (
            0  # delta >= 0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        )
        weights[delta < 0] = np.exp(
            i * np.abs(delta[delta < 0]) / sum_of_neg_deltas
        )  # updates the weights for the negative deltas. Gives more weight to larger residuals using an exponential function.
        weights[0] = np.exp(
            i * (delta[delta < 0]).max() / sum_of_neg_deltas
        )  # updates the weights for the first and last data points to ensure edges of data are not ignored or underweighed.
        weights[-1] = weights[0]
    return baseline


def add_photometry(nwbfile: NWBFile, metadata: dict):
    print("Adding photometry...")
    # get metadata for photometry from metadata file
    signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
    # photometry_sampling_rate_in_hz = metadata["photometry"]["sampling_rate"] Commented out to use the old sampling rate variable

    signals = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)
    # used to be : signals = scipy.io.loadmat(datepath+'signals.mat',matlab_compatible=True)

    # raw signals are saved as ndarrays
    ref = signals["ref"]
    sig1 = signals["sig1"]
    sig2 = signals["sig2"]

    # _____________________________________________________________________________________

    SR = 10000  # sampling rate of photometry system
    Fs = 250  # downsample frequency

    # downsample your raw data from 10kHz to 250Hz.
    # Takes every 40th sample and stores it as sig1,sig2, or ref
    green = sig1[0][:: int(SR / Fs)]
    reference = ref[0][:: int(SR / Fs)]
    red = sig2[0][:: int(SR / Fs)]

    # _____________________________________________________________________________________

    # smooth signals

    smooth_window = int(Fs / 30)

    # Convert raw signals to Pandas Series for rolling mean
    raw_green = pd.Series(green)
    raw_reference = pd.Series(reference)
    raw_red = pd.Series(red)

    reference = np.array(raw_reference.rolling(window=smooth_window, min_periods=1).mean()).reshape(
        len(raw_reference), 1
    )
    signal_green = np.array(raw_green.rolling(window=smooth_window, min_periods=1).mean()).reshape(len(raw_green), 1)
    signal_red = np.array(raw_red.rolling(window=smooth_window, min_periods=1).mean()).reshape(len(raw_red), 1)
    # 'min_periods' sets the minimum number of observations required for a valid computation.
    # 1 means if there's only one observation it will still compute the mean

    # _____________________________________________________________________________________

    # Baseline subtraction using airPLS

    lambd = 1e8
    porder = 1
    itermax = 50

    # smoothed background lines
    ref_base = airPLS(raw_reference.T, lambda_=lambd, porder=porder, itermax=itermax).reshape(len(raw_reference), 1)
    g_base = airPLS(raw_green.T, lambda_=lambd, porder=porder, itermax=itermax).reshape(len(raw_green), 1)
    r_base = airPLS(raw_red.T, lambda_=lambd, porder=porder, itermax=itermax).reshape(len(raw_red), 1)

    # subtract the respective moving airPLS baseline from the smoothed signal and reference

    remove = 0
    reference = reference[remove:] - ref_base[remove:]
    gsignal = signal_green[remove:] - g_base[remove:]
    rsignal = signal_red[remove:] - r_base[remove:]

    # _____________________________________________________________________________________

    # Standardize by z-scoring signals and plot

    # Standardization assumes that your observations fit a Gaussian distribution (bell curve)
    # with a well behaved mean and standard deviation.
    z_reference = (reference - np.median(reference)) / np.std(reference)
    gz_signal = (gsignal - np.median(gsignal)) / np.std(gsignal)
    rz_signal = (rsignal - np.median(rsignal)) / np.std(rsignal)

    # _____________________________________________________________________________________

    # Remove the contribution of the reference signal from the signals using a Lasso regression
    # Lasso: Least Absolute Shrinkage and Selection Operator.
    # ...in simple terms, Lasso regression is like a detective that helps you find the simplest
    # equation to predict something by focusing on the most important factors and ignoring the rest.

    # Finds a balance between model simplicity and accuracy.
    # It achieves this by adding a penalty term to the traditional linear regression model,
    # which encourages sparse solutions where some coefficients are forced to be exactly zero.

    lin = Lasso(alpha=0.0001, precompute=True, max_iter=1000, positive=True, random_state=9999, selection="random")

    lin.fit(z_reference, gz_signal)
    lin.fit(z_reference, rz_signal)
    z_reference_fitted = lin.predict(z_reference).reshape(len(z_reference), 1)

    # _____________________________________________________________________________________

    # deltaF / F

    gzdFF = gz_signal - z_reference_fitted
    rzdFF = rz_signal - z_reference_fitted

    # OLD TIM CODE

    # #Make dataframe with all data organized by sample number
    # a = np.tile(0,(len(gzdFF),6)) # each row is a 250Hz time stamp
    # data = np.full_like(a, np.nan, dtype=np.double) #make a sample number x variable number array of nans
    # #fill in nans with behavioral data.
    # # columns == x,y,GRAB-ACh,dLight,port,rwd,roi
    # # assigns values to columns that correspond to their signal
    # data[:,0] = z_reference_fitted.T[0] # fitted z-scored reference
    # data[:,1] = gzdFF.T[0] # green z-scored
    # data[:,2] = rzdFF.T[0] # red z-scored
    # data[:,3] = ref.T[0] # raw 405 reference (Should I add another column for0 'z_reference'?)
    # data[:,4] = sig1 # raw green
    # data[:,5] = sig2 # raw red 565

    # sampledata = pd.DataFrame(data,columns = ['z-ref','green','red','ref','470','565'])

    # z-score and save signal as zscore (this is sz scored twice)
    # gzscored = np.divide(np.subtract(sampledata.green,sampledata.green.mean()),sampledata.green.std())
    # sampledata['green_z_scored'] = gzscored
    # rzscored = np.divide(np.subtract(sampledata.red,sampledata.red.mean()),sampledata.red.std())
    # sampledata['red_z_scored'] = rzscored

    # Create ndx-fiber-photometry objects

    # TODO: extract nosepoke times

    visits = signals["visits"][0]  # This contains all the port visits in NON-DOWNSAMPLED timestamps

    # TODO: make sure these visits correspond or match the number of visits in the behavior file (arduino) (include a sanity check)
    # shiftVisByN = int(input("adjust visits by removing n indices from start?"+\
    # " (input n; if 1 is first index, input 1)"))
    # visits = visits[shiftVisByN:] # @10Khz!!!

    visits = np.divide(visits, SR / Fs)  # timestamps of DOWNSAMPED reward port visits
    visits = visits.astype(int)

    # TODO: add variables that correspond to signals and visits in NWB

    # if photometry exists, it serves as the main clock, so we do not need to realign these timestamps

    # TODO: add to NWB file !!
