from pynwb import NWBFile
import numpy as np
import struct
from scipy.signal import butter, filtfilt, lfilter, hilbert


def add_photometry(nwbfile: NWBFile, metadata: dict):
    print("Adding photometry...")
    # get metadata for photometry from metadata file
    signals_phot_file_path = metadata["photometry"]["signals_phot_file_path"]
    signals_box_file_path = metadata["photometry"]["signals_box_file_path"]
    
    photometry_sampling_rate_in_hz = metadata["photometry"]["sampling_rate"]

    # TODO: extract photometry signals

    # Create ndx-fiber-photometry objects

    # TODO: extract nosepoke times

    # if photometry exists, it serves as the main clock, so we do not need to realign these timestamps

    # TODO: add to NWB file

    # TODO: How to give error signal / sanity check?
    



def read_phot_data(signals_phot_file_path):
    phot = {}
    phot_file = signals_phot_file_path + '*.phot'

    with open(phot_file, 'rb') as fid:
        
        # read binary data from the file, specifying the big-endian data format '>'
        phot['magic_key'] = struct.unpack('>I', fid.read(4))[0]
        phot['header_size'] = struct.unpack('>h', fid.read(2))[0]  
        phot['main_version'] = struct.unpack('>h', fid.read(2))[0]
        phot['secondary_version'] = struct.unpack('>h', fid.read(2))[0]
        phot['sampling_rate'] = struct.unpack('>h', fid.read(2))[0]
        phot['bytes_per_sample'] = struct.unpack('>h', fid.read(2))[0]
        phot['num_channels'] = struct.unpack('>h', fid.read(2))[0]    

       # after reading the character arrays, decode them from utf-8 and stripping the null characters (\x00)
        phot['file_name'] = fid.read(256).decode('utf-8').strip('\x00')
        phot['date'] = fid.read(256).decode('utf-8').strip('\x00')
        phot['time'] = fid.read(256).decode('utf-8').strip('\x00')

        # loop through the four channels and extract the location, signal, frequency, max and min values in the same way
        phot['channels'] = []

        # Initialize a list of empty dictionaries for the channels
        for i in range(4):
            phot['channels'].append({})

        # Read and decode the Location for all channels first
        for i in range(4):
            phot['channels'][i]['location'] = fid.read(256).decode('utf-8', errors='ignore').strip('\x00')

        # Read and decode the Signal for all channels
        for i in range(4):
            phot['channels'][i]['signal'] = fid.read(256).decode('utf-8', errors='ignore').strip('\x00')

        # Read Frequency for all channels
        for i in range(4):
            phot['channels'][i]['freq'] = struct.unpack('>h', fid.read(2))[0]

        # Read Max Voltage for all channels
        for i in range(4):
            phot['channels'][i]['max_v'] = struct.unpack('>h', fid.read(2))[0] / 32767.0

        # Read Min Voltage for all channels
        for i in range(4):
            phot['channels'][i]['min_v'] = struct.unpack('>h', fid.read(2))[0] / 32767.0

        phot['signal_label'] = []
        for signal in range(8):
            # phot['signal_label'].append(fid.read(256).decode('utf-8').strip('\x00'))
            signal_label = fid.read(256).decode('utf-8').strip('\x00')
            phot['signal_label'].append(signal_label)

        # handle the padding by reading until the header size is reached
        position = fid.tell()
        pad_size = phot['header_size'] - position
        phot['pad'] = fid.read(pad_size)

        # reshape the read data into a 2D array where the number of channels is the first dimension
        data = np.fromfile(fid, dtype=np.dtype('>i2'))
        phot['data'] = np.reshape(data, (phot['num_channels'], -1), order='F')

    print(phot)
    return phot


def read_phot_data(signals_box_file_path):
    box = {}
    box_file = signals_box_file_path + '*.box'

    with open(box_file, 'rb') as fid:
        
        # Read binary data with big-endian (">")
        box['magic_key'] = struct.unpack('>I', fid.read(4))[0]
        box['header_size'] = struct.unpack('>h', fid.read(2))[0]
        box['main_version'] = struct.unpack('>h', fid.read(2))[0]    
        box['secondary_version'] = struct.unpack('>h', fid.read(2))[0]
        box['sampling_rate'] = struct.unpack('>h', fid.read(2))[0]
        box['bytes_per_sample'] = struct.unpack('>h', fid.read(2))[0]
        box['num_channels'] = struct.unpack('>h', fid.read(2))[0]

        # Read and decode file name, date, and time
        box['file_name'] = fid.read(256).decode('utf-8').strip('\x00')
        box['date'] = fid.read(256).decode('utf-8').strip('\x00')
        box['time'] = fid.read(256).decode('utf-8').strip('\x00')    

        # Read channel locations
        box['ch1_location'] = fid.read(256).decode('utf-8').strip('\x00')
        box['ch2_location'] = fid.read(256).decode('utf-8').strip('\x00')
        box['ch3_location'] = fid.read(256).decode('utf-8').strip('\x00')    

        # Get current file position
        position = fid.tell()
        
        # Calculate pad size and read padding
        pad_size = box['header_size'] - position

        box['pad'] = fid.read(pad_size)

        # Read the remaining data and reshape it
        data = np.fromfile(fid, dtype=np.uint8)
        box['data'] = np.reshape(data, (box['num_channels'], -1), order='F')

    print(box)
    return box


def process_pulses(box):
    diff_data = np.diff(box['data'][2, :].astype(np.int16))
    start = np.where(diff_data < -1)[0][0]
    pulses = np.where(diff_data > 1)[0]
    visits = pulses - start

    return visits


def lockin_detection(input_signal, exc1, exc2, Fs, **kwargs):
    # Default values
    filter_order = 5
    tau = 10
    de_trend = False
    full = False

    # Parse optional arguments
    for key, value in kwargs.items():
        if key == 'tau':
            tau = value
        elif key == 'filterorder':
            filter_order = value
        elif key == 'detrend':
            de_trend = value
        elif key == 'full':
            full = value
        else:
            print(f'Invalid optional argument: {key}')

    tau /= 1000  # Convert to seconds
    Fc = 1 / (2 * np.pi * tau)
    fL = 0.01

    # High-pass filter design (Same as MATLAB filter design)
    b, a = butter(filter_order, Fc / (Fs / 2), 'high')
    
    # Single-direction filtering to match MATLAB's 'filter'
    input_signal = lfilter(b, a, input_signal)

    # Demodulation
    demod1 = input_signal * exc1
    demod2 = input_signal * exc2

    # Trend filter design
    if de_trend:
        b, a = butter(filter_order, [fL, Fc] / (Fs / 2))
    else:
        b, a = butter(filter_order, Fc / (Fs / 2))

    if not full:
        # Use lfilter for single-direction filtering
        sig1 = lfilter(b, a, demod1)
        sig2 = lfilter(b, a, demod2)
    else:
        # Full mode
        sig1x = lfilter(b, a, demod1)
        sig2x = lfilter(b, a, demod2)

        # Get imaginary part of the Hilbert transform for phase-shifted signal
        exc1_hilbert = np.imag(hilbert(exc1))
        exc2_hilbert = np.imag(hilbert(exc2))

        demod1 = input_signal * exc1_hilbert
        demod2 = input_signal * exc2_hilbert
        
        if de_trend:
            b, a = butter(filter_order, [fL, Fc] / (Fs / 2))
        else:
            b, a = butter(filter_order, Fc / (Fs / 2))
        
        sig1y = lfilter(b, a, demod1)
        sig2y = lfilter(b, a, demod2)

        # Combine signals using Pythagorean theorem
        sig1 = np.sqrt(sig1x**2 + sig1y**2)
        sig2 = np.sqrt(sig2x**2 + sig2y**2)

    return sig1, sig2


def lockin_detection(input_signal, exc1, exc2, Fs, **kwargs):
    tau = 10
    filter_order = 5

    # Get the necessary data from the phot structure
    detector = phot['data'][5, :]
    exc1 = phot['data'][6, :]
    exc2 = phot['data'][7, :]

    # Call lockin_detection function
    sig1, ref = lockin_detection(detector, exc1, exc2, phot['sampling_rate'], tau=tau, filterorder=filter_order, detrend=False, full=True)

    detector = phot['data'][2, :]
    exc1 = phot['data'][0, :]
    exc2 = phot['data'][1, :]

    # Call lockin_detection function for the second set of signals
    sig2, ref2 = lockin_detection(detector, exc1, exc2, phot['sampling_rate'], tau=tau, filterorder=filter_order, detrend=False, full=True)

    # Cut off the beginning of the signal to match behavioral data
    sig1 = sig1[start:]
    sig2 = sig2[start:]
    ref = ref[start:]
    ref2 = ref2[start:]

    loc = phot['channels'][2]['location'][:15]  # First 15 characters of the location

    return sig1, sig2, ref, ref2, loc

