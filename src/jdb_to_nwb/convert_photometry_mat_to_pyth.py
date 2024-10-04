from pynwb import NWBFile
import numpy as np
import struct



def import_photometry(nwbfile: NWBFile, metadata: dict):
    print("Importing photometry...")
    # get metadata for photometry from metadata file
    signals_phot_file_path = metadata["photometry"]["signals_mat_file_path"]

    # TODO: import phot and box files and generate dataframe to process

def read_photometry_data(file):
    phot = {}
    phot_file = file + '.phot'

    with open(phot_file, 'rb') as fid:
        phot['magic_key'] = struct.unpack('I', fid.read(4))[0]
        phot['header_size'] = struct.unpack('h', fid.read(2))[0]
        phot['main_version'] = struct.unpack('h', fid.read(2))[0]
        phot['secondary_version'] = struct.unpack('h', fid.read(2))[0]
        phot['sampling_rate'] = struct.unpack('h', fid.read(2))[0]
        phot['bytes_per_sample'] = struct.unpack('h', fid.read(2))[0]
        phot['num_channels'] = struct.unpack('h', fid.read(2))[0]

        phot['file_name'] = fid.read(256).decode('utf-8').strip('\x00')
        phot['date'] = fid.read(256).decode('utf-8').strip('\x00')
        phot['time'] = fid.read(256).decode('utf-8').strip('\x00')

        phot['channels'] = []
        for i in range(4):
            channel = {}
            channel['location'] = fid.read(256).decode('utf-8').strip('\x00')
            channel['signal'] = fid.read(256).decode('utf-8').strip('\x00')
            channel['freq'] = struct.unpack('h', fid.read(2))[0]
            channel['max_v'] = struct.unpack('h', fid.read(2))[0] / 32767
            channel['min_v'] = struct.unpack('h', fid.read(2))[0] / 32767
            phot['channels'].append(channel)

        phot['signal_label'] = []
        for signal in range(8):
            phot['signal_label'].append(fid.read(256).decode('utf-8').strip('\x00'))

        position = fid.tell()
        pad_size = phot['header_size'] - position
        phot['pad'] = fid.read(pad_size)

        data = np.fromfile(fid, dtype=np.int16)
        phot['data'] = np.reshape(data, (phot['num_channels'], -1))

    return phot

def read_box_data(file):
    box = {}
    box_file = file + '.box'

    with open(box_file, 'rb') as fid:
        box['magic_key'] = struct.unpack('I', fid.read(4))[0]
        box['header_size'] = struct.unpack('h', fid.read(2))[0]
        box['main_version'] = struct.unpack('h', fid.read(2))[0]
        box['secondary_version'] = struct.unpack('h', fid.read(2))[0]
        box['sampling_rate'] = struct.unpack('h', fid.read(2))[0]
        box['bytes_per_sample'] = struct.unpack('h', fid.read(2))[0]
        box['num_channels'] = struct.unpack('h', fid.read(2))[0]

        box['file_name'] = fid.read(256).decode('utf-8').strip('\x00')
        box['date'] = fid.read(256).decode('utf-8').strip('\x00')
        box['time'] = fid.read(256).decode('utf-8').strip('\x00')

        box['ch1_location'] = fid.read(256).decode('utf-8').strip('\x00')
        box['ch2_location'] = fid.read(256).decode('utf-8').strip('\x00')
        box['ch3_location'] = fid.read(256).decode('utf-8').strip('\x00')

        position = fid.tell()
        pad_size = box['header_size'] - position
        box['pad'] = fid.read(pad_size)

        data = np.fromfile(fid, dtype=np.uint8)
        box['data'] = np.reshape(data, (box['num_channels'], -1))

    return box

def process_pulses(box_data):
    diff_data = np.diff(box_data['data'][2, :])
    start = np.where(diff_data < -1)[0][0]
    pulses = np.where(diff_data > 1)[0]
    visits = pulses - start

    return visits

from scipy.signal import butter, filtfilt, hilbert

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

    # High-pass filter design
    b, a = butter(filter_order, Fc / (Fs / 2), 'high')
    input_signal = filtfilt(b, a, input_signal)

    # Demodulation
    demod1 = input_signal * exc1
    demod2 = input_signal * exc2

    # Trend filter design
    if de_trend:
        b, a = butter(filter_order, [fL, Fc] / (Fs / 2))
    else:
        b, a = butter(filter_order, Fc / (Fs / 2))

    if not full:
        sig1 = filtfilt(b, a, demod1)
        sig2 = filtfilt(b, a, demod2)
    else:
        sig1x = filtfilt(b, a, demod1)
        sig2x = filtfilt(b, a, demod2)

        # Get imaginary part of the Hilbert transform for phase-shifted signal
        exc1_hilbert = np.imag(hilbert(exc1))
        exc2_hilbert = np.imag(hilbert(exc2))

        demod1 = input_signal * exc1_hilbert
        demod2 = input_signal * exc2_hilbert

        sig1y = filtfilt(b, a, demod1)
        sig2y = filtfilt(b, a, demod2)

        # Combine signals using Pythagorean theorem
        sig1 = np.sqrt(sig1x**2 + sig1y**2)
        sig2 = np.sqrt(sig2x**2 + sig2y**2)

    return sig1, sig2

def lockin_detection_main(phot, start, pathstr):
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

    # Save signals to a file
    np.savez(f"{pathstr}/signals.npz", sig1=sig1, sig2=sig2, ref=ref, loc=loc, visits=phot['visits'])
