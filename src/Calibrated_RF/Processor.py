import os
import numpy as np
import random


class POWDERRF_Processor:
    """Loader and sampler for the POWDER protocol co-channel RF dataset.

    Discovers ``.out`` / ``.bin`` capture files, filters them by occupancy,
    and draws random IQ windows for training or evaluation.

    Parameters
    ----------
    datadirs : list of str, optional
        Directories that contain POWDER capture files.
    remove_tx : list of int, optional
        Transmitter indices that must not be active. Files with any of these
        transmitters on are dropped.
    must_have_tx : list of int, optional
        Transmitter indices that must be active in every kept file.
    max_tx : int or float, optional
        Maximum number of active transmitters in a kept capture
        (default ``inf``).
    min_tx : int, optional
        Minimum number of active transmitters in a kept capture
        (default ``-1``).

    Attributes
    ----------
    samples_per_sig : int
        Number of windows drawn from each randomly chosen file.
    file_samples : int
        Number of complex samples assumed per capture file.
    Fs : float
        Sampling rate in Hz.
    FFT_length : int
        FFT size used when ``signal_type="fft"``.
    carrier_freq : float
        Carrier frequency in Hz.
    filepaths : list of str
        Capture files that passed the occupancy filters.
    """

    samples_per_sig = 1
    file_samples = 20000000
    Fs = 33.33e6
    FFT_length = 1024
    carrier_freq = 2.425e9

    def __init__(
        self,
        datadirs=[],
        remove_tx=[],
        must_have_tx=[],
        max_tx=float("inf"),
        min_tx=-1,
    ):
        self.filepaths = []
        self.remove_tx = list(remove_tx)
        self.must_have_tx = list(must_have_tx)
        self.max_tx = max_tx
        self.min_tx = min_tx
        for i in range(len(datadirs)):
            if datadirs[i][-1] != "/":
                datadirs[i] += "/"
            files = os.listdir(datadirs[i])
            for file in files:
                if (".out" not in file) and (".bin" not in file):
                    continue
                tx_vector = file.split("_")[1].replace("tx", "")
                tx_vector_sum = sum(int(tx_state) for tx_state in tx_vector)

                if tx_vector_sum > max_tx:
                    continue
                if tx_vector_sum < min_tx:
                    continue

                INDEX_FLAG = False
                for index in remove_tx:
                    if int(tx_vector[index]) == 1:
                        INDEX_FLAG = True
                if INDEX_FLAG:
                    continue

                NONINDEX_FLAG = False
                for index in must_have_tx:
                    if int(tx_vector[index]) == 0:
                        NONINDEX_FLAG = True
                if NONINDEX_FLAG:
                    continue

                filepath = datadirs[i] + file
                self.filepaths.append(filepath)
        self.datadirs = datadirs
        self.sample_count = 0

   
    def __call__(
        self,
        train_test_split=0.8,
        signal_type="IQ_All",
        sample_len=1024,
        n_samples=1000,
        sample_index=[0, 20e6],
    ):
        """Sample windows from the filtered capture files and split them.

        Parameters
        ----------
        train_test_split : float, optional
            Fraction of sampled windows assigned to the training split
            (default ``0.8``).
        signal_type : str, optional
            Feature representation. See :meth:`sample_signal`.
        sample_len : int, optional
            Number of complex samples per window (default ``1024``).
        n_samples : int, optional
            Total number of windows to draw (default ``1000``).
        sample_index : list of float or None, optional
            Optional ``[start, end]`` complex-sample range within each file.

        Returns
        -------
        tuple of numpy.ndarray
            ``(x_train, y_train, x_test, y_test)``. Labels are multi-hot
            transmitter occupancy vectors with ``label_drop_tx`` columns dropped.
        """
  
        x_data = []
        y_data = []
        files = self.filepaths

        for file_key in range(n_samples//self.samples_per_sig):
            file_path = files[random.randint(0, len(files) - 1)]
            file_name = file_path.split("/")[-1]
            x_samples = self.sample_signal(file_path, sample_len = sample_len, n_samples = self.samples_per_sig, signal_type=signal_type, sample_index=sample_index)
            y_samples = file_name.split("_")[1].replace("tx", "")
            y_samples = [int(y_samp) for y_samp in y_samples]
            #delete unwanted indicies/classes
            for i in sorted(self.remove_tx, reverse=True):
                del y_samples[i]
            y_data.extend([y_samples]*len(x_samples))
            x_data.extend(x_samples)

        y_data = np.array(y_data)
        x_data = np.array(x_data)
                
        shuffle_index = list(range(0, x_data.shape[0]))
        random.shuffle(shuffle_index)
        y_data = y_data[shuffle_index]
        x_data = x_data[shuffle_index]

        train_split_index = int(np.floor(x_data.shape[0] * train_test_split))
        x_train = x_data[:train_split_index].astype(np.float32)
        x_test = x_data[train_split_index:].astype(np.float32)
        y_train = y_data[:train_split_index].astype(np.int8)
        y_test = y_data[train_split_index:].astype(np.int8)

        return x_train, y_train, x_test, y_test

    def sample_signal(
        self,
        file_path,
        sample_len=1024,
        n_samples=10,
        signal_type="fft",
        sample_index=[0, 20e6],
    ):
        """Read random IQ windows from a capture file and convert features.

        Parameters
        ----------
        file_path : str
            Path to a complex64 memory-mapped capture.
        sample_len : int, optional
            Window length in complex samples (default ``100``).
        n_samples : int, optional
            Number of windows to draw from this file (default ``10``).
        signal_type : {"Complex", "Real", "Imag", "Mag", "Phase", "fft", "IQ_All"}
            Feature layout to return:

            - ``"Complex"``: real and imag channels
            - ``"Real"`` / ``"Imag"`` / ``"Mag"`` / ``"Phase"``: a single channel
            - ``"fft"``: FFT of the window as ``[I, Q, Mag, Phase]``
            - ``"IQ_All"``: time-domain ``[I, Q, Mag, Phase]``
        sample_index : list of float or None, optional
            If ``[start, end]``, draw offsets from that range.

        Returns
        -------
        numpy.ndarray
            Sampled feature tensor. Channel layouts with more than one feature
            have shape ``(n_samples, n_channels, sample_len)``.
        """

        signal_list = []
        for count in range(n_samples):
            samples_offset = random.randint(sample_index[0], sample_index[1] - 1 - sample_len)
            samples = np.memmap(file_path, dtype = np.complex64, mode = "r", offset=samples_offset*np.dtype(np.complex64).itemsize, shape = (sample_len,))
            samples = np.array(samples)
            #probably does not need to made list, just stack below
            #samples = samples / np.sqrt(np.mean(np.abs(samples)**2))
            signal_list.append(samples)

        IQ_samples = np.stack(signal_list, axis = 0)  # shape (N, 1024)
        if signal_type == "Complex":
            I_ = np.expand_dims(np.real(IQ_samples), axis=1)
            Q_ = np.expand_dims(np.imag(IQ_samples), axis=1)
            samples = np.concatenate([I_, Q_], axis=1)
        elif signal_type == "Real":
            samples = np.real(IQ_samples)
        elif signal_type == "Imag":
            samples = np.imag(IQ_samples)
        elif signal_type == "Mag":
            samples = np.abs(IQ_samples)
        elif signal_type == "Phase":
            samples = np.arctan2(np.imag(IQ_samples), np.real(IQ_samples))
        elif signal_type == "fft":
            window = np.blackman(IQ_samples.shape[-1])
            windowed_samples = IQ_samples * window
            spectrum = np.fft.fftshift(
                np.fft.fft(windowed_samples, n=self.FFT_length, axis=-1),
                axes=-1,
            )
            Mag = np.expand_dims(np.abs(spectrum), axis=1)
            Phase = np.expand_dims(np.arctan2(np.imag(spectrum), np.real(spectrum)), axis=1)
            I_ = np.expand_dims(np.real(spectrum), axis=1)
            Q_ = np.expand_dims(np.imag(spectrum), axis=1)
            samples = np.concatenate([I_, Q_, Mag, Phase], axis=1)
        elif signal_type == "IQ_All":
            Mag = np.expand_dims(np.abs(IQ_samples), axis=1)
            Phase = np.expand_dims(np.arctan2(np.imag(IQ_samples), np.real(IQ_samples)), axis=1)
            I_ = np.expand_dims(np.real(IQ_samples), axis=1)
            Q_ = np.expand_dims(np.imag(IQ_samples), axis=1)
            samples = np.concatenate([I_, Q_, Mag, Phase], axis=1)
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")

        return samples
