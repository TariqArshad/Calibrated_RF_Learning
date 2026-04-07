import os
import numpy as np
import random


class POWDERRF_Processor():
    #how many signals to sample from a single file when doing random sampling from dataset, more samples means faster loading 
    samples_per_sig = 1
    file_samples = 20000000
    #fft params
    Fs = 33.33e6
    FFT_length = 1024
    carrier_freq = 2.425e9
    def __init__(self, datadirs = [] , remove_index = [], must_have_index = [], cochannel = True, remove_empty_channel = True):
        """
        Dataloader for the POWDER Protocol Cochannel dataset
        datadirs = dirctories holding powder data
        remove_index = transmitters to remove 
        must_have_index = transmitters that must be in every signal
        """
        self.filepaths = []
        self.remove_index = remove_index
        for i in range(len(datadirs)): 
            if datadirs[i][-1] != "/": datadirs[i] += "/"
            files = os.listdir(datadirs[i])
            for file in files:
                #only keep files with data
                if (".out" not in file) and (".bin" not in file): continue
                tx_vector = file.split("_")[1].replace("tx", "")
                tx_vector_sum = 0
                for tx_state in tx_vector: tx_vector_sum += int(tx_state)
                #Filter out empty channel
                if (tx_vector_sum == 0) and remove_empty_channel: continue
                #Filter out unwanted signals
                INDEX_FLAG = False
                for index in remove_index:
                    if int(tx_vector[index]) == 1: INDEX_FLAG = True
                if INDEX_FLAG: continue  
                #keep must have signals
                NONINDEX_FLAG = False
                for index in must_have_index:
                    if int(tx_vector[index]) == 0: NONINDEX_FLAG = True
                if NONINDEX_FLAG: continue 

                #If doing single channel remove cochannel signals
                if (cochannel == False) and (tx_vector_sum > 1): continue
                filepath = datadirs[i] + file
                self.filepaths.append(filepath)
        self.datadirs = datadirs
        self.sample_count = 0

    def __call__(self,train_test_split = .8, signal_type = "All", sample_len = 1024 ,n_samples = 1000):
        """
        train_test_split = percetnage of overall dataset to be training
        signal_type = if signal should be All,real, imag, Mag, or Phase
        n_samples = number of signals to samples
        filter_index =  
        """
    
        x_data = []
        y_data = []

        files = self.filepaths
        
        for file_key in range(n_samples//self.samples_per_sig):
            file_path = files[random.randint(0, len(files) - 1)]
            file_name = file_path.split("/")[-1]
            x_samples = self.sample_signal(file_path, sample_len = sample_len, n_samples = self.samples_per_sig, signal_type=signal_type)
            
            y_samples = file_name.split("_")[1].replace("tx", "")
            y_samples = [int(y_samp) for y_samp in y_samples]
            #delete unwanted indicies/classes
            for i in sorted(self.remove_index, reverse=True):
                del y_samples[i]
            y_data.extend([y_samples]*len(x_samples))
            x_data.extend(x_samples)

        y_data = np.array(y_data)
        x_data = np.array(x_data)
        
        shuffle_index = list(range(0, x_data.shape[0]))

        random.shuffle(shuffle_index)

        y_data = y_data[shuffle_index]
        x_data = x_data[shuffle_index]

        train_split_index = int(np.floor(x_data.shape[0]*train_test_split))

        x_train = x_data[:train_split_index]
        x_test = x_data[train_split_index:]
        y_train = y_data[:train_split_index]
        y_test = y_data[train_split_index:]

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        return x_train, y_train, x_test, y_test
        
    def sample_signal(self, file_path, sample_len = 100, n_samples = 10, signal_type = "All"):
        signal_list = []
        
        for count in range(n_samples):
            samples_offset = random.randint(0, self.file_samples - 1 - sample_len)
            samples = np.memmap(file_path, dtype = np.complex64, mode = "r", offset=samples_offset*np.dtype(np.complex64).itemsize, shape = (sample_len,))
            samples = np.array(samples)
            signal_list.append(samples.tolist())
        
        IQ_samples = np.array(signal_list)  # shape (N, 1024)
        if signal_type == "Complex":
            I_ = np.real(IQ_samples)
            I_ = np.expand_dims(I_, axis = 1)
            Q_ = np.imag(IQ_samples)
            Q_ = np.expand_dims(Q_, axis = 1)
            samples = np.concatenate([I_, Q_] ,axis = 1)
        elif signal_type == "Real":
            samples = np.real(IQ_samples)
        elif signal_type == "Imag":
            samples = np.imag(IQ_samples)
        elif signal_type == "Mag":
            samples = np.abs(IQ_samples).tolist()
        elif signal_type == "Phase":
            samples = np.arctan2(np.imag(IQ_samples)/np.real(IQ_samples))
        elif signal_type == "fft":
            window = np.hanning(IQ_samples.shape[0])
            windowed_samples= IQ_samples * window  
            spectrum = np.fft.fftshift(
                np.fft.fft(windowed_samples, n=self.FFT_length, axis=-1) 
                / self.FFT_length, axes=-1)
            Mag = np.abs(spectrum)
            Mag = np.expand_dims(Mag, axis = 1)
            Phase =  np.arctan2(np.imag(spectrum), np.real(spectrum))
            Phase = np.expand_dims(Phase, axis = 1)
            I_ = np.real(spectrum)
            I_ = np.expand_dims(I_, axis = 1)
            Q_ = np.imag(spectrum)
            Q_ = np.expand_dims(Q_, axis = 1)
            samples = np.concatenate([I_, Q_, Mag, Phase] ,axis = 1)   
 
        elif signal_type == "IQ_All":
            Mag = np.abs(IQ_samples)
            Mag = np.expand_dims(Mag, axis = 1)
            Phase =  np.arctan2(np.imag(IQ_samples), np.real(IQ_samples))
            Phase = np.expand_dims(Phase, axis = 1)
            I_ = np.real(IQ_samples)
            I_ = np.expand_dims(I_, axis = 1)
            Q_ = np.imag(IQ_samples)
            Q_ = np.expand_dims(Q_, axis = 1)
            samples = np.concatenate([I_, Q_, Mag, Phase] ,axis = 1)  

        elif signal_type == "All":
            signal_Mag = np.abs(IQ_samples)
            signal_Mag = np.expand_dims(signal_Mag, axis = 1)
            signal_Phase =  np.arctan2(np.imag(IQ_samples), np.real(IQ_samples))
            signal_Phase = np.expand_dims(signal_Phase, axis = 1)
            signal_I_ = np.real(IQ_samples)
            signal_I_ = np.expand_dims(signal_I_, axis = 1)
            signal_Q_ = np.imag(IQ_samples)
            signal_Q_ = np.expand_dims(signal_Q_, axis = 1)
            window = np.hanning(self.FFT_length)
            windowed_samples= IQ_samples * window  
            spectrum = np.fft.fftshift(
                np.fft.fft(windowed_samples, n=self.FFT_length, axis=-1) 
                / self.FFT_length, axes=-1)
            fft_Mag = np.abs(spectrum)
            fft_Mag = np.expand_dims(fft_Mag, axis = 1)
            fft_Phase =  np.arctan2(np.imag(spectrum), np.real(spectrum))
            fft_Phase = np.expand_dims(fft_Phase, axis = 1)
            fft_I_ = np.real(spectrum)
            fft_I_ = np.expand_dims(fft_I_, axis = 1)
            fft_Q_ = np.imag(spectrum)
            fft_Q_ = np.expand_dims(fft_Q_, axis = 1)

            samples = np.concatenate([signal_I_, signal_Q_, signal_Mag, signal_Phase, fft_I_, fft_Q_, fft_Mag, fft_Phase] ,axis = 1) 

        return samples
