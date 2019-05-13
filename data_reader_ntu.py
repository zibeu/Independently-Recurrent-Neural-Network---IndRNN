import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import os

NPY_DIR = 'nturgb_npy'
TRAIN_DS = '_train.csv'
TEST_DS = '_test.csv'

def read_test_dataset():
    samples = []
    ds_file_name = os.path.join(NPY_DIR, TEST_DS)
    with open(ds_file_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def read_dataset():
    samples = []
    ds_file_name = os.path.join(NPY_DIR, TRAIN_DS)
    with open(ds_file_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def generator(samples, batch_size=32, sh=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        if sh:
            shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            clips = []
            padded_clips = []
            labels = []
            sequence_len = 0
            for batch_sample in batch_samples:
                [name, label_str] = batch_sample
                clip = np.load(os.path.join(NPY_DIR, name))
                clip_len = clip.shape[0]

                if clip_len > sequence_len:
                    sequence_len = clip_len
                clips.append(clip)
                labels.append(int(label_str))

            for clip in clips:
                clip_len = clip.shape[0]
                pad_len = sequence_len - clip_len
                clip = np.pad(clip, ((0, pad_len), (0,0), (0,0)), mode='constant')
                padded_clips.append(clip)

            features = np.stack(padded_clips)
            if sh:
                yield shuffle(features, labels)
            else:
                yield features, labels


class DataHandler(object):
    def __init__(self, samples, batch_size, seq_len, shuffle):
        self.generator = generator(
                samples, 
                batch_size=batch_size,
                sh=shuffle,
                )
        self.size = len(samples)
    def GetBatch(self):
        return next(self.generator)
    def GetDatasetSize(self):
        return self.size



def initialize_data_handlers(batch_size, seq_len):
    train_samples, eval_samples = read_dataset()
    test_samples = read_test_dataset()
    dh_train = DataHandler(
            train_samples, 
            batch_size, 
            seq_len, 
            True,
            )
    dh_eval = DataHandler(
            eval_samples, 
            batch_size, 
            seq_len,
            False,
            )
    dh_test = DataHandler(
            test_samples, 
            batch_size, 
            seq_len,
            False,
            )
    return dh_train, dh_eval, dh_test
