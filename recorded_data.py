# !/usr/bin/env python
# !/usr/local/bin/python
import os
import re
import sys
import wave
from enum import Enum

import numpy
import numpy as np
import skimage.io  # scikit-image

try:
    import librosa
except:
    print('pip install librosa')

from random import shuffle

try:
    from six.moves import urllib
    from six.moves import xrange
except:
    pass

# TODO contribute
# TODO add data
SOURCE_URL = 'http://mycloud.ch/'  # recordings
DATA_DIR = 'data/'
path = 'data/WAV/F_Short/'
CHUNK = 4096
width = 64
height = 64


class Source:
    DIGIT_WAVES = 'recorded_data.tar'


class Target(Enum):
    digits = 1
    speaker = 2


def progresshook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = '\r%5.1f%% %*d / %d' % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write('\n')
        else:  # total size is unknown
            sys.stderr.write('read %d\n' % (readsofar,))


def maybe_download(file, work_directory=DATA_DIR):
    '''Download the data'''
    print('Looking for data %s in %s' % (file, work_directory))
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, re.sub('.*\/', '', file))
    if not os.path.exists(filepath):
        if not file.startswith('http'):
            url_filename = SOURCE_URL + file
        else:
            url_filename = file
        print('Downloading from %s to %s' % (url_filename, filepath))
        filepath, _ = urllib.request.urlretrieve(url_filename, filepath, progresshook)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
        # os.system('ln -s '+work_directory)
    if os.path.exists(filepath):
        print('Extracting %s to %s' % (filepath, work_directory))
        os.system('tar xf ' + filepath + ' -C ' + work_directory)
        print('Data ready!')
    return filepath.replace('.tar', '')


def speaker(filename):  # filename
    if not '-' in filename:
       return 'Unknown'
    return filename.split('-')[1]


def get_speakers(path=path):
    # maybe_download(Source)
    files = os.listdir(path)

    def nobad(name):
        return '-' in name and not '%' in name.split('-')[1]

    speakers = list(set(map(speaker, filter(nobad, files))))
    return speakers


def load_mcff_file(name):
    if not name.endswith('.wav'):
        return None
    wave, sr = librosa.load(path + name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    mfcc_zero = len(mfcc[0])
    # ATTENTION WITH LARGE FILES
    mfcc = np.pad(mfcc, ((0, 0), (0, 80 - mfcc_zero)), mode='constant', constant_values=0)
    return mfcc


def load_wav_file(name):
    f = wave.open(name, 'rb')
    # print('loading %s'%name)
    chunk = []
    data0 = f.readframes(CHUNK)
    while data0:  # f.getnframes()
        # data=numpy.fromstring(data0, dtype='float32')
        # data = numpy.fromstring(data0, dtype='uint16')
        data = numpy.fromstring(data0, dtype='uint8')
        data = (data + 128) / 255.  # 0-1 for Better convergence
        # chunks.append(data)
        chunk.extend(data)
        data0 = f.readframes(CHUNK)
    # finally trim:
    chunk = chunk[0:CHUNK * 4]  # should be enough for now -> cut
    chunk.extend(numpy.zeros(CHUNK * 4 - len(chunk)))  # fill with padding 0's
    # print('%s loaded'%name)
    return chunk


def mfcc_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):
    # maybe_download(source, DATA_DIR)
    if target == Target.speaker:
        speakers = get_speakers()
    batch_features = []
    labels = []
    files = os.listdir(path)
    while True:
        print('loaded batch of %d files' % len(files))
        shuffle(files)
        for file in files:
            if not file.endswith('.wav'):
                continue
            wave, sr = librosa.load(path + file, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc_zero = len(mfcc[0])
            if target == Target.speaker:
                label = one_hot_from_item(speaker(file), speakers)
            # label = file  # sparse_labels(file, pad_to=20)  # max_output_length
            else:
                raise Exception('add defined target')
            labels.append(label)
            # print(np.array(mfcc).shape())
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - mfcc_zero)), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch
                labels = []


# If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s
def wave_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):  # speaker
    # maybe_download(source, DATA_DIR)
    if target == Target.speaker:
        speakers = get_speakers()
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # big!!
    files = os.listdir(path)
    while True:
        shuffle(files)
        print('loaded batch of %d files' % len(files))
        for wav in files:
            if not wav.endswith('.wav'):
                continue
            if target == Target.speaker:
                labels.append(one_hot_from_item(speaker(wav), speakers))
            else:
                raise Exception('add defined target')
            chunk = load_wav_file(path + wav)
            batch_waves.append(chunk)
            # batch_waves.append(chunks[input_width])
            if len(batch_waves) >= batch_size:
                yield batch_waves, labels
                batch_waves = []  # Reset for next batch
                labels = []


class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False, load=False):
        '''Construct a DataSet. one_hot arg is used only if fake_data is true.'''
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            num = len(images)
            assert num == len(labels), ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            print('len(images) %d' % num)
            self._num_examples = num
        self.cache = {}
        self._image_names = numpy.array(images)
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._images = []
        if load:  # Otherwise loaded on demand
            self._images = self.load(self._image_names)

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    # only apply to a subset of all images at one time
    def load(self, image_names):
        print('loading %d images' % len(image_names))
        return list(map(self.load_image, image_names))  # python3 map object WTF

    def load_image(self, image_name):
        if image_name in self.cache:
            return self.cache[image_name]
        else:
            image = skimage.io.imread(DATA_DIR + image_name).astype(numpy.float32)
            # images = numpy.multiply(images, 1.0 / 255.0)
            self.cache[image_name] = image
            return image

    def next_batch(self, batch_size, fake_data=False):
        # Returns the 'batch_size' examples
        if fake_data:
            fake_image = [1] * width * height
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._image_names = self._image_names[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.load(self._image_names[start:end]), self._labels[start:end]


# multi-label
def dense_to_some_hot(labels_dense, num_classes=140):
    # TODO
    raise 'TODO dense_to_some_hot'


def one_hot_to_items(hot, items):
    items_result = {}
    rank = 0
    w = np.argsort(hot)
    for x in np.nditer(w):
        item = items[x]
        rank += 1
        items_result.update({rank: item})
    return items_result


def one_hot_to_item(hot, items):
    i = np.argmax(hot)
    item = items[i]
    return item


def one_hot_from_item(item, items):
    x = [0] * len(items)  # numpy.zeros(len(items))
    i = items.index(item)
    x[i] = 1
    return x


if __name__ == '__main__':
    print('loaded')
