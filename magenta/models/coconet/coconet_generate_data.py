# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from magenta.models.coconet import lib_data
from magenta.models.coconet import lib_pianoroll
from magenta.models.coconet import lib_evaluation
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_util
import numpy as np
import tensorflow as tf
import pretty_midi

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('prime_midi_melody_fpath', None,
                    'Path to the base directory for different datasets.')
flags.DEFINE_integer("gen_batch_size", 3,
                     "Num of samples to generate in a batch.")                     
flags.DEFINE_string('data_npz_out', None,
                    'Path to the base directory for different datasets.')
flags.DEFINE_float('quantization_level', 0.125, 'Quantization duration.'
                   'For qpm=120, notated quarter note equals 0.5.')

flags.DEFINE_integer('num_instruments', 4,
                     'Maximum number of instruments that appear in this '
                     'dataset.  Use 0 if not separating instruments and '
                     'hence does not matter how many there are.')
flags.DEFINE_bool('separate_instruments', True,
                  'Separate instruments into different input feature'
                  'maps or not.')
flags.DEFINE_integer('crop_piece_len', 64, 'The number of time steps '
                     'included in a crop')

def pianoroll_shape(separate_instruments, crop_piece_len, num_pitches, num_instruments):
    if separate_instruments:
      return [crop_piece_len, num_pitches, num_instruments]
    else:
      return [crop_piece_len, num_pitches, 1]

def main(noargs):
    min_pitch = 36
    max_pitch = 81
    num_pitches = (max_pitch - min_pitch) + 1
    shortest_duration = 0.125
    num_instruments = 4
    qpm = 60
    pianoroll_shape_arr = pianoroll_shape(FLAGS.separate_instruments, FLAGS.crop_piece_len, num_pitches, num_instruments)
    shape = [FLAGS.gen_batch_size] + pianoroll_shape_arr
    np.set_printoptions(threshold=sys.maxsize)
    encoder = lib_pianoroll.PianorollEncoderDecoder(
      shortest_duration=shortest_duration,
      min_pitch=min_pitch,
      max_pitch=max_pitch,
      separate_instruments=FLAGS.separate_instruments,
      num_instruments=num_instruments,
      quantization_level=FLAGS.quantization_level)

    train = []
    test = []
    valid = []
    for file_index, file in enumerate(os.listdir(FLAGS.prime_midi_melody_fpath)):
        if file.endswith(".mid"):
            midi_in = pretty_midi.PrettyMIDI(os.path.join(FLAGS.prime_midi_melody_fpath, file))
            if len(midi_in.instruments) != 4:
                tf.logging.info('incorrect number of instruments ,'+os.path.join(FLAGS.prime_midi_melody_fpath, file)+' skipping for now')
                continue
            try:
                data = encoder.encode_midi_to_pianoroll(midi_in, shape)
            except ValueError:
                tf.logging.info('incorrect pitch information ,'+os.path.join(FLAGS.prime_midi_melody_fpath, file)+' skipping for now')
                continue

            roll = data.tolist()[0]

            npz_data=[]
            for t, arr in enumerate(roll):
                data = [min_pitch] * num_instruments
                for pitch, pitch_arr in enumerate(arr):
                    for index,instrument in enumerate(pitch_arr):
                        if instrument == 1:
                            data[index] += pitch
                            if data[index] > max_pitch:
                                data[index] = max_pitch
                                print('invalid data')
                            continue
                if data != [min_pitch] * num_instruments:
                    npz_data.append(data)
            if file_index % 3 == 0:
                train.append(npz_data)
            elif file_index % 3 == 1:
                test.append(npz_data)
            else:
                valid.append(npz_data)
    np.savez(FLAGS.data_npz_out, train=np.array(train), test=np.array(test), valid=np.array(valid))
    
if __name__ == "__main__":
  tf.app.run()