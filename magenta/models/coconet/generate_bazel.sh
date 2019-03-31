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

#!/bin/bash

set -x
set -e

# Change this to dir for saving experiment logs.
logdir=$HOME/logs
# Change this to where data is loaded from.
# data_dir=testdata/
# data_dir=$HOME/data/
data_npz_out=$HOME/data/jsbfull/Jsb16thSeparated.npz
# Change this to your dataset class, which can be defined in lib_data.py.
# dataset=TestData
dataset=Jsb16thSeparated

prime_midi_melody_fpath=jsbach_midi

# Data preprocessing.
crop_piece_len=32
separate_instruments=True
quantization_level=0.125  # 16th notes

# Hyperparameters.
maskout_method=orderless
num_layers=32
num_filters=64
batch_size=10
use_sep_conv=True
architecture='dilated'
num_dilation_blocks=1
dilate_time_only=False
repeat_last_dilation_level=False
num_pointwise_splits=2
interleave_split_every_n_layers=2


# Run command.
python coconet_generate_data.py \
  --logdir=$logdir \
  --log_process=True \
  --data_npz_out=$data_npz_out \
  --dataset=$dataset \
  --crop_piece_len=$crop_piece_len \
  --separate_instruments=$separate_instruments \
  --quantization_level=$quantization_level \
  --prime_midi_melody_fpath=$prime_midi_melody_fpath 