# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import hashlib
import os

import torch


def cal_file_md5(filename):
    """ Calculates the MD5 value of the file
    Args:
        filename: The path name of the file.

    Return:
        The MD5 value of the file.

    """
    with open(filename, "rb") as f:
        md5 = hashlib.md5()
        md5.update(f.read())
        hash_value = md5.hexdigest()
    return hash_value


def compress_model(state, filename, model_arch):
    model_folder = "../checkpoints"
    try:
        os.makedirs(model_folder)
    except OSError:
        pass

    new_filename = model_arch + "-" + cal_file_md5(filename)[:8] + ".pth"
    torch.save(state, os.path.join(model_folder, new_filename))
