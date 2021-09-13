# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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

# ==============================================================================
# File description: Realize the verification function after model training.
# ==============================================================================
import shutil

import torchvision.utils

from config import *


def main() -> None:
    # Create a experiment result folder.
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    # Load model weights.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    with torch.no_grad():
        for index in range(100):
            # Create an image that conforms to the Gaussian distribution.
            fixed_noise = torch.randn([64, 100], device=device)
            fixed_noise = fixed_noise.half()
            image = model(fixed_noise)
            torchvision.utils.save_image(image, os.path.join(exp_dir, f"{index:03d}.bmp"))
            print(f"The {index + 1:03d} image is being created using the model...")


if __name__ == "__main__":
    main()
