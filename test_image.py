# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import logging

import ssrgan.models as models
from ssrgan.utils.common import create_folder
from tester import Estimate

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research and application of GAN based super resolution "
                                                 "technology for pathological microscopic images.")
    # basic parameters
    parser.add_argument("--lr", type=str, required=True,
                        help="Test low resolution image name.")
    parser.add_argument("--hr", type=str, required=True,
                        help="Raw high resolution image name.")
    parser.add_argument("--outf", default="test", type=str, metavar="PATH",
                        help="The location of the image in the evaluation process. (default: ``test``).")
    parser.add_argument("--device", default="cpu",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``cpu``).")
    parser.add_argument("--detail", dest="detail", action="store_true",
                        help="Use comprehensive assessment.")

    # model parameters
    parser.add_argument("-a", "--arch", metavar="ARCH", default="bionet",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: bionet)")
    parser.add_argument("--upscale-factor", type=int, default=4, choices=[4],
                        help="Low to high resolution scaling factor. (default:4).")
    parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                        help="Path to latest checkpoint for model. (default: ````).")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")

    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")
    print(args)

    create_folder(args.outf)
    detail = True if args.detail else False

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.1.1")
    print("\tBuild ................ 2020.11.30-1116-0c5adc7e")

    logger.info("Creating Testing Engine")
    estimate = Estimate(args)

    logger.info("Staring testing model")
    estimate.run()
    print("##################################################\n")

    logger.info("Test single image performance evaluation completed successfully.\n")
