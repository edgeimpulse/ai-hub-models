# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_640x480_Rooney.jpg"
)

def run_inference_on_image(
    model: FaceDetLite,
    orig_image: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
    is_test: bool,
    i: int,
    args
):
    app = FaceDetLiteApp(model)
    res, out = app.run_inference_on_image(orig_image, i)
    out_dict = []

    # out_dict["bounding box"] = str(res)
    for bbox in res:
        out_dict.append(
            {
                "L": bbox[0],
                "T": bbox[1],
                "W": bbox[2],
                "H": bbox[3],
                "score": float(bbox[4]),
            }
        )
    # print(f"out_dict: {out_dict}")
    if not is_test:
        output_path = (
            args.output_dir or str(Path() / "raw_data")
        ) + f"/output_{i}.json"

        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(out_dict, wf, ensure_ascii=False, indent=4)
        # display_or_save_image(out, "build", f"output{i}.png")
        # print(f"Model outputs are saved at: {output_path}")

# Run face_det_lite model end-to-end on a sample image.
# The demo will output the face bounding boxes in json files
# the bounding box represented by left, top, width, and height.
def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(FaceDetLite)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(FaceDetLite, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    use_ei_data = True

    if not use_ei_data:
        # use default demo image
        orig_image = load_image(args.image)
        run_inference_on_image(model, orig_image, is_test, 0, args)
    else:
        # For testing, we can also load a dataset of images
        dataset = np.load('ei-fomo-face-detection-image-X_training.34.npy')
        for i in range(dataset.shape[0]):
            # for i in range(0, 1):  # For testing, only process the first image
            print(f"Processing image {i + 1}")
            orig_image = dataset[i]
            orig_image = (orig_image * 255).astype(np.uint8).reshape((480, 640, 3))
            orig_image = Image.fromarray(orig_image)
            run_inference_on_image(model, orig_image, is_test, i, args)

if __name__ == "__main__":
    main()
