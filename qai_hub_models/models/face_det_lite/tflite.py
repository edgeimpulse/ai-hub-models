import numpy as np
import tensorflow as tf
import torch
from face_det_lite import detect
import cv2
from PIL import Image
import json
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.models.face_det_lite.model import (MODEL_ASSET_VERSION, MODEL_ID)

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_640x480_Rooney.jpg"
)

def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | torch.Tensor | tuple[int, int],
    bottom_right: np.ndarray | torch.Tensor | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
):
    if not isinstance(top_left, tuple):
        top_left = (int(top_left[0].item()), int(top_left[1].item()))
    if not isinstance(bottom_right, tuple):
        bottom_right = (int(bottom_right[0].item()), int(bottom_right[1].item()))
    cv2.rectangle(frame, top_left, bottom_right, color, size)


def post_proc(dets, img):
    res = []
    for n in range(0, len(dets)):
        xmin, ymin, w, h = dets[n].xywh
        score = dets[n].score

        L = int(xmin)
        R = int(xmin + w)
        T = int(ymin)
        B = int(ymin + h)
        W = int(w)
        H = int(h)

        if L < 0 or T < 0 or R >= 640 or B >= 480:
            if L < 0:
                L = 0
            if T < 0:
                T = 0
            if R >= 640:
                R = 640 - 1
            if B >= 480:
                B = 480 - 1

        # Enlarge bounding box to cover more face area
        b_Left = L - int(W * 0.05)
        b_Top = T - int(H * 0.05)
        b_Width = int(W * 1.1)
        b_Height = int(H * 1.1)

        if (
            b_Left >= 0
            and b_Top >= 0
            and b_Width - 1 + b_Left < 640
            and b_Height - 1 + b_Top < 480
        ):
            L = b_Left
            T = b_Top
            W = b_Width
            H = b_Height
            R = W - 1 + L
            B = H - 1 + T

        res.append([L, T, W, H, score])
    # print(f"Bounding boxes: {res}")
    np_out = np.asarray(img)
    np_out = torch.tensor(np_out).byte().numpy()
    for box in dets:
        box = box.box
        draw_box_from_xyxy(
            np_out,
            torch.tensor(box[0:2]),
            torch.tensor(box[2:4]),
            color=(0, 255, 0),
            size=2,
        )
    out = Image.fromarray(np_out)
    return res, out


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="face_det_lite-lightweight-face-detection-float.tflite")
interpreter.allocate_tensors()

# Load input data from the npy file
# dataset = np.load('ei-fomo-face-detection-image-X_training.34.npy')
# dataset = np.load('473334_14_X_train_features-v9.npy')
img_array = np.asarray(load_image(INPUT_IMAGE_ADDRESS))
img_array = img_array.astype("float32") / 255.0
img_array = img_array[np.newaxis, ...]
dataset = np.expand_dims(img_array[:, :, :, -1], axis=-1)

for i in range(dataset.shape[0]):
    image_data = (dataset[i] * 255).astype(np.uint8).reshape((480, 640))

    input_data = dataset[i].reshape((1, 480, 640, 1))
    # input_data = input_data[:, :, :, -1]
    # input_data = input_data[..., np.newaxis]

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = [interpreter.get_tensor(od['index']) for od in output_details]

    # output details
    # for od in interpreter.get_output_details():
    #     print(f"Output {od['name']} shape: {od['shape']}, dtype: {od['dtype']}")

    hm = np.transpose(output_data[0], (0, 3, 1, 2))
    box = np.transpose(output_data[1], (0, 3, 1, 2))
    landmark = np.transpose(output_data[2], (0, 3, 1, 2))

    minimum_confidence_rating = 0.55
    # Print or process the output as needed
    bboxes = detect(hm, box, landmark, minimum_confidence_rating)
    print(f"Detected {len(bboxes)} faces")
    img = Image.fromarray(image_data, mode='L')

    res, out = post_proc(bboxes, img)
    # print(f"res: {res}")
    out_dict = []

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

    with open(f"tflite_output/output{i}.json", "w", encoding="utf-8") as wf:
        json.dump(out_dict, wf, ensure_ascii=False, indent=4)

    out.save(f"tflite_output/output{i}.png")

    torch_res = json.load(open(f"build/output{i}.json", "r"))

    # comparte torch_res and out_dict
    # for bbox, idx in zip(torch_res, range(len(torch_res))):
    #     print(f"Comparing bounding box {idx}")
    #     if bbox["L"] != out_dict[idx]["L"] or bbox["T"] != out_dict[idx]["T"] or bbox["W"] != out_dict[idx]["W"] or bbox["H"] != out_dict[idx]["H"]:
    #         print(f"Mismatch in bounding box {idx}: {bbox} vs {out_dict[idx]}")
    #     if not np.isclose(bbox["score"], out_dict[idx]["score"], atol=1e-6, rtol=1e-6):
    #         print(f"Score mismatch for bounding box {idx}: {bbox['score']} vs {out_dict[idx]['score']}")

    # print(f"Detected {len(bboxes)} faces")
    # print (f"Bounding boxes: {bboxes}")