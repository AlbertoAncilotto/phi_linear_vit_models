import argparse
import onnxruntime as ort
import numpy as np
import os, json, time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
from onnx_opcounter import calculate_params
import onnx

def main():
    parser = argparse.ArgumentParser(description='Run ONNX model on COCO dataset with configurable paths.')
    parser.add_argument('--model-path', type=str, default='models/phi_det_2M.onnx', help='Path to the ONNX model')
    parser.add_argument('--annotation-path', type=str, default='datasets/coco/annotations/instances_val2017.json', help='Path to COCO annotation file')
    parser.add_argument('--image-dir', type=str, default='datasets/coco/images/val2017/', help='Path to directory with COCO images')
    args = parser.parse_args()

    output_file = 'predictions.json'

    session = ort.InferenceSession(args.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    session.run(None, {session.get_inputs()[0].name: np.random.rand(1, 3, 640, 640).astype(np.float32), session.get_inputs()[1].name: np.array([[620, 640]])}) # Warmup
    coco = COCO(args.annotation_path)
    cats = coco.loadCats(coco.getCatIds())
    cat_dict = {ca['name']: ca['id'] for ca in cats}
    mscoco_label2lcategory = {i: k for i, k in enumerate(cat_dict.keys())}
    image_ids = coco.getImgIds()

    predictions = []
    total_time = 0
    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_info['file_name'])

        img = Image.open(img_path).convert('RGB')
        orig_size = torch.tensor(img.size)[None]
        img_input = T.Compose([T.Resize((640, 640)), T.ToTensor()])(img)[None].numpy()

        start = time.time()
        outputs = session.run(None, {session.get_inputs()[0].name: img_input, session.get_inputs()[1].name: orig_size.numpy()})
        total_time += time.time() - start

        labels, boxes, scores = outputs[0][0], outputs[1][0], outputs[2][0]
        for i, score in enumerate(scores):
            if score >= 0.001:
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": img_id,
                    "category_id": cat_dict[mscoco_label2lcategory[labels[i]]],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score)
                })

    with open(output_file, 'w') as f: json.dump(predictions, f)
    coco_eval = COCOeval(coco, coco.loadRes(output_file), 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    model = onnx.load_model(args.model_path)
    params = calculate_params(model)
    print('Number of params:', params)
    print(f"Average latency: {total_time / len(image_ids):.4f} seconds")

if __name__ == "__main__":
    main()
