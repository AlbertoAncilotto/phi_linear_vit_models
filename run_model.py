from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort
import numpy as np
import os
import cv2
import json
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image


onnx_model_path = 'model_2M.onnx'
coco_annotation_file = 'datasets/coco/annotations/instances_val2017.json'
coco_image_dir = 'datasets/coco/images/val2017/'
output_file = 'predictions.json'
VISUALIZE = False  # Set to True to visualize predictions

session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

coco = COCO(coco_annotation_file)
cats = coco.loadCats(coco.getCatIds())
cat_dict = {ca['name']: ca['id'] for ca in cats}
mscoco_label2lcategory = {i: k for i, k in enumerate(cat_dict.keys())}
image_ids = coco.getImgIds()

predictions = []
for img_id in tqdm(image_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(coco_image_dir, img_info['file_name'])
    
    if img_path.endswith('.jpg'):
        # Preprocessing with cv2
        # img = cv2.imread(img_path)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # img_resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0  # Resize and normalize
        # img_resized = np.transpose(img_resized, (2, 0, 1))  # Change to (C, H, W)
        # img_input = np.expand_dims(img_resized, axis=0)  # Add batch dimension

        im_pil = Image.open(img_path).convert('RGB')
        w,h = im_pil.size
        orig_size = torch.tensor([w, h])[None]
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        img_input = transforms(im_pil)[None]
        img_cv = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        input_name = session.get_inputs()[0].name
        input_size = session.get_inputs()[1].name
        outputs = session.run(None, {input_name: img_input.numpy(), input_size: np.array([[w, h]])}) 
        labels, boxes, scores = outputs[0][0], outputs[1][0], outputs[2][0]
        
        for i in range(len(scores)):
            score = scores[i]
            if score < 0.001:  # Filter low-confidence detections
                continue

            x1, y1, x2, y2 = boxes[i]
            category_id = cat_dict[mscoco_label2lcategory[labels[i]]]  

            prediction = {
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            }
            predictions.append(prediction)

            if VISUALIZE:
                cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                label = f'{category_id} : {score:.2f}'
                cv2.putText(img_cv, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if VISUALIZE:
            cv2.imwrite('pred.jpg', img_cv)
            breakpoint()


with open(output_file, 'w') as f:
    json.dump(predictions, f)

coco_pred = coco.loadRes(output_file)
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
