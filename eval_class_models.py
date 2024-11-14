import argparse
import math
import numpy as np
import torch
import torch.utils.data
import onnxruntime as ort
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from onnx_opcounter import calculate_params
import onnx


def accuracy(output: np.ndarray, target: torch.Tensor) -> float:
    pred = output.argmax(axis=1)
    correct = (pred == target.cpu().numpy()).sum()
    return correct / target.size(0) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/dataset/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=288)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()
    
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.path,
            transforms.Compose([
                    transforms.Resize(int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    session = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    top1_correct = 0
    total_samples = 0

    with tqdm(total=len(data_loader), desc=f"Evaluating {args.model} on ImageNet") as t:
        for images, labels in data_loader:
            images_np = images.numpy()
            outputs = session.run(None, {session.get_inputs()[0].name: images_np})[0]
            batch_top1 = accuracy(outputs, labels)
            top1_correct += batch_top1 * images.size(0)
            total_samples += images.size(0)
            avg_top1 = top1_correct / total_samples
            t.set_postfix({"top1_avg": avg_top1})
            t.update(1)

    
    model = onnx.load_model(args.model)
    params = calculate_params(model)
    print('Number of params:', params)
    print(f"Top-1 Accuracy: {avg_top1:.2f}%")


if __name__ == "__main__":
    main()
