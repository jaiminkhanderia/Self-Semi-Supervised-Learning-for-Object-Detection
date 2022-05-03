from pathlib import Path
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T

from dataset import LabeledDataset
from engine import evaluate
from utils import collate_fn


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model", add_help=False)

    # Checkpoints
    parser.add_argument("--ckpt-dir", type=Path, required=True,
                        help='Path to the checkpoint folder')

    # Evaluate params
    parser.add_argument("--batch-size", type=int, default=8,
                        help='Batch size')

    # Running
    parser.add_argument("--num-workers", type=int, default=8)

    return parser


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model(args, num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(args.ckpt_dir / 'model.pth', map_location=device)
    for key in list(checkpoint["model"].keys()):
        if "module" in key:
            new_key = key.split("module.")[1]
            checkpoint["model"][new_key] = checkpoint["model"][key]
            del checkpoint["model"][key]
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def main(args):
    print(args)
    gpu = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 101

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    model = get_model(args, num_classes, gpu).cuda(gpu)
    evaluate(model, valid_loader, device=gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)