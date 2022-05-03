from pathlib import Path
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T

from dataset import LabeledDataset
from engine import evaluate
from utils import init_distributed_mode, collate_fn


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
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model(args, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(args.ckpt_dir / 'model.pth')
    for key in list(checkpoint["model"].keys()):
        if "module" in key:
            new_key = key.split("module.")[1]
            checkpoint["model"][new_key] = checkpoint["model"][key]
            del checkpoint["model"][key]
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    num_classes = 101
    per_device_batch_size = args.batch_size // args.world_size

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=per_device_batch_size, num_workers=args.num_workers, 
                                               pin_memory=True, sampler=valid_sampler, collate_fn=collate_fn)
    
    model = get_model(args, num_classes).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    evaluate(model, valid_loader, device=gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)