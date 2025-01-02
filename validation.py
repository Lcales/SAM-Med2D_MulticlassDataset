import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
from utils import get_logger, FocalDiceloss_IoULoss, generate_point, save_masks
from DataLoader import TestingDataset
from metrics import SegMetrics
from segment_anything import sam_model_registry
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="Work directory")
    parser.add_argument("--run_name", type=str, default="validation", help="Run name for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for input")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="Metrics for evaluation")
    parser.add_argument("--model_type", type=str, default="vit_b", help="Model architecture")
    parser.add_argument("--epoch_model_dir", type=str, required=True, help="Directory containing epoch models")
    parser.add_argument("--log_file", type=str, default=None, help="Path to save validation logs")
    parser.add_argument("--save_pred", type=bool, default=False, help="Whether to save prediction masks")
    args = parser.parse_args()
    return args

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif isinstance(value, (list, torch.Size)):
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def evaluate_model(args, model, test_loader, logger=None):
    criterion = FocalDiceloss_IoULoss()
    model.eval()
    test_pbar = tqdm(test_loader, desc="Validating")
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    l = len(test_loader)

    for batched_input in test_pbar:
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        labels = batched_input["label"]

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(batched_input["point_coords"], batched_input["point_labels"]),
                boxes=batched_input.get("boxes", None),
                masks=None,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        masks, _ = postprocess_masks(low_res_masks, args.image_size, batched_input["original_size"])
        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]

    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: f'{test_iter_metrics[i]:.4f}' for i in range(len(test_iter_metrics))}
    avg_loss = np.mean(test_loss)

    if logger:
        logger.info(f"Validation Loss: {avg_loss:.4f}, Metrics: {test_metrics}")
    print(f"Validation Loss: {avg_loss:.4f}, Metrics: {test_metrics}")

    return avg_loss, test_metrics

def main(args):
    print('*' * 80)
    print("Validation Run Configurations:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('*' * 80)

    # Initialize Logger
    logger = None
    if args.log_file:
        logger = get_logger(args.log_file)

    # Initialize Model
    model = sam_model_registry[args.model_type](args).to(args.device)

    # Load Validation Dataset
    val_dataset = TestingDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        mode='test',
        requires_name=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Iterate through epoch models
    checkpoint_files = sorted(
        [f for f in os.listdir(args.epoch_model_dir) if f.endswith("_sam.pth")],
        key=lambda x: int(x.split('epoch')[1].split('_')[0])
    )
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(args.epoch_model_dir, checkpoint_file)
        print(f"Evaluating model: {checkpoint_file}")

        # Load Model Weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        
        # Evaluate Model
        evaluate_model(args, model, val_loader, logger)

if __name__ == "__main__":
    args = parse_args()
    main(args)
