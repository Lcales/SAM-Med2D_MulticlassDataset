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
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="validation_run", help="run model name")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="./liver_dataset/val", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")  
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save result")
    parser.add_argument("--log_file", type=str, default="workdir/validation.log", help="log file path")
    parser.add_argument("--epoch_model_dir", type=str, default="workdir/epoch_models", help="path to save epoch models")  # Linea aggiunta
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


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

    for i, batched_input in enumerate(test_loader):
        batched_input = to_device(batched_input, args.device)
        
        # Controlla le chiavi di batched_input
        print(f"Keys in batched_input: {batched_input.keys()}")
        
        # Verifica la presenza delle chiavi necessarie
        if 'name' in batched_input:
            img_name = batched_input['name'][0]
        else:
            img_name = f"image_{i}"  # Usa un nome generico se 'name' non è presente
            print(f"'name' key not found, using default: {img_name}")

        # Assicurati che le chiavi essenziali siano presenti
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]

        # Gestione del prompt se non specificato
        if args.prompt_path is None:
            prompt_dict = {}  # Assicurati che `prompt_dict` sia definito
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        # Elaborazione del modello con il blocco torch.no_grad()
        with torch.no_grad():
            # Codifica dell'immagine
            image_embeddings = model.image_encoder(batched_input["image"])
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(batched_input["point_coords"], batched_input["point_labels"]),
                boxes=batched_input.get("boxes", None),
                masks=None,
            )

            # Decodifica delle maschere
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        # Post-elaborazione delle maschere
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

        # Salvataggio delle predizioni, se richiesto
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        # Calcolo della loss
        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        # Calcolo delle metriche per il batch
        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]

    # Calcolo delle metriche medie per l'intero set di validazione
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: f'{test_iter_metrics[i]:.4f}' for i in range(len(test_iter_metrics))}
    avg_loss = np.mean(test_loss)

    # Log dei risultati finali
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
