import os
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_logger
from test import (
    sam_model_registry,
    FocalDiceloss_IoULoss,
    TestingDataset,
    to_device,
    postprocess_masks,
    prompt_and_decoder,
    generate_point,
    save_masks,
    SegMetrics,
    parse_args
)


def main():
    args = parse_args()  # Usa lo stesso parser di test.py
    logger = get_logger(os.path.join(args.work_dir, "validation.log"))

    # Otteniamo i checkpoint
    checkpoint_dir = args.epoch_model_dir
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith("_sam.pth")],
        key=lambda x: int(x.split("epoch")[1].split("_")[0])
    )

    # Costruiamo il DataLoader per il validation set
    test_dataset = TestingDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        mode="test",
        requires_name=True,
        point_num=args.point_num,
        return_ori_mask=True,
        prompt_path=args.prompt_path,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    epoch = 0

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        epoch = epoch + 1

        # Carichiamo il modello
        model = sam_model_registry[args.model_type](args).to(args.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        model.eval()

        # Inizializziamo le metriche
        criterion = FocalDiceloss_IoULoss()
        test_loss = []
        test_iter_metrics = [0] * len(args.metrics)
        test_metrics = {}
        prompt_dict = {}

        test_pbar = tqdm(test_loader)
        for i, batched_input in enumerate(test_pbar):
            batched_input = to_device(batched_input, args.device)
            ori_labels = batched_input["ori_label"]
            original_size = batched_input["original_size"]
            labels = batched_input["label"]
            img_name = batched_input["name"][0]

            if args.prompt_path is None:
                prompt_dict[img_name] = {
                    "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                    "point_coords": batched_input["point_coords"]
                    .squeeze(1)
                    .cpu()
                    .numpy()
                    .tolist(),
                    "point_labels": batched_input["point_labels"]
                    .squeeze(1)
                    .cpu()
                    .numpy()
                    .tolist(),
                }

            with torch.no_grad():
                image_embeddings = model.image_encoder(batched_input["image"])

            if args.boxes_prompt:
                save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
                batched_input["point_coords"], batched_input["point_labels"] = None, None
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings
                )
                points_show = None

            else:
                save_path = os.path.join(
                    args.work_dir,
                    args.run_name,
                    f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt",
                )
                batched_input["boxes"] = None
                point_coords, point_labels = [batched_input["point_coords"]], [
                    batched_input["point_labels"]
                ]

                for iter in range(args.iter_point):
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(
                        args, batched_input, model, image_embeddings
                    )
                    if iter != args.iter_point - 1:
                        batched_input = generate_point(
                            masks, labels, low_res_masks, batched_input, args.point_num
                        )
                        batched_input = to_device(batched_input, args.device)
                        point_coords.append(batched_input["point_coords"])
                        point_labels.append(batched_input["point_labels"])
                        batched_input["point_coords"] = torch.concat(
                            point_coords, dim=1
                        )
                        batched_input["point_labels"] = torch.concat(
                            point_labels, dim=1
                        )

                points_show = (
                    torch.concat(point_coords, dim=1),
                    torch.concat(point_labels, dim=1),
                )

            masks, pad = postprocess_masks(
                low_res_masks, args.image_size, original_size
            )
            if args.save_pred:
                save_masks(
                    masks,
                    save_path,
                    img_name,
                    args.image_size,
                    original_size,
                    pad,
                    batched_input.get("boxes", None),
                    points_show,
                )

            loss = criterion(masks, ori_labels, iou_predictions)
            test_loss.append(loss.item())

            test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
            test_batch_metrics = [
                float("{:.4f}".format(metric)) for metric in test_batch_metrics
            ]

            for j in range(len(args.metrics)):
                test_iter_metrics[j] += test_batch_metrics[j]

        test_iter_metrics = [metric / len(test_loader) for metric in test_iter_metrics]
        test_metrics = {
            args.metrics[i]: "{:.4f}".format(test_iter_metrics[i])
            for i in range(len(test_iter_metrics))
        }

        average_loss = np.mean(test_loss)
        logger.info(f"epoch: {epoch + 1}, lr: {args.lr if 'lr' in vars(args) else 'N/A'}, "
                    f"Validation loss: {average_loss:.4f}, metrics: {test_metrics}")
    

        if args.prompt_path is None:
            with open(
                os.path.join(args.work_dir, f"{args.image_size}_prompt.json"), "w"
            ) as f:
                json.dump(prompt_dict, f, indent=2)



if __name__ == "__main__":
    main()
