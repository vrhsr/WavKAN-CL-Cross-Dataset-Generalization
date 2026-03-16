"""Generate prediction probabilities for uncertainty/calibration evaluation."""
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN


def build_model(model_name, device):
    if model_name == 'wavkan':
        return WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=64).to(device)
    if model_name == 'resnet':
        return ResNet1D(in_channels=1, num_classes=2).to(device)
    if model_name == 'vit':
        return ViT1D(seq_len=250, num_classes=2).to(device)
    if model_name == 'spline_kan':
        return SplineKANClassifier(input_dim=250, num_classes=2).to(device)
    if model_name == 'mlp':
        return SimpleMLP(input_dim=250, num_classes=2).to(device)
    if model_name == 'dann':
        return DANN(in_channels=1, num_classes=2, feature_dim=256).to(device)
    raise ValueError(f'Unknown model: {model_name}')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HarmonizedDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, device)
    ckpt = args.checkpoint or f'experiments/{args.model}_endpoint.pth'
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.numpy()
            logits = model.predict(x) if isinstance(model, DANN) else model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            for yt, yp, pr in zip(y, probs, preds):
                rows.append({'y_true': int(yt), 'y_prob': float(yp), 'y_pred': int(pr)})

    out = pd.DataFrame(rows)
    out_path = args.out_file or 'experiments/predictions_eval.csv'
    out.to_csv(out_path, index=False)

    y_true = out['y_true'].values
    y_prob = out['y_prob'].values
    y_pred = out['y_pred'].values
    print(f'Saved predictions to {out_path}')
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"AUROC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"AUPRC: {average_precision_score(y_true, y_prob):.4f}")
    print(f"Brier: {brier_score_loss(y_true, y_prob):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp', 'dann'])
    parser.add_argument('--data_file', default='data/ptbxl_processed.csv')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out_file', default='experiments/predictions_eval.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    main(parser.parse_args())
