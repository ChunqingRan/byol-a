import os
import re
import json
import time
import random
import datetime
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchaudio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(description="ShipsEar WOA contrastive training and evaluation")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"C:/Users/user/Desktop/PythonProject/Data/ShipsEar_16k_30s_hop15_WOA_FullAugmented_slimming/woa_augmentation_log.csv",
        help="Path to woa_augmentation_log.csv",
    )
    parser.add_argument("--base_dir", type=str, default="", help="Optional base directory for relative paths in CSV")
    parser.add_argument("--out_dir", type=str, default="", help="Optional output folder")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--eval_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--target_len", type=int, default=938)
    parser.add_argument("--fft_high", type=int, default=4096)
    parser.add_argument("--hop_high", type=int, default=512)
    parser.add_argument("--mels_high", type=int, default=128)
    parser.add_argument("--fft_low", type=int, default=512)
    parser.add_argument("--hop_low", type=int, default=512)
    parser.add_argument("--mels_low", type=int, default=64)
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Optional comma-separated class names, e.g. ClassA,ClassB,ClassC,ClassD,ClassE",
    )
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Logger:
    def __init__(self, filename):
        self.log = open(filename, "a", encoding="utf-8")

    def print(self, message):
        msg = f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] {message}"
        print(msg)
        self.log.write(msg + "\n")
        self.log.flush()

    def close(self):
        self.log.close()


def _find_column(df, candidates):
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    return None


def _normalize_path_str(path_str):
    if pd.isna(path_str):
        return ""
    p = str(path_str).strip().replace("\\", "/")
    return p


def _resolve_audio_path(path_str, csv_dir, base_dir=""):
    p = _normalize_path_str(path_str)
    if not p:
        return ""

    if os.path.isabs(p) and os.path.exists(p):
        return p

    if re.match(r"^[A-Za-z]:/", p):
        if os.path.exists(p):
            return p
        p_wo_drive = re.sub(r"^[A-Za-z]:", "", p)
        p_wo_drive = p_wo_drive.lstrip("/")
        for root in [base_dir, csv_dir]:
            if root:
                cand = os.path.join(root, p_wo_drive)
                if os.path.exists(cand):
                    return cand

    for root in [base_dir, csv_dir]:
        if root:
            cand = os.path.join(root, p)
            if os.path.exists(cand):
                return cand

    return p


def _split_mask(series, split_type):
    x = series.fillna("").astype(str).str.lower()
    if split_type == "train_meta":
        return x.str.contains("train")
    if split_type == "test_meta":
        return x.str.contains("test")
    return x.str.contains(split_type.lower())


def _guess_label_from_row(row, class_to_idx, cls_names, label_col, fallback_cols):
    if label_col is not None:
        value = row.get(label_col)
        if pd.notna(value):
            if str(value).isdigit():
                v = int(value)
                if v in class_to_idx.values():
                    return v
            txt = str(value)
            if txt in class_to_idx:
                return class_to_idx[txt]
            txt_low = txt.lower()
            for name in cls_names:
                if name.lower() in txt_low:
                    return class_to_idx[name]

    for c in fallback_cols:
        txt = str(row.get(c, ""))
        txt_low = txt.lower()
        for name in cls_names:
            if name.lower() in txt_low:
                return class_to_idx[name]

    return 0


class PairedContrastiveDataset(Dataset):
    def __init__(self, csv_path, split_type="train_meta", config=None, logger=None):
        self.config = config
        self.logger = logger
        csv_path = str(csv_path)
        csv_dir = str(Path(csv_path).resolve().parent)
        base_dir = self.config.get("base_dir", "")

        df = pd.read_csv(csv_path)

        split_col = _find_column(df, ["split_type", "split", "set", "subset"])
        if split_col is not None:
            df = df[_split_mask(df[split_col], split_type)].copy()

        out_col = _find_column(df, ["out_path", "out", "mix_path", "aug_path", "audio_out"])
        in_col = _find_column(df, ["in_path", "in", "clean_path", "src_path", "audio_in"])

        if out_col is None and in_col is None:
            raise ValueError("CSV is missing audio path columns. Please provide at least out_path or in_path.")

        if out_col is None:
            out_col = in_col
        if in_col is None:
            in_col = out_col

        self.out_paths = [_resolve_audio_path(p, csv_dir, base_dir) for p in df[out_col].tolist()]
        self.in_paths = [_resolve_audio_path(p, csv_dir, base_dir) for p in df[in_col].tolist()]

        cls_names = self.config["class_names"]
        class_to_idx = self.config["class_to_idx"]
        label_col = _find_column(df, ["label", "class", "target", "category", "ship_type"])
        fallback_cols = [c for c in ["rel_path", out_col, in_col] if c is not None]

        self.labels = [
            _guess_label_from_row(row, class_to_idx, cls_names, label_col, fallback_cols)
            for _, row in df.iterrows()
        ]

        self.tgt_samps = int(self.config["sample_rate"] * self.config["duration"])

        if self.logger:
            self.logger.print(
                f"Loaded [{split_type}] dataset -> {len(self.out_paths)} pairs | split_col={split_col} | out_col={out_col} | in_col={in_col} | label_col={label_col}"
            )

    def __len__(self):
        return len(self.out_paths)

    def _load_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if torch.isnan(wav).any() or torch.isinf(wav).any():
                return torch.randn(1, self.tgt_samps) * 1e-5

            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.config["sample_rate"]:
                wav = torchaudio.functional.resample(wav, sr, self.config["sample_rate"])
            if wav.size(1) < self.tgt_samps:
                wav = F.pad(wav, (0, self.tgt_samps - wav.size(1)))
            else:
                start = random.randint(0, wav.size(1) - self.tgt_samps)
                wav = wav[:, start : start + self.tgt_samps]
            return wav
        except Exception:
            return torch.randn(1, self.tgt_samps) * 1e-5

    def __getitem__(self, idx):
        wav_out = self._load_audio(self.out_paths[idx])
        wav_in = self._load_audio(self.in_paths[idx])
        lbl = self.labels[idx]
        return wav_out.squeeze(0), wav_in.squeeze(0), torch.tensor(lbl, dtype=torch.long)


class ChannelSubtraction(nn.Module):
    def __init__(self, num_mels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_mels, 1) * 0.5)

    def forward(self, log_mel):
        env_profile = log_mel.mean(dim=-1, keepdim=True)
        return log_mel - (self.alpha * env_profile)


class GPUAudioPreprocessor(nn.Module):
    def __init__(self, config, res_type="high", apply_aug=True):
        super().__init__()
        self.target_len = config["target_len"]
        n_fft = config["fft_high"] if res_type == "high" else config["fft_low"]
        hop = config["hop_high"] if res_type == "high" else config["hop_low"]
        mels = config["mels_high"] if res_type == "high" else config["mels_low"]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["sample_rate"], n_fft=n_fft, hop_length=hop, n_mels=mels
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.subtraction = ChannelSubtraction(mels)

        self.apply_aug = apply_aug
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=max(1, int(mels * 0.2)))
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=max(1, int(self.target_len * 0.1)))

    def forward(self, wav):
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        mel = self.mel_transform(wav)
        log_mel = self.amp_to_db(mel)
        if log_mel.size(-1) < self.target_len:
            log_mel = F.pad(log_mel, (0, self.target_len - log_mel.size(-1)))
        else:
            log_mel = log_mel[..., : self.target_len]

        log_mel = log_mel.unsqueeze(1)
        clean_log_mel = self.subtraction(log_mel)

        c_mean = clean_log_mel.mean(dim=[-2, -1], keepdim=True)
        c_std = clean_log_mel.std(dim=[-2, -1], keepdim=True)
        norm_mel = (clean_log_mel - c_mean) / (c_std + 1e-5)
        norm_mel = torch.nan_to_num(norm_mel, nan=0.0, posinf=0.0, neginf=0.0)

        if self.training and self.apply_aug:
            norm_mel = norm_mel.squeeze(1)
            norm_mel = self.freq_mask(norm_mel)
            norm_mel = self.time_mask(norm_mel)
            norm_mel = norm_mel.unsqueeze(1)

        return norm_mel


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x).view(x.size(0), x.size(1))
        y = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.se(self.conv(x)))


class DualResNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.b1 = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.b2 = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x1, x2):
        f1, f2 = self.b1(x1).flatten(1), self.b2(x2).flatten(1)
        return self.fc(torch.cat([f1, f2], dim=1))


class SimCLR(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x1, x2):
        return self.projector(self.encoder(x1, x2))


def info_nce_loss(features, temperature=0.5):
    b = features.shape[0] // 2
    labels = torch.cat([torch.arange(b) for _ in range(2)], dim=0).to(features.device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    features = F.normalize(features, dim=1, eps=1e-5)
    sim_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

    positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels_ce = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    return F.cross_entropy(logits, labels_ce)


class LinearProbeEvaluator(nn.Module):
    def __init__(self, encoder_path, num_classes):
        super().__init__()
        self.encoder = DualResNet(feat_dim=128)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, s1, s2):
        with torch.no_grad():
            feat = self.encoder(s1, s2)
        return self.classifier(feat), feat


def pretrain_contrastive(config, out_dir, logger):
    logger.print("=== [PHASE 1] Start contrastive pretraining ===")
    device = config["device"]

    dataset = PairedContrastiveDataset(config["CSV_PATH"], split_type="train_meta", config=config, logger=logger)
    if len(dataset) == 0:
        raise RuntimeError("train_meta is empty. Please check whether the split column contains train entries.")

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=len(dataset) >= config["batch_size"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )

    prep_high = GPUAudioPreprocessor(config, "high").to(device)
    prep_low = GPUAudioPreprocessor(config, "low").to(device)
    encoder = DualResNet(feat_dim=128)
    model = SimCLR(encoder).to(device)

    params = list(model.parameters()) + list(prep_high.parameters()) + list(prep_low.parameters())
    optimizer = optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["pretrain_epochs"])
    scaler = GradScaler(enabled=("cuda" in str(device)))

    best_loss = float("inf")
    best_ckpt_path = os.path.join(out_dir, "encoder_contrastive_best.pt")
    best_prep_high = os.path.join(out_dir, "prep_high.pt")
    best_prep_low = os.path.join(out_dir, "prep_low.pt")

    pretrain_loss_hist = []
    pretrain_lr_hist = []

    for epoch in range(config["pretrain_epochs"]):
        model.train()
        prep_high.train()
        prep_low.train()
        total_loss = 0.0

        for wav_out, wav_in, _ in loader:
            wav_out, wav_in = wav_out.to(device), wav_in.to(device)
            optimizer.zero_grad()

            with autocast(enabled=("cuda" in str(device))):
                s1_out, s2_out = prep_high(wav_out), prep_low(wav_out)
                s1_in, s2_in = prep_high(wav_in), prep_low(wav_in)
                z1 = model(s1_out, s2_out)
                z2 = model(s1_in, s2_in)
                loss = info_nce_loss(torch.cat([z1, z2], dim=0))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if not torch.isnan(loss):
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        pretrain_loss_hist.append(float(avg_loss))
        pretrain_lr_hist.append(float(lr_now))

        logger.print(
            f"Pretrain Epoch [{epoch + 1:03d}/{config['pretrain_epochs']}] | Avg Loss: {avg_loss:.4f} | LR: {lr_now:.6f}"
        )

        if avg_loss < best_loss and not np.isnan(avg_loss):
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), best_ckpt_path)
            torch.save(prep_high.state_dict(), best_prep_high)
            torch.save(prep_low.state_dict(), best_prep_low)
            logger.print("⭐ Found lower loss, saved best model.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pretrain_loss_hist) + 1), pretrain_loss_hist, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Pretraining Contrastive Loss")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "00_pretrain_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pretrain_lr_hist) + 1), pretrain_lr_hist, marker="x", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Pretraining Learning Rate Schedule")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "00_pretrain_lr_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return best_ckpt_path, best_prep_high, best_prep_low


def run_evaluation(config, out_dir, logger, encoder_path, prep_high_path, prep_low_path):
    logger.print("=== [PHASE 2] Start linear evaluation ===")
    device = config["device"]

    train_ds = PairedContrastiveDataset(config["CSV_PATH"], split_type="train_meta", config=config, logger=logger)
    test_ds = PairedContrastiveDataset(config["CSV_PATH"], split_type="test_meta", config=config, logger=logger)

    if len(test_ds) == 0:
        raise RuntimeError("test_meta is empty. Please check whether the split column contains test entries.")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )

    model = LinearProbeEvaluator(encoder_path, config["num_classes"]).to(device)

    prep_high = GPUAudioPreprocessor(config, "high", apply_aug=False).to(device)
    prep_low = GPUAudioPreprocessor(config, "low", apply_aug=False).to(device)
    prep_high.load_state_dict(torch.load(prep_high_path, map_location=device))
    prep_low.load_state_dict(torch.load(prep_low_path, map_location=device))

    prep_high.eval()
    prep_low.eval()
    for p in prep_high.parameters():
        p.requires_grad = False
    for p in prep_low.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    history = {"epoch": [], "test_loss": [], "test_acc": []}

    y_true, y_pred, y_probs, y_feats = [], [], [], []

    for epoch in range(1, config["eval_epochs"] + 1):
        model.train()
        for wav_out, _, lbl in train_loader:
            wav_out, lbl = wav_out.to(device), lbl.to(device)
            with torch.no_grad():
                s1, s2 = prep_high(wav_out), prep_low(wav_out)
            optimizer.zero_grad()
            out, _ = model(s1, s2)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

        model.eval()
        t_loss, t_acc, t_tot = 0.0, 0.0, 0
        epoch_true, epoch_pred, epoch_probs, epoch_feats = [], [], [], []

        with torch.no_grad():
            for wav_out, _, lbl in test_loader:
                wav_out, lbl = wav_out.to(device), lbl.to(device)
                s1, s2 = prep_high(wav_out), prep_low(wav_out)
                out, feat = model(s1, s2)

                loss = criterion(out, lbl)
                t_loss += loss.item() * wav_out.size(0)
                t_acc += (out.argmax(1) == lbl).sum().item()
                t_tot += wav_out.size(0)

                epoch_true.extend(lbl.cpu().numpy())
                epoch_pred.extend(out.argmax(1).cpu().numpy())
                epoch_probs.extend(F.softmax(out, dim=1).cpu().numpy())
                epoch_feats.extend(feat.cpu().numpy())

        test_l = t_loss / max(1, t_tot)
        test_a = (t_acc / max(1, t_tot)) * 100.0
        history["epoch"].append(epoch)
        history["test_loss"].append(test_l)
        history["test_acc"].append(test_a)

        logger.print(
            f"Eval Epoch [{epoch:02d}/{config['eval_epochs']}] Loss: {test_l:.4f} | Acc: {test_a:.2f}%"
        )

        if epoch == config["eval_epochs"]:
            y_true, y_pred, y_probs, y_feats = epoch_true, epoch_pred, epoch_probs, epoch_feats

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    y_feats = np.array(y_feats)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(config["num_classes"])),
        target_names=config["class_names"],
        digits=4,
        zero_division=0,
    )
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {
        "final_test_acc": float(history["test_acc"][-1]),
        "macro_recall": macro_recall,
        "num_test_samples": int(len(y_true)),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.print("\n" + report)
    logger.print(f"Macro Recall: {macro_recall:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Test Acc (%)", color="tab:blue")
    ax1.plot(history["epoch"], history["test_acc"], color="tab:blue", marker="o")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Test Loss", color="tab:red")
    ax2.plot(history["epoch"], history["test_loss"], color="tab:red", marker="x", linestyle="dashed")
    plt.title("Evaluation Metrics over Epochs")
    plt.savefig(os.path.join(out_dir, "01_epoch_metrics.png"), dpi=300)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config["num_classes"])))
    cm_norm = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_norm[i] = cm[i].astype(float) / row_sums[i]

    annot = np.asarray(
        [f"{v1}\n({v2:.1%})" for v1, v2 in zip(cm.flatten(), cm_norm.flatten())]
    ).reshape(cm.shape)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=config["class_names"],
        yticklabels=config["class_names"],
    )
    plt.title("Confusion Matrix on test_meta")
    plt.savefig(os.path.join(out_dir, "02_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    try:
        X_emb = TSNE(n_components=2, random_state=42).fit_transform(y_feats)
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, config["num_classes"]))
        for i, c in enumerate(config["class_names"]):
            idx = y_true == i
            if np.any(idx):
                plt.scatter(X_emb[idx, 0], X_emb[idx, 1], label=c, alpha=0.8, color=colors[i], edgecolors="w")
        plt.legend(title="Classes")
        plt.title("t-SNE Feature Distribution")
        plt.savefig(os.path.join(out_dir, "03_tsne_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.print(f"t-SNE plotting failed: {e}")

    y_bin = label_binarize(y_true, classes=list(range(config["num_classes"])))
    plt.figure(figsize=(10, 8))
    colors = ["blue", "red", "green", "orange", "purple", "brown", "cyan", "magenta"]
    for i in range(config["num_classes"]):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{config["class_names"][i]} (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title("Multi-class ROC Curve on test_meta")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "04_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.print(f"✅ Pipeline finished. Results saved to: {out_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV does not exist: {args.csv_path}")

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        class_names = ["ClassA", "ClassB", "ClassC", "ClassD", "ClassE"]

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    out_dir = args.out_dir if args.out_dir else f"./Result_Contrastive_AutoEval_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    logger = Logger(os.path.join(out_dir, "full_pipeline_log.txt"))

    config = {
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "target_len": args.target_len,
        "fft_high": args.fft_high,
        "hop_high": args.hop_high,
        "mels_high": args.mels_high,
        "fft_low": args.fft_low,
        "hop_low": args.hop_low,
        "mels_low": args.mels_low,
        "CSV_PATH": args.csv_path,
        "base_dir": args.base_dir,
        "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "eval_epochs": args.eval_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
    }

    start = time.time()
    logger.print("Starting ShipsEar WOA contrastive train-and-evaluate pipeline")
    logger.print(json.dumps({k: v for k, v in config.items() if k != "class_to_idx"}, ensure_ascii=False, indent=2))

    best_enc, best_prep_h, best_prep_l = pretrain_contrastive(config, out_dir, logger)
    run_evaluation(config, out_dir, logger, best_enc, best_prep_h, best_prep_l)

    elapsed = (time.time() - start) / 60.0
    logger.print(f"Total elapsed time: {elapsed:.2f} minutes")
    logger.close()


if __name__ == "__main__":
    main()
