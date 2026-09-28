#!/usr/bin/env python3
"""
Fine-tune a 3D vision encoder on a downstream CT classification task, to compare the
**chest-Uniferum** vision encoder (from checkpoint-75000) against the **same timm_3d
architecture with ImageNet weights**.

Supports binary / multi-class / multi-label classification — set `task` in the config.
The backbone is identical in both cases (timm_3d tf_efficientnetv2_b0, in_chans=1); only the
initial weights differ, so any difference in downstream accuracy reflects the pre-training.

Usage:  python train_classifier.py --config configs/uniferum_encoder.yaml
"""
import argparse, os, yaml, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm_3d
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score

# ----------------------------- model -----------------------------
def build_encoder(source, uniferum_ckpt, model_name):
    """source: 'uniferum' (load vision_model.model.* from checkpoint) or 'imagenet' (timm pretrained)."""
    backbone = timm_3d.create_model(model_name, pretrained=(source == 'imagenet'),
                                    in_chans=1, num_classes=0, global_pool='avg')
    if source == 'uniferum':
        sd = load_file(uniferum_ckpt)
        enc = {k.replace('vision_model.model.', ''): v for k, v in sd.items()
               if k.startswith('vision_model.model.')}
        missing, unexpected = backbone.load_state_dict(enc, strict=False)
        print(f'[encoder=uniferum] loaded {len(enc)} keys | missing {len(missing)} | unexpected {len(unexpected)}')
        assert len(enc) > 300, 'expected ~418 vision_model.model.* keys — check the checkpoint path'
    else:
        print('[encoder=imagenet] timm_3d ImageNet-inflated weights')
    return backbone, backbone.num_features   # num_features = 1280 for b0

class Classifier(nn.Module):
    def __init__(self, backbone, feat_dim, num_outputs, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        # -------- EDIT THE HEAD HERE if you want a deeper head (see README) --------
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_outputs))
    def forward(self, x):
        return self.head(self.backbone(x))

# ----------------------------- data -----------------------------
def window(a, center, width):
    lo, hi = center - width / 2, center + width / 2
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)

def load_volume(path):
    z = np.load(path)
    return z['arr'] if 'arr' in z.files else z[z.files[0]]

class CTClsDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.reset_index(drop=True); self.c = cfg
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = load_volume(r[self.c['img_col']]).astype(np.float32)
        # prewindowed: npz already normalised to [0,1] (e.g. LUNG1 npz_histology_preprocessed);
        # otherwise the volume is raw HU and we window it here (raw-HU npz, e.g. npz_5yr_survival).
        arr = vol if self.c.get('prewindowed', False) else window(vol, self.c['window_center'], self.c['window_width'])
        t = torch.from_numpy(arr)[None, None]                                   # (1,1,H,W,Z)
        s = self.c['input_size']
        t = F.interpolate(t, size=(s, s, s), mode='trilinear', align_corners=False)[0]  # (1,s,s,s)
        if self.c['task'] == 'multilabel':
            y = torch.tensor([float(r[l]) for l in self.c['label_cols']])       # multi-hot
        elif self.c['task'] == 'multiclass':
            y = torch.tensor(int(r[self.c['label_col']]))                       # class index
        else:  # binary
            y = torch.tensor([float(r[self.c['label_col']])])
        return t, y

# ----------------------------- metrics -----------------------------
def evaluate(model, loader, task, device):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            logit = model(x.to(device)).cpu()
            P.append(logit); Y.append(y)
    P = torch.cat(P); Y = torch.cat(Y)
    if task == 'multiclass':
        prob = P.softmax(1).numpy(); y = Y.numpy()
        acc = accuracy_score(y, prob.argmax(1))
        aucs = []
        for k in range(prob.shape[1]):
            yk = (y == k).astype(int)
            if 0 < yk.sum() < len(yk): aucs.append(roc_auc_score(yk, prob[:, k]))
        return {'accuracy': round(acc, 4), 'macro_AUROC': round(float(np.mean(aucs)), 4)}
    else:  # binary / multilabel -> per-column AUROC
        prob = torch.sigmoid(P).numpy(); y = Y.numpy()
        aucs = []
        for k in range(prob.shape[1]):
            if 0 < y[:, k].sum() < len(y): aucs.append(roc_auc_score(y[:, k], prob[:, k]))
        return {'macro_AUROC': round(float(np.mean(aucs)), 4), 'n_labels_scored': len(aucs)}

# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    c = yaml.safe_load(open(args.config))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(c.get('seed', 0))

    backbone, feat = build_encoder(c['encoder_source'], c.get('uniferum_ckpt'), c['model_name'])
    n_out = len(c['label_cols']) if c['task'] == 'multilabel' else (c['num_classes'] if c['task'] == 'multiclass' else 1)
    model = Classifier(backbone, feat, n_out, c.get('dropout', 0.1)).to(dev)

    if c.get('freeze_encoder', False):
        for p in model.backbone.parameters(): p.requires_grad = False
        print('[linear-probe] encoder frozen; training head only')

    # discriminative LR: encoder vs head
    enc_lr = c['learning_rate'] * c.get('encoder_lr_mult', 1.0)
    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': enc_lr},
        {'params': model.head.parameters(),     'lr': c['learning_rate']},
    ], weight_decay=c.get('weight_decay', 0.01))

    if c['task'] == 'multiclass':
        loss_fn = nn.CrossEntropyLoss()
    else:
        # pos_weight upweights the positive class for imbalanced binary/multilabel targets
        # (e.g. LUNG1 5-yr survival is ~18% positive). Set `pos_weight` in the config:
        #   binary -> a scalar (e.g. n_neg/n_pos); multilabel -> a list, one weight per label.
        pw = c.get('pos_weight')
        pw = torch.tensor(pw, dtype=torch.float32, device=dev) if pw is not None else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)   # binary & multilabel

    def mk(split): return DataLoader(CTClsDataset(pd.read_csv(c[f'{split}_csv']), c),
                                     batch_size=c['batch_size'], shuffle=(split == 'train'),
                                     num_workers=c.get('num_workers', 8), drop_last=(split == 'train'))
    tr, va, te = mk('train'), mk('val'), mk('test')

    best, best_state = -1, None
    for ep in range(c['epochs']):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            logit = model(x.to(dev))
            y = y.to(dev)
            loss = loss_fn(logit, y if c['task'] == 'multiclass' else y.float())
            loss.backward(); opt.step()
        m = evaluate(model, va, c['task'], dev)
        key = m.get('macro_AUROC', m.get('accuracy'))
        print(f'epoch {ep+1}/{c["epochs"]}  val {m}')
        if key > best:
            best, best_state = key, {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    print('\n=== TEST (best-val checkpoint) ===')
    print(f'encoder_source={c["encoder_source"]}  task={c["task"]}  ->  {evaluate(model, te, c["task"], dev)}')
    os.makedirs(c['output_dir'], exist_ok=True)
    torch.save(best_state, os.path.join(c['output_dir'], 'best.pt'))

if __name__ == '__main__':
    main()
