#!/usr/bin/env python3
"""
Fine-tune / linear-probe a *published 3D CT foundation vision encoder* on a downstream
CT classification task — to compare **Merlin** (Stanford abdominal CT) and **CT-CLIP_v2**
(CT-RATE chest CT) as downstream representations, alongside the chest VISION-CT vs ImageNet
comparison in ../vision_encoder_finetune.

Unlike that sibling kit, Merlin and CT-CLIP are *different architectures* with their *own*
preprocessing, so this is a "which pretrained encoder transfers best" comparison, not a
same-architecture init comparison. Each encoder:
  - loads its published weights,
  - preprocesses the RAW volume with its own recipe (monai; reorients LAS/other -> RAS),
  - emits a pooled feature vector, on top of which we train a classification head.

encoder_source:
  merlin  -> Merlin(ImageEmbedding=True); feature 2048-d;  input 224x224x160, spacing (1.5,1.5,3), HU[-1000,1000]->[0,1]
  ctclip  -> CT-CLIP_v2 CTViT tokens mean-pooled; feature 512-d; input 240x480x480, spacing (0.75,0.75,1.5), HU[-1000,1000]->[-1,1]

ENV (each encoder needs its own venv — they have conflicting deps):
  merlin  ->  python  # (Merlin environment)
  ctclip  ->  python  # (CT-CLIP environment)

Usage:  <env-python> train_foundation_classifier.py --config configs/lung1_5yr_survival_merlin.yaml
"""
import argparse, os, yaml, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
                              ScaleIntensityRanged, SpatialPadd, CenterSpatialCropd, ToTensord)

# ============================ encoder plug-ins ============================
def merlin_transform():
    return Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=['image'], axcodes='RAS'),
        Spacingd(keys=['image'], pixdim=(1.5, 1.5, 3.0), mode='bilinear'),
        ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image'], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(roi_size=[224, 224, 160], keys=['image']), ToTensord(keys=['image'])])

def ctclip_transform():
    return Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=['image'], axcodes='RAS'),
        Spacingd(keys=['image'], pixdim=(0.75, 0.75, 1.5), mode='bilinear'),
        ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True),
        SpatialPadd(keys=['image'], spatial_size=[480, 480, 240]),
        CenterSpatialCropd(roi_size=[480, 480, 240], keys=['image']), ToTensord(keys=['image'])])

def arrange_merlin(vol):   # monai (1,X,Y,Z) -> encoder input (1,224,224,160)
    return vol
def arrange_ctclip(vol):   # (1,X,Y,Z) -> (1, Z=240, X=480, Y=480) frames-first
    return vol[0].permute(2, 0, 1).contiguous().unsqueeze(0)

def build_merlin(cfg):
    from merlin import Merlin
    enc = Merlin(ImageEmbedding=True)
    def feat(x):
        o = enc(x)                                 # Merlin returns (1,b,2048) or (b,1,2048)
        return o.reshape(-1, o.shape[-1])          # -> (b,2048), feature dim last
    return enc, 2048, feat

def build_ctclip(cfg):
    from transformer_maskgit import CTViT
    from transformers import BertTokenizer, BertModel
    from ct_clip import CTCLIP
    tok = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    te = BertModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized'); te.resize_token_embeddings(len(tok))
    ie = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10,
               spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=ie, text_encoder=te, dim_text=768, dim_image=294912, dim_latent=512,
                  extra_latent_projection=True, use_mlm=False, downsample_image_embeds=False, use_all_token_embeds=False)
    clip.load(cfg['ctclip_ckpt'])
    enc = clip.visual_transformer                  # keep ONLY the vision encoder (CTViT)
    # free the text tower + contrastive projections (unused; matters for full-FT memory)
    for attr in ('text_transformer', 'to_text_latent', 'to_visual_latent',
                 'to_text_latent_extra', 'to_visual_latent_extra'):
        if hasattr(clip, attr):
            try: setattr(clip, attr, None)
            except Exception: pass
    def feat(x):
        tokens = enc(x, return_encoded_tokens=True)   # (b,H,W,Z,512)
        return tokens.mean(dim=(1, 2, 3))             # (b,512)
    return enc, 512, feat

ENCODERS = {
    'merlin': dict(build=build_merlin, transform=merlin_transform, arrange=arrange_merlin),
    'ctclip': dict(build=build_ctclip, transform=ctclip_transform, arrange=arrange_ctclip),
}

# ============================ model ============================
class Classifier(nn.Module):
    def __init__(self, feat_dim, num_outputs, dropout=0.1):
        super().__init__()
        # ---- EDIT THE HEAD HERE for a deeper head; keep identical across encoders ----
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_outputs))
    def forward(self, f): return self.head(f)

# ============================ data ============================
class VolDataset(Dataset):
    """Reads the RAW volume per PatientID and applies the encoder's own preprocessing."""
    def __init__(self, df, cfg, transform, arrange):
        self.df = df.reset_index(drop=True); self.c = cfg; self.xf = transform; self.arrange = arrange
    def __len__(self): return len(self.df)
    def _path(self, r):
        if self.c.get('nifti_template'):
            return self.c['nifti_template'].format(pid=r[self.c.get('pid_col', 'PatientID')])
        return r[self.c['img_col']]
    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = self.xf({'image': self._path(r)})['image']
        x = self.arrange(vol)
        x = (x.as_tensor() if hasattr(x, 'as_tensor') else x).float()   # strip monai MetaTensor
        if self.c['task'] == 'multilabel':
            y = torch.tensor([float(r[l]) for l in self.c['label_cols']])
        elif self.c['task'] == 'multiclass':
            y = torch.tensor(int(r[self.c['label_col']]))
        else:
            y = torch.tensor([float(r[self.c['label_col']])])
        return x, y

# ============================ metrics ============================
def evaluate(logits, Y, task):
    if task == 'multiclass':
        prob = logits.softmax(1).numpy(); y = Y.numpy()
        acc = accuracy_score(y, prob.argmax(1))
        aucs = [roc_auc_score((y == k).astype(int), prob[:, k]) for k in range(prob.shape[1])
                if 0 < (y == k).sum() < len(y)]
        return {'accuracy': round(acc, 4), 'macro_AUROC': round(float(np.mean(aucs)), 4)}
    prob = torch.sigmoid(logits).numpy(); y = Y.numpy()
    aucs = [roc_auc_score(y[:, k], prob[:, k]) for k in range(prob.shape[1]) if 0 < y[:, k].sum() < len(y)]
    return {'macro_AUROC': round(float(np.mean(aucs)), 4), 'n_labels_scored': len(aucs)}

# ============================ feature cache (frozen encoder / linear probe) ============================
@torch.no_grad()
def extract_features(enc, feat_fn, loader, dev):
    enc.eval(); F, Y = [], []
    for x, y in loader:
        F.append(feat_fn(x.to(dev)).float().cpu()); Y.append(y)
    return torch.cat(F), torch.cat(Y)

# ============================ train ============================
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    c = yaml.safe_load(open(args.config))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(c.get('seed', 0))
    spec = ENCODERS[c['encoder_source']]
    enc, feat_dim, feat_fn = spec['build'](c); enc = enc.to(dev)
    xf, arrange = spec['transform'](), spec['arrange']
    n_out = len(c['label_cols']) if c['task'] == 'multilabel' else (c['num_classes'] if c['task'] == 'multiclass' else 1)
    head = Classifier(feat_dim, n_out, c.get('dropout', 0.1)).to(dev)

    def loader(split, shuffle):
        df = pd.read_csv(c[f'{split}_csv'])
        return DataLoader(VolDataset(df, c, xf, arrange), batch_size=c['batch_size'],
                          shuffle=shuffle, num_workers=c.get('num_workers', 8), drop_last=False)

    if c['task'] == 'multiclass':
        loss_fn = nn.CrossEntropyLoss()
    else:
        pw = c.get('pos_weight'); pw = torch.tensor(pw, dtype=torch.float32, device=dev) if pw is not None else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    def do_loss(logit, y):
        return loss_fn(logit, y if c['task'] == 'multiclass' else y.float())

    freeze = c.get('freeze_encoder', True)   # default: linear probe (foundation encoders are heavy)
    print(f"[{c['encoder_source']}] feat_dim={feat_dim}  task={c['task']}  freeze_encoder={freeze}")

    if freeze:
        # ---- LINEAR PROBE: encode every volume ONCE, then train the head on cached features ----
        for p in enc.parameters(): p.requires_grad = False
        cache = {s: extract_features(enc, feat_fn, loader(s, False), dev) for s in ('train', 'val', 'test')}
        print('cached features:', {s: tuple(cache[s][0].shape) for s in cache})
        opt = torch.optim.AdamW(head.parameters(), lr=c['learning_rate'], weight_decay=c.get('weight_decay', 0.01))
        Ftr, Ytr = cache['train']; tds = DataLoader(TensorDataset(Ftr, Ytr), batch_size=c.get('head_batch_size', 64), shuffle=True)
        best, best_state = -1, None
        for ep in range(c['epochs']):
            head.train()
            for f, y in tds:
                opt.zero_grad(); do_loss(head(f.to(dev)), y.to(dev)).backward(); opt.step()
            head.eval()
            with torch.no_grad(): mval = evaluate(head(cache['val'][0].to(dev)).cpu(), cache['val'][1], c['task'])
            key = mval.get('macro_AUROC', mval.get('accuracy'))
            if key > best: best, best_state = key, {k: v.cpu() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        with torch.no_grad(): mte = evaluate(head(cache['test'][0].to(dev)).cpu(), cache['test'][1], c['task'])
    else:
        # ---- FULL FINE-TUNE (expensive; CT-CLIP CTViT at 480^3 likely needs batch_size 1) ----
        enc_lr = c['learning_rate'] * c.get('encoder_lr_mult', 0.1)
        opt = torch.optim.AdamW([{'params': enc.parameters(), 'lr': enc_lr},
                                 {'params': head.parameters(), 'lr': c['learning_rate']}],
                                weight_decay=c.get('weight_decay', 0.01))
        tr, va, te = loader('train', True), loader('val', False), loader('test', False)
        best, best_state = -1, None
        for ep in range(c['epochs']):
            enc.train(); head.train()
            for x, y in tr:
                opt.zero_grad(); do_loss(head(feat_fn(x.to(dev))), y.to(dev)).backward(); opt.step()
            lg, Y = [], []
            enc.eval(); head.eval()
            with torch.no_grad():
                for x, y in va: lg.append(head(feat_fn(x.to(dev))).cpu()); Y.append(y)
            mval = evaluate(torch.cat(lg), torch.cat(Y), c['task'])
            key = mval.get('macro_AUROC', mval.get('accuracy'))
            print(f'epoch {ep+1}/{c["epochs"]}  val {mval}')
            if key > best: best, best_state = key, {'enc': {k: v.cpu() for k, v in enc.state_dict().items()},
                                                    'head': {k: v.cpu() for k, v in head.state_dict().items()}}
        enc.load_state_dict(best_state['enc']); head.load_state_dict(best_state['head'])
        lg, Y = [], []
        enc.eval(); head.eval()
        with torch.no_grad():
            for x, y in te: lg.append(head(feat_fn(x.to(dev))).cpu()); Y.append(y)
        mte = evaluate(torch.cat(lg), torch.cat(Y), c['task'])

    print('\n=== TEST (best-val checkpoint) ===')
    print(f"encoder_source={c['encoder_source']}  task={c['task']}  ->  {mte}")
    os.makedirs(c['output_dir'], exist_ok=True)
    torch.save(head.state_dict(), os.path.join(c['output_dir'], 'head_best.pt'))

if __name__ == '__main__':
    main()
