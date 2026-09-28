#!/usr/bin/env python3
"""Fine-tune (or train from scratch) a MONAI SegResNet on ONE AMOS/BTCV CV fold.

- Architecture matches the wholeBody_ct_segmentation bundle (SegResNet, init_filters=32,
  blocks_down=[1,2,2,4], blocks_up=[1,1,1]); with `pretrained: true` we load the bundle's
  wholeBody weights and re-init only the final conv (105 -> n_classes).
- GT masks are the TRUE native-scheme masks (folds/*.json point at `.../mask/`), remapped to a
  contiguous target label set (kidney L+R merged).
- Train on random foreground patches; validate with sliding-window inference; report per-organ Dice.

Run (mrseg env has monai):
  python train_segresnet.py --config configs/amos_cv.yaml --fold 0
"""
import argparse, os, json, yaml, numpy as np, torch
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (Compose, MapTransform, EnsureChannelFirstd, Orientationd,
                              RandCropByPosNegLabeld, RandFlipd, RandShiftIntensityd, ToTensord)

REMAP = {'amos': {1:1, 2:2, 3:2, 4:3, 6:4, 7:5, 8:6, 10:7, 15:8},
         'btcv': {1:1, 2:2, 3:2, 4:3, 6:4, 7:5, 8:6, 11:7}}
ORGANS = {'amos': ['spleen','kidney','gallbladder','liver','stomach','aorta','pancreas','prostate'],
          'btcv': ['spleen','kidney','gallbladder','liver','stomach','aorta','pancreas']}

class LoadNpzd(MapTransform):
    """Load our npz volumes (key 'arr'); clip image HU to [-1000,1000] & z-score; remap label."""
    def __init__(self, ds):
        super().__init__(['image', 'label']); self.rm = REMAP[ds]
    def __call__(self, d):
        d = dict(d)
        img = np.load(d['image']); img = img[img.files[0]].astype(np.float32)
        img = np.clip(img, -1000, 1000)
        img = (img - img.mean()) / (img.std() + 1e-6)
        lab = np.load(d['label']); lab = lab[lab.files[0]]
        out = np.zeros_like(lab, dtype=np.int64)
        for k, v in self.rm.items():
            out[lab == k] = v
        d['image'] = img[None]; d['label'] = out[None].astype(np.float32)   # (1,H,W,D)
        return d

def transforms(ds, patch, train):
    t = [LoadNpzd(ds), ToTensord(keys=['image', 'label'])]
    if train:
        t += [RandCropByPosNegLabeld(keys=['image','label'], label_key='label', spatial_size=[patch]*3,
                                     pos=2, neg=1, num_samples=2, image_key='image', allow_smaller=True),
              RandFlipd(keys=['image','label'], prob=0.2, spatial_axis=0),
              RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3)]
    return Compose(t)

def build_model(n_classes, pretrained, ckpt, dev):
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=n_classes, init_filters=32,
                    blocks_down=[1,2,2,4], blocks_up=[1,1,1], dropout_prob=0.2).to(dev)
    if pretrained:
        sd = torch.load(ckpt, map_location='cpu'); sd = sd.get('state_dict', sd)
        sd = {k: v for k, v in sd.items() if not k.startswith('conv_final.2.conv')}  # drop 105-class head
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f'[pretrained] loaded wholeBody weights | missing {len(missing)} unexpected {len(unexpected)} '
              f'(final conv re-init to {n_classes} classes)')
    else:
        print('[scratch] SegResNet random init')
    return net

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--fold', type=int, required=True)
    a = ap.parse_args(); c = yaml.safe_load(open(a.config))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(c.get('seed', 0))
    ds = c['dataset']; organs = ORGANS[ds]; n_classes = len(organs) + 1
    fold = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.config)), '..', f'folds/{ds}_fold{a.fold}.json')))
    patch = c.get('patch', 96)
    tr = DataLoader(Dataset(fold['train'], transforms(ds, patch, True)), batch_size=c.get('batch_size', 2),
                    shuffle=True, num_workers=c.get('num_workers', 8))
    va = DataLoader(Dataset(fold['val'], transforms(ds, patch, False)), batch_size=1, num_workers=4)

    net = build_model(n_classes, c.get('pretrained', True), c.get('pretrained_ckpt'), dev)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(net.parameters(), lr=c.get('learning_rate', 1e-4), weight_decay=c.get('weight_decay', 1e-5))
    dice = DiceMetric(include_background=False, reduction='mean_batch')

    def validate():
        net.eval(); dice.reset()
        with torch.no_grad():
            for b in va:
                img = b['image'].to(dev)
                logits = sliding_window_inference(img, [c.get('val_patch', 160)]*3, 2, net, overlap=0.5)
                pred = torch.nn.functional.one_hot(logits.argmax(1), n_classes).permute(0,4,1,2,3)
                gt = torch.nn.functional.one_hot(b['label'].long().squeeze(1), n_classes).permute(0,4,1,2,3).to(dev)
                dice(pred, gt)
        per = dice.aggregate().cpu().numpy()   # (n_classes-1,) per-organ (bg excluded)
        return {organs[i]: round(float(per[i]), 4) for i in range(len(organs))}, float(np.mean(per))

    epochs = c.get('epochs', 100); best = -1; best_state = None; best_per = None
    for ep in range(epochs):
        net.train()
        for b in tr:
            opt.zero_grad()
            out = net(b['image'].to(dev))
            loss = loss_fn(out, b['label'].to(dev))
            loss.backward(); opt.step()
        if (ep+1) % c.get('val_interval', 10) == 0 or ep == epochs-1:
            per, mean = validate()
            print(f'ep {ep+1}/{epochs}  val macro-Dice {mean:.4f}  {per}', flush=True)
            if mean > best: best, best_state, best_per = mean, {k: v.cpu() for k, v in net.state_dict().items()}, per

    os.makedirs(c['output_dir'], exist_ok=True)
    tag = f"{ds}_fold{a.fold}_{'pretrained' if c.get('pretrained',True) else 'scratch'}"
    torch.save(best_state, os.path.join(c['output_dir'], f'{tag}.pt'))
    json.dump({'fold': a.fold, 'dataset': ds, 'pretrained': c.get('pretrained', True),
               'macro_dice': best, 'per_organ': best_per},
              open(os.path.join(c['output_dir'], f'{tag}_dice.json'), 'w'), indent=1)
    print(f'\n=== BEST fold {a.fold} macro-Dice {best:.4f} ===\n{best_per}')

if __name__ == '__main__':
    main()
