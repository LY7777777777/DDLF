# coding: utf-8
"""
Stage1: ImageNet ResNet18
Stage2 : dual-branch  + MR-FP + Pareto merge
Results: outputs/result/<dataset>/<method>/<backbone>/{y_hat,y_true,logits}.txt
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as tvm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from Dataset import Dataset
from DDGF_stage2 import build_tau_sampler_by_idx, run_mrfp_stage2
from Utils import *

def infer_num_classes(train_txt: str) -> int:
    max_lab = -1
    with open(train_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                max_lab = max(max_lab, int(parts[1]))
    if max_lab < 0:
        raise ValueError(f'Could not infer num_classes from {train_txt}')
    return max_lab + 1


def _build_backbone(name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """ResNet18 from torchvision ImageNet weights; fc replaced for num_classes."""
    name = name.lower()
    if name != 'resnet18':
        raise ValueError(
            f'Unsupported backbone {name!r}. This script uses torchvision resnet18 only; '
            'install timm and extend _build_backbone if you need other nets.'
        )
    try:
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1
        model = tvm.resnet18(weights=weights)
    except Exception:
        model = tvm.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model.to(device)


def _imgnet_train_val_transforms(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def parse_args():
    p = argparse.ArgumentParser(description='K-fold CV training (last epoch evaluation)')
    p.add_argument('--dataset', type=str, default='KLSG')
    p.add_argument('--backbone', type=str, default='resnet18')
    p.add_argument('--method', type=str, default='DDGF_nomf')
    p.add_argument('--p_value', type=int, default=0)
    p.add_argument('--k_value', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs_stage2', type=int, default=40)
    p.add_argument('--lr_stage2', type=float, default=2e-3, help='Stage2 lr.')

    p.add_argument(
        '--stage2_max_grad_norm',
        type=float,
        default=20, #20
        help='Clip merged gradient L2 norm before SGD step .',
    )
    p.add_argument('--tau_b', type=float, default=0.0, help='Tau for balanced-branch WeightedRandomSampler.')
    p.add_argument('--loss_mode', type=str, default='none', choices=['none', 'bal', 'uni', 'bal+uni'])
    p.add_argument('--merge_mode', type=str, default='pareto', choices=['pareto', 'fixed'])
    p.add_argument('--fixed_alpha', type=float, default=1.0)
    p.add_argument('--fp_weight', type=float, default=1.0)
    p.add_argument('--topk_fp', type=int, default=0)
    p.add_argument('--lam_mr', type=float, default=4.0, help='MR-FP weight.')
    p.add_argument('--alpha_target', type=float, default=8.0, help='Pareto merge alpha.')#8
    p.add_argument('--pareto_alpha_max', type=float, default=8.0)#8
    p.add_argument('--save_results', type=str, default='True')
    p.add_argument('--save_models', type=str, default='False')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--num_classes', type=int, default=-1, help='-1: infer from train.txt labels')
    p.add_argument('--seed', type=int, default=77)
    p.add_argument(
        '--quiet',
        action='store_true',
        help='Disable per-epoch log lines (only print summary at end).',
    )
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ('true', '1', 'yes', 'y')


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    data_dir = os.path.join(ROOT, '..' , 'Data', args.dataset)
    train_txt = os.path.join(data_dir, 'train.txt')
    if not os.path.isfile(train_txt):
        raise FileNotFoundError(f'Missing {train_txt}. Expected Data/<dataset>/train.txt under project root.')

    kfold_tr = os.path.join(data_dir, 'kfold_train.txt')
    kfold_va = os.path.join(data_dir, 'kfold_val.txt')
    if not os.path.isfile(kfold_tr) or not os.path.isfile(kfold_va):
        raise FileNotFoundError(f'Missing kfold files under {data_dir}')

    num_classes = args.num_classes if args.num_classes > 0 else infer_num_classes(train_txt)

    train_idx, val_idx = get_kfold_img_idx(args.p_value, args.k_value, args.dataset, sample_type=None)
    if train_idx is None or val_idx is None:
        raise RuntimeError(f'get_kfold_img_idx returned None for dataset={args.dataset} p={args.p_value} k={args.k_value}')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_iter, val_iter = get_kfold_img_iters(
        args.batch_size,
        data_dir,
        train_idx,
        val_idx,
        mean,
        std,
        num_workers=args.num_workers,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_backbone(args.backbone, num_classes, device)

    # ========== Stage 1 (baseline): full net, plain CE, SGD + cosine ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    tag = f'[{args.dataset} p{args.p_value}-k{args.k_value}]'
    if not args.quiet:
        print(
            f'{tag} Start: epochs={args.epochs} batch_size={args.batch_size} device={device}',
            flush=True,
        )

    for epoch in range(args.epochs):
        # --- Stage1 train: one full pass over train_iter ---
        model.train()
        running_loss = 0.0
        n_samples = 0
        for X, y in train_iter:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * X.size(0)
            n_samples += X.size(0)
        scheduler.step()
        if not args.quiet:
            avg_loss = running_loss / max(n_samples, 1)
            lr_now = scheduler.get_last_lr()[0]
            model.eval()
            val_gmean = float(evaluate_gmean(val_iter, model, device=device))
            model.train()
            print(
                f'{tag} [Stage1] epoch {epoch + 1}/{args.epochs}  '
                f'train_loss={avg_loss:.4f}  lr={lr_now:.6f}  val_gmean={val_gmean:.4f}',
                flush=True,
            )

    # -------- Stage2: dual-branch (mrfp_test), same SGD as Stage1 --------
    if args.epochs_stage2 > 0:
        lr2 = args.lr_stage2 if args.lr_stage2 > 0 else 1e-4
        train_tf, val_tf = _imgnet_train_val_transforms(mean, std)
        train_ds_s2 = Dataset(train_txt, train_idx, transform=train_tf)
        val_ds_s2 = Dataset(train_txt, val_idx, transform=val_tf)
        train_loader_uni = torch.utils.data.DataLoader(
            train_ds_s2,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        sampler_b, _counts = build_tau_sampler_by_idx(
            train_txt, train_idx, num_classes, tau=args.tau_b,
            num_samples=len(train_idx), replacement=True,
        )
        train_loader_bal = torch.utils.data.DataLoader(
            train_ds_s2,
            batch_size=args.batch_size,
            sampler=sampler_b,
            shuffle=False,
            num_workers=args.num_workers,
        )
        val_loader_s2 = torch.utils.data.DataLoader(
            val_ds_s2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if not args.quiet:
            print(
                f'{tag} Stage2: epochs={args.epochs_stage2} lr={lr2} '
                f'clip={args.stage2_max_grad_norm} loss_mode={args.loss_mode}',
                flush=True,
            )
        run_mrfp_stage2(
            model=model,
            train_loader_uni=train_loader_uni,
            train_loader_bal=train_loader_bal,
            val_loader=val_loader_s2,
            device=device,
            nb_classes=num_classes,
            num_epochs=args.epochs_stage2,
            lr=lr2,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
            loss_mode=args.loss_mode,
            merge_mode=args.merge_mode,
            fixed_alpha=args.fixed_alpha,
            fp_weight=args.fp_weight,
            topk_fp=args.topk_fp,
            class_prior=None,
            lam_mr=args.lam_mr,
            alpha_target=args.alpha_target,
            pareto_alpha_max=args.pareto_alpha_max,
            quiet=args.quiet,
            tag=tag,
            max_grad_norm=args.stage2_max_grad_norm,
        )

    # Last-epoch checkpoint
    model.eval()
    gmean, y_hat, y_true, logits = evaluate_gmean(
        val_iter, model, device=device, if_get_y=True, if_get_logits=True
    )

    if _truthy(args.save_results):
        result_dir = os.path.join(ROOT, '../' ,'outputs', 'result', args.dataset, args.method, args.backbone)
        os.makedirs(result_dir, exist_ok=True)
        remove_fold_results(result_dir, args.p_value, args.k_value)
        logits_np = logits.detach().cpu().numpy()
        write_result_to_file(
            result_dir,
            y_hat.detach().cpu(),
            y_true.detach().cpu(),
            logits=logits_np,
            p=args.p_value,
            k=args.k_value,
        )
        print(f'[p{args.p_value}-k{args.k_value}] Last-epoch G-mean: {float(gmean):.4f} -> {result_dir}')

    if _truthy(args.save_models):
        ckpt_dir = os.path.join(ROOT, '../' , 'outputs', 'models', args.dataset, args.method, args.backbone)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'model_p{args.p_value}_k{args.k_value}.pth')
        torch.save(model.state_dict(), path)
        print(f'Saved weights: {path}')


if __name__ == '__main__':
    main()
