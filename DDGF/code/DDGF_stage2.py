# coding: utf-8
"""
Stage-2 dual-branch training
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from Utils import cal_gmean


def _stage2_lr_triangle(epoch_idx: int, num_epochs: int, peak_lr: float) -> float:
    """Linear ramp 0 -> peak_lr -> 0 over epoch indices (inclusive endpoints at 0)."""
    if num_epochs <= 0 or peak_lr <= 0:
        return float(peak_lr)
    if num_epochs == 1:
        return float(peak_lr)
    if num_epochs == 2:
        # Two points cannot both be 0 and hit peak; use 0 then full peak.
        return 0.0 if epoch_idx == 0 else float(peak_lr)
    x = epoch_idx / (num_epochs - 1)
    factor = 1.0 - abs(2.0 * x - 1.0)
    return float(peak_lr * max(0.0, factor))


def load_labels_from_train_txt(train_txt_path: str, idx_list: List[int], nb_classes: int):
    with open(train_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    labels = np.array([int(lines[i].rstrip().split()[1]) for i in idx_list], dtype=np.int64)
    counts = np.bincount(labels, minlength=nb_classes)
    return labels, counts


def build_tau_sampler_by_idx(
    train_txt_path: str,
    idx_list: List[int],
    nb_classes: int,
    tau: float,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> Tuple[WeightedRandomSampler, np.ndarray]:
    tau = float(max(0.0, min(1.0, tau)))
    labels, counts = load_labels_from_train_txt(train_txt_path, idx_list, nb_classes)
    base = counts[labels].astype(np.float32)
    base = np.maximum(base, 1.0)
    weights = np.power(base, tau - 1.0).astype(np.float32)
    weights = torch.from_numpy(weights)
    if num_samples is None:
        num_samples = len(idx_list)
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=int(num_samples), replacement=replacement
    )
    return sampler, counts


def mr_fpc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    fp_weight: float = 1.0,
    topk_fp: int = 0,
    class_prior: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    K = int(num_classes)
    if class_prior is None:
        prior = torch.full((K,), 1.0 / K, device=logits.device, dtype=p.dtype)
    else:
        prior = class_prior.to(device=logits.device, dtype=p.dtype).clamp_min(eps)

    per_cls = []
    for c in range(K):
        pos = (labels == c)
        if not pos.any():
            continue
        neg = ~pos
        pc_pos = p[pos, c].clamp_min(eps)
        pos_term = (-torch.log(pc_pos)).mean()
        if neg.any():
            pc_neg = p[neg, c]
            if topk_fp and topk_fp > 0:
                k = min(int(topk_fp), int(pc_neg.numel()))
                fp_stat = pc_neg.topk(k).values.mean()
            else:
                fp_stat = pc_neg.mean()
            fp_term = torch.log1p(fp_stat / prior[c])
        else:
            fp_term = torch.tensor(0.0, device=logits.device, dtype=p.dtype)
        per_cls.append(pos_term + fp_weight * fp_term)

    if not per_cls:
        return torch.tensor(0.0, device=logits.device, dtype=p.dtype)
    return torch.stack(per_cls).mean()


def ce_mrfp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ce_criterion: nn.Module,
    num_classes: int,
    lam_mr: float,
    enable_mrfp: bool,
    fp_weight: float = 1.0,
    topk_fp: int = 0,
    class_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = ce_criterion(logits, labels)
    if enable_mrfp and lam_mr > 0:
        loss = loss + lam_mr * mr_fpc_loss(
            logits, labels, num_classes=num_classes,
            fp_weight=fp_weight, topk_fp=topk_fp, class_prior=class_prior,
        )
    return loss


def pareto_safe_alpha(
    dot: float,
    nu: float,
    nbv: float,
    alpha_target: float,
    alpha_max: float,
    eps: float = 1e-12,
) -> float:
    if dot >= 0.0:
        alpha = alpha_target
    else:
        a_low = (-dot) / (nbv + eps)
        a_up = nu / ((-dot) + eps)
        if a_low <= a_up:
            alpha = min(max(alpha_target, a_low), a_up)
        else:
            alpha = 0.0
    return float(max(0.0, min(alpha_max, alpha)))


def resolve_alpha(
    merge_mode: str,
    dot: float,
    nu: float,
    nbv: float,
    alpha_target: float,
    alpha_max: float,
    fixed_alpha: float,
    eps: float = 1e-12,
) -> float:
    if merge_mode == 'pareto':
        return pareto_safe_alpha(dot, nu, nbv, alpha_target, alpha_max, eps=eps)
    if merge_mode == 'fixed':
        return float(fixed_alpha)
    raise ValueError(f'Unsupported merge_mode: {merge_mode}')


def run_mrfp_stage2(
    model: nn.Module,
    train_loader_uni: DataLoader,
    train_loader_bal: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    nb_classes: int,
    num_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    loss_mode: str = 'bal',
    merge_mode: str = 'pareto',
    fixed_alpha: float = 1.0,
    fp_weight: float = 1.0,
    topk_fp: int = 0,
    class_prior: Optional[torch.Tensor] = None,
    lam_mr: float = 4.0,
    alpha_target: float = 7.0,
    pareto_alpha_max: float = 4.0,
    pareto_eps: float = 1e-12,
    quiet: bool = False,
    tag: str = '',
    max_grad_norm: float = 2.0,
) -> None:
    """Dual-branch Pareto/fixed merge; fixed lam_mr and alpha_target (no ramps)."""
    valid_modes = {'none', 'bal', 'uni', 'bal+uni'}
    if loss_mode not in valid_modes:
        raise ValueError(f'Unsupported loss_mode: {loss_mode}')
    if merge_mode not in {'pareto', 'fixed'}:
        raise ValueError(f'Unsupported merge_mode: {merge_mode}')

    apply_mrfp_uni = loss_mode in {'uni', 'bal+uni'}
    apply_mrfp_bal = loss_mode in {'bal', 'bal+uni'}

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    for epoch2 in range(num_epochs):
        lr_epoch = _stage2_lr_triangle(epoch2, num_epochs, lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_epoch

        params = list(model.parameters())
        steps_per_epoch = max(len(train_loader_uni), len(train_loader_bal))
        it_u = iter(train_loader_uni)
        it_b = iter(train_loader_bal)

        model.train()
        for _ in range(steps_per_epoch):
            try:
                images_u, labels_u = next(it_u)
            except StopIteration:
                it_u = iter(train_loader_uni)
                images_u, labels_u = next(it_u)
            try:
                images_b, labels_b = next(it_b)
            except StopIteration:
                it_b = iter(train_loader_bal)
                images_b, labels_b = next(it_b)

            images_u = images_u.to(device, non_blocking=True)
            labels_u = labels_u.to(device, non_blocking=True)
            images_b = images_b.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)

            logits_u = model(images_u)
            loss_u = ce_mrfp_loss(
                logits_u, labels_u, criterion, nb_classes,
                lam_mr=lam_mr, enable_mrfp=apply_mrfp_uni,
                fp_weight=fp_weight, topk_fp=topk_fp, class_prior=class_prior,
            )

            logits_b = model(images_b)
            loss_b = ce_mrfp_loss(
                logits_b, labels_b, criterion, nb_classes,
                lam_mr=lam_mr, enable_mrfp=apply_mrfp_bal,
                fp_weight=fp_weight, topk_fp=topk_fp, class_prior=class_prior,
            )

            optimizer.zero_grad(set_to_none=True)
            grads_u = torch.autograd.grad(
                loss_u, params, retain_graph=True, create_graph=False, allow_unused=False,
            )
            grads_b = torch.autograd.grad(
                loss_b, params, retain_graph=False, create_graph=False, allow_unused=False,
            )

            dot_t = torch.zeros((), device=device)
            nu_t = torch.zeros((), device=device)
            nbv_t = torch.zeros((), device=device)
            for gu, gb in zip(grads_u, grads_b):
                dot_t += (gu * gb).sum()
                nu_t += (gu * gu).sum()
                nbv_t += (gb * gb).sum()

            dot = float(dot_t.item())
            nu = float(nu_t.item())
            nbv = float(nbv_t.item())

            alpha = resolve_alpha(
                merge_mode, dot, nu, nbv,
                alpha_target=alpha_target,
                alpha_max=pareto_alpha_max,
                fixed_alpha=fixed_alpha,
                eps=pareto_eps,
            )

            for p, gu, gb in zip(params, grads_u, grads_b):
                p.grad = gu + alpha * gb

            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

            optimizer.step()

        if not quiet:
            model.eval()
            y_true_list, y_pred_list = [], []
            n_ok, n_tot = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X).argmax(dim=1)
                    y_true_list.extend(y.cpu().numpy().tolist())
                    y_pred_list.extend(pred.cpu().numpy().tolist())
                    n_ok += (pred == y).sum().item()
                    n_tot += y.numel()
            val_acc = n_ok / max(1, n_tot)
            y_t = np.asarray(y_true_list, dtype=np.int64)
            y_p = np.asarray(y_pred_list, dtype=np.int64)
            val_gmean = cal_gmean(y_t, y_p)
            val_gmean_f = float(val_gmean) if val_gmean is not None else float('nan')
            print(
                f'{tag} [Stage2] epoch {epoch2 + 1}/{num_epochs} '
                f'val_acc={val_acc:.4f} lr={lr_epoch:.6f} val_gmean={val_gmean_f:.4f} ',
                flush=True,
            )
            model.train()
