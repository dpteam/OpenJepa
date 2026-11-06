"""Обучение JEPA на траекториях.

Запуск:

    python -m openjepa.train.pretrain --config configs/pretrain_toy.yaml

Логика намеренно линейная: один оптимизатор, один планировщик LR, один цикл.
Всё, что сложнее, добавляется сверху и не ломает воспроизводимость.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.synthetic import BallEnvConfig, TrajectoryDataset, collate
from ..models.hierarchy import HJEPA, HierarchyConfig
from ..models.jepa import JEPA, JEPAConfig


# --------------------------------------------------------------------------- #
# утилиты
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return base_lr * (0.1 + 0.9 * step / max(1, warmup))
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def build_model(cfg: dict):
    m = cfg["model"]
    base = JEPAConfig(
        img_size=m["img_size"],
        patch_size=m["patch_size"],
        in_chans=m.get("in_chans", 3),
        dim=m["dim"],
        depth=m["depth"],
        heads=m["heads"],
        proj_dim=m["proj_dim"],
        proj_hidden=m.get("proj_hidden", 512),
        proj_norm=m.get("proj_norm", "batch"),
        action_dim=m["action_dim"],
        ctx=m.get("ctx", 2),
        horizon=m.get("horizon", 1),
        pred_depth=m.get("pred_depth", 3),
        pred_hidden=m.get("pred_hidden", 384),
        lamb=m.get("lamb", 0.05),
        inv_weight=m.get("inv_weight", 0.5),
        metric_weight=m.get("metric_weight", 1.0),
        probe_weight=m.get("probe_weight", 1.0),
        var_weight=m.get("var_weight", 1.0),
        var_target=m.get("var_target", 1.0),
        position_in_latent=m.get("position_in_latent", True),
        stop_grad_target=m.get("stop_grad_target", False),
        sigreg_slices=m.get("sigreg_slices", 256),
    )
    if m.get("hierarchical", False):
        hcfg = HierarchyConfig(
            levels=m.get("levels", 3),
            strides=m.get("strides", [1, 4, 16]),
            base=base,
        )
        return HJEPA(hcfg), hcfg
    return JEPA(base), base


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# основной цикл
# --------------------------------------------------------------------------- #
def train(cfg: dict, out_dir: str | Path | None = None) -> Path:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))

    out = Path(out_dir or cfg.get("out_dir", "runs/toy"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    d = cfg["data"]
    env_cfg = BallEnvConfig(**d.get("env", {}))
    train_ds = TrajectoryDataset(
        n_traj=d.get("n_traj", 4000),
        length=d["length"],
        cfg=env_cfg,
        seed=seed,
    )
    val_ds = TrajectoryDataset(
        n_traj=d.get("n_val", 128),
        length=d["length"],
        cfg=env_cfg,
        seed=seed + 991,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("workers", 0),
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("eval_batch_size", 32),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model, model_cfg = build_model(cfg)
    model = model.to(device)
    print(f"device={device}  params={count_params(model):,}")

    tcfg = cfg["train"]
    opt = torch.optim.AdamW(model.param_groups(tcfg["lr"], tcfg.get("weight_decay", 0.05)))
    total_steps = len(train_dl) * tcfg["epochs"]
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.05))
    use_amp = tcfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(
            ["step", "epoch", "loss", "pred", "inv", "metric", "probe", "sigreg", "val_pred", "val_std", "lr", "sec"]
        )

    best = float("inf")
    step = 0
    t0 = time.time()

    for epoch in range(tcfg["epochs"]):
        model.train()
        run_loss = run_pred = run_reg = run_inv = run_met = run_probe = 0.0
        n_batches = 0

        for batch in train_dl:
            frames = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            positions = batch.get("positions")
            if positions is not None:
                positions = positions.to(device, non_blocking=True)

            lr = cosine_warmup(step, total_steps, warmup, tcfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out_dict = model(frames, actions, positions)
                loss = out_dict["loss"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()

            run_loss += float(loss.detach())
            run_pred += float(out_dict["pred_loss"])
            run_inv += float(out_dict.get("inv_loss", torch.zeros(())))
            run_met += float(out_dict.get("metric_loss", torch.zeros(())))
            run_probe += float(out_dict.get("probe_loss", torch.zeros(())))
            run_reg += float(out_dict["sigreg_loss"])
            n_batches += 1
            step += 1

        # Пересчитываем BN-статистики на текущих весах, иначе eval-метрика
        # меряет не то, что выучила модель (см. recalibrate_bn).
        recalibrate_bn(model, train_dl, device, max_batches=8)
        # Проба должна видеть те латенты, которые будут при инференсе,
        # и сойтись: недотренированная проба даёт ошибку 25 px вместо 0.4.
        calibrate_probe(
            model,
            train_dl,
            device,
            steps=tcfg.get("probe_steps", 400),
            lr=tcfg.get("probe_lr", 3e-3),
        )
        val_pred, val_std = evaluate(model, val_dl, device, use_amp)
        avg = run_loss / max(1, n_batches)
        writer.writerow(
            [
                step,
                epoch,
                f"{avg:.6f}",
                f"{run_pred / max(1, n_batches):.6f}",
                f"{run_inv / max(1, n_batches):.6f}",
                f"{run_met / max(1, n_batches):.6f}",
                f"{run_probe / max(1, n_batches):.6f}",
                f"{run_reg / max(1, n_batches):.6f}",
                f"{val_pred:.6f}",
                f"{val_std:.6f}",
                f"{lr:.6e}",
                f"{time.time() - t0:.1f}",
            ]
        )
        log_file.flush()
        print(
            f"epoch {epoch:3d} | loss {avg:.4f} | pred {run_pred / max(1, n_batches):.4f} "
            f"| inv {run_inv / max(1, n_batches):.4f} "
            f"| met {run_met / max(1, n_batches):.4f} "
            f"| probe {run_probe / max(1, n_batches):.4f} "
            f"| sigreg {run_reg / max(1, n_batches):.4f} | val_pred {val_pred:.4f} "
            f"| val_std {val_std:.3f} | {time.time() - t0:.0f}s"
        )
        if val_std < 0.05:
            print(
                "  WARNING: похоже на коллапс латента в eval-режиме "
                f"(val_std={val_std:.4f}); чекпоинт не считается лучшим"
            )

        if val_pred < best and val_std >= 0.05:
            best = val_pred
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_pred": best,
                },
                out / "best.pt",
            )
        if tcfg.get("save_every") and (epoch + 1) % tcfg["save_every"] == 0:
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch},
                out / f"epoch{epoch + 1}.pt",
            )

    log_file.close()
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": tcfg["epochs"] - 1}, out / "last.pt")
    print(f"готово: {out} (best val_pred={best:.4f})")
    return out


@torch.no_grad()
def recalibrate_bn(model, dl, device, max_batches: int = 8) -> None:
    """Пересчитывает running-статистики BatchNorm на текущих весах.

    Зачем это нужно. BatchNorm хранит скользящее среднее/дисперсию активаций.
    Пока модель учится, распределение активаций дрейфует, и статистики,
    накопленные на шаге 100, плохо описывают модель на шаге 3000. В
    eval-режиме это приводит к другому масштабу латента: в нашем прогоне
    норма латента была 36 в eval против 7.6 в train, а `pred_loss` — 3.2
    против 0.24. То есть метрика врала в 13 раз, а инференс работал на
    «сломанном» представлении.

    Решение — стандартное (BN recalibration): перед оценкой сбрасываем
    статистики и прогоняем несколько батчей в train-режиме, чтобы они
    соответствовали текущим весам. Данные берём обучающие, чтобы не
    подсматривать в валидацию.
    """
    bns = [
        m
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # накапливаем точное среднее по всем батчам

    was_training = model.training
    model.train()
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        model(batch["frames"].to(device), batch["actions"].to(device),
              batch["positions"].to(device) if "positions" in batch else None)

    for m in bns:
        m.momentum = 0.1
    model.train(was_training)


def base_jepa(model):
    """Возвращает JEPA-часть модели.

    `JEPA` — сама себе база; `HJEPA` хранит общий энкодер как `JEPA` в поле
    `encoder`. Такой хелпер нужен, чтобы калибровка и оценка работали для
    обеих моделей.
    """
    return getattr(model, "encoder", model) if hasattr(model, "predictors") else model


@torch.no_grad()
def _cache_features(
    model, dl, device, max_batches: int = 4
) -> dict[str, torch.Tensor] | None:
    """Считает латенты и токены в eval-режиме, складывает в кэш.

    Зачем кэш. Пробы — маленькие головы, им нужно много шагов, чтобы сойтись
    (60 шагов давали 25 px, 400 — доли пикселя). Считать энкодер на каждом
    шаге было бы дорого; признаки считаются один раз.

    Токены хранятся в fp16: 4 батча × 32 траектории × 17 кадров × 64 патча ×
    192 = ~27M значений, в fp32 это 107 МБ, в fp16 — вдвое меньше.
    """
    model.eval()
    base = base_jepa(model)
    zs, toks, ps = [], [], []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        positions = batch.get("positions")
        if positions is None:
            return None
        tok, _ = base.encoder.forward_tokens(batch["frames"].to(device))
        pooled = base.projector(tok.mean(dim=2)).float()
        if base.cfg.position_in_latent:
            zfull = torch.cat([pooled, base.probe(tok.float())], dim=-1)
        else:
            zfull = pooled
        zs.append(zfull.reshape(-1, base.latent_dim))
        toks.append(tok.half().reshape(-1, tok.shape[-2], tok.shape[-1]))
        ps.append(positions.to(device).float().reshape(-1, positions.shape[-1]))
    if not zs:
        return None
    return {
        "z": torch.cat(zs, dim=0),
        "tok": torch.cat(toks, dim=0),
        "p": torch.cat(ps, dim=0),
    }


def calibrate_probe(
    model,
    dl,
    device,
    steps: int = 400,
    lr: float = 3e-3,
    max_batches: int = 4,
    verbose: bool = False,
) -> float:
    """Дообучает обе пробы (`z -> p` и `tokens -> p`) на eval-признаках.

    Зачем отдельный этап. Проектор использует `BatchNorm`, который в `train()`
    нормирует по текущему батчу, а в `eval()` — по running-статистикам. Из-за
    этого латент в двух режимах разный, и проба, обученная в `train()`,
    в инференсе видит «другие» векторы.

    Второй момент: **пробы склонны к недотренированности**. На нашем
    чекпоинте 60 шагов давали ошибку 25 px, 400 шагов — доли пикселя.
    Поэтому по умолчанию 400 шагов и lr 3e-3.

    Решение стандартное: энкодер замораживаем, головы учатся на тех
    признаках, которые они реально увидят при инференсе. Это та же логика,
    что «линейная проба на замороженных признаках» в LeJEPA/DINO.
    """
    spatial = getattr(model, "probe", None)
    latent = getattr(model, "z_probe", None)
    if spatial is None or latent is None:
        base = base_jepa(model)
        spatial = spatial if spatial is not None else getattr(base, "probe", None)
        latent = latent if latent is not None else getattr(base, "z_probe", None)
    if spatial is None and latent is None:
        return float("nan")

    cache = _cache_features(model, dl, device, max_batches)
    if cache is None:
        return float("nan")
    z_all, tok_all, p_all = cache["z"], cache["tok"], cache["p"]
    n = z_all.shape[0]
    if n < 2:
        return float("nan")

    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    heads = [h for h in (spatial, latent) if h is not None and len(list(h.parameters())) > 0]
    for h in heads:
        for p in h.parameters():
            p.requires_grad_(True)

    # Сброс голов: веса с прошлого этапа могут держать в локальном минимуме.
    for h in heads:
        for m in h.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        lin = [m for m in h.modules() if isinstance(m, torch.nn.Linear)]
        if lin:
            torch.nn.init.zeros_(lin[-1].weight)
            torch.nn.init.zeros_(lin[-1].bias)

    params = [p for h in heads for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, steps), eta_min=lr * 0.02
    )
    bs = min(512, n)
    last = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=z_all.device)
        target = p_all[idx]
        loss = z_all.new_zeros(())
        if spatial is not None:
            loss = loss + torch.nn.functional.smooth_l1_loss(
                spatial(tok_all[idx].float()), target
            )
        if latent is not None:
            loss = loss + torch.nn.functional.smooth_l1_loss(
                latent(z_all[idx]), target
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        last = float(loss.detach())

    for p in model.parameters():
        p.requires_grad_(True)
    model.train(was_training)
    if verbose:
        print(f"    probe: loss={last:.5f} ({steps} шагов на {n} кадрах)")
    return last


@torch.no_grad()
def evaluate(model, dl, device, use_amp: bool) -> tuple[float, float]:
    """Возвращает (pred_loss, min_std_по_осям_латента).

    Вторая метрика — защита от «тихого» коллапса: если в eval-режиме
    дисперсия латента падает почти до нуля, pred_loss тоже падает (все
    предсказания тривиальны), и без этой проверки можно выбрать худший
    чекпоинт.
    """
    model.eval()
    total, n = 0.0, 0
    min_std = float("inf")
    for batch in dl:
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)
        positions = batch.get("positions")
        if positions is not None:
            positions = positions.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = model(frames, actions, positions)
            base = base_jepa(model)
            z = base.encode_sequence(frames).float().reshape(-1, base.latent_dim)
        total += float(out["pred_loss"]) * frames.shape[0]
        n += frames.shape[0]
        min_std = min(min_std, float(z.std(dim=0).min()))
    return total / max(1, n), min_std


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение JEPA / H-JEPA")
    ap.add_argument("--config", required=True, help="путь к yaml/json конфигу")
    ap.add_argument("--out", default=None, help="переопределить out_dir")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, args.out)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
