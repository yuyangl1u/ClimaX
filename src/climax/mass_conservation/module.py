# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from climax.regional_forecast.arch import RegionalClimaX
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from climax.utils.pos_embed import interpolate_pos_embed


class RegionalForecastModule(LightningModule):
    """Lightning module for regional forecasting with the ClimaX model.

    Args:
        net (ClimaX): ClimaX model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: RegionalClimaX,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        lambda_physics: float = 0.1,  # 新增物理损失权重

    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def get_patch_size(self):
        return self.net.patch_size

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch

        # 获取模型输出的损失和预测结果
        loss_dict, preds = self.net.forward(
            x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=self.lat, region_info=region_info
        )
        loss_dict = loss_dict[0]

        # 提取物理约束所需的变量（假设输入 x 和预测 preds 的通道顺序已知）
        rho_current = x[:, 0, ...]  # 输入 x 的第 0 通道是当前密度
        rho_next = preds[:, 0, ...]  # 预测 preds 的第 0 通道是密度
        u = preds[:, 1, ...]  # 第 1 通道是 x 方向速度
        v = preds[:, 2, ...]  # 第 2 通道是 y 方向速度

        # 计算时间导数 ∂ρ/∂t（lead_times 单位是小时，转换为秒）
        dt_seconds = lead_times * 3600
        drho_dt = (rho_next - rho_current) / dt_seconds.unsqueeze(-1).unsqueeze(-1)

        # 计算空间梯度 ∇·(ρu)（假设网格均匀，使用 torch.gradient）
        # 注意：需根据实际网格调整（例如使用 region_info 中的经纬度距离）
        rho_u = rho_next * u
        rho_v = rho_next * v

        # 纬度方向梯度（height 维度）
        dudx = torch.gradient(rho_u, dim=2)[0]  # shape: [B, C, H, W]
        # 经度方向梯度（width 维度）
        dvdy = torch.gradient(rho_v, dim=3)[0]  # shape: [B, C, H, W]
        divergence = dudx + dvdy

        # 连续性方程残差：∂ρ/∂t + ∇·(ρu) = 0
        continuity_residual = drho_dt + divergence
        continuity_loss = torch.mean(continuity_residual ** 2)

        # 总损失 = 原损失 + 物理约束损失（使用超参数 lambda_physics）
        lambda_physics = self.hparams.lambda_physics  # 需在 __init__ 中添加此参数
        total_loss = loss_dict["loss"] + lambda_physics * continuity_loss

        # 记录物理损失
        self.log("train/physics_loss", continuity_loss, on_step=True, on_epoch=False, prog_bar=True)
        loss_dict["loss"] = total_loss

        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
            region_info=region_info,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
            region_info=region_info,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
