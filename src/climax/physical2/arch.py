# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from climax.arch import ClimaX

class RegionalClimaX(ClimaX):
    def __init__(self, default_vars, img_size=..., patch_size=2, embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.1, drop_rate=0.1):
        super().__init__(default_vars, img_size, patch_size, embed_dim, depth, decoder_depth, num_heads, mlp_ratio, drop_path, drop_rate)


        self.phys_vars = ["2m_temperature", "10m_u_component_of_wind", "geopotential_500"]

        # 获取物理变量在输入中的通道索引
        self.phys_var_indices = [default_vars.index(var) for var in self.phys_vars if var in default_vars]

        # 替换原始Transformer块为物理引导的块
        self.blocks = nn.ModuleList([
            PhysicsAwareTransformerBlock(embed_dim, num_heads, self.phys_var_indices)
            for _ in range(depth)
        ])
    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables, region_info):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # get the patch ids corresponding to the region
        region_patch_ids = region_info['patch_ids']
        x = x[:, :, region_patch_ids, :]

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed[:, region_patch_ids, :]

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat, region_info):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            region_info: Containing the region's information

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x, lead_times, variables, region_info)  # B, L, D
        print("Encoder output shape:", out_transformers.shape)
        preds = self.head(out_transformers)  # B, L, V*p*p
        # 在 PhysicsAwareTransformerBlock 的 forward 方法中添加
        self.log("train/phys_weight", self.phys_weight.item(), on_step=True)
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        preds = self.unpatchify(preds, h = max_h - min_h + 1, w = max_w - min_w + 1)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix, region_info):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat, region_info=region_info)

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]
        clim = clim[:, min_h:max_h+1, min_w:max_w+1]

        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]

    # 在 arch.py 中添加以下代码

    class PhysicsAwareTransformerBlock(nn.Module):
        """Transformer Block with enhanced attention to physics-related variables."""

        def __init__(self, embed_dim, num_heads, physics_var_indices):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads)
            self.phys_var_indices = physics_var_indices  # 物理变量的通道索引列表
            self.embed_dim = embed_dim

            # 定义物理变量的注意力权重增强系数
            self.phys_weight = nn.Parameter(torch.ones(1))

        def forward(self, x):
            # x shape: [Batch, Sequence, Embedding]
            # 生成物理变量的注意力掩码
            mask = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
            mask[self.phys_var_indices] = True  # 标记物理变量位置

            # 增强物理变量的Query向量
            query = x.clone()
            query[:, mask] *= self.phys_weight  # 放大物理变量的Query

            # 计算注意力
            attn_output, _ = self.attention(query, x, x)
            return attn_output