from typing import Tuple

import torch
from torch import nn


class LimoNetFTV(nn.Module):
    def __init__(
        self,
        goal_dim: int = 3,
        path_length: int = 50,
        se2_dim: int = 3,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        image_size: Tuple[int, int] = (308, 476),
        patch_size: int = 14,
        temporal_len: int = 4,
        num_views: int = 3,
        compressed_tokens: int = 16,
        view_tokens: int = 16,
        attn_heads: int = 6,
        decoder_layers: int = 4,
        ff_dim_factor: int = 4,
    ):
        super().__init__()
        self.backbone = None
        self._initialized = False

        self.goal_dim = goal_dim
        self.path_length = path_length
        self.se2_dim = se2_dim
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_len = temporal_len
        self.num_views = num_views
        self.compressed_tokens = compressed_tokens
        self.view_tokens = view_tokens
        self.attn_heads = attn_heads
        self.decoder_layers = decoder_layers
        self.ff_dim_factor = ff_dim_factor

        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size

        self.setup()

    def setup(self):
        if self._initialized:
            return

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self.backbone_name,
            pretrained=self.pretrained,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

        self.embed_dim = self.backbone.embed_dim
        ff_dim = self.embed_dim * self.ff_dim_factor

        self.compression_queries = nn.Parameter(
            torch.randn(self.compressed_tokens, self.embed_dim) * 0.02
        )
        self.compression_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attn_heads,
            batch_first=True,
        )
        self.compression_norm = nn.LayerNorm(self.embed_dim)

        self.temporal_embed = nn.Embedding(self.temporal_len, self.embed_dim)
        self.view_embed = nn.Embedding(self.num_views, self.embed_dim)
        self.spatial_embed = nn.Embedding(self.compressed_tokens, self.embed_dim)

        self.view_queries = nn.Parameter(
            torch.randn(self.view_tokens, self.embed_dim) * 0.02
        )
        self.view_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attn_heads,
            batch_first=True,
        )
        self.view_norm = nn.LayerNorm(self.embed_dim)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attn_heads,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(self.embed_dim)

        self.goal_proj = nn.Linear(self.goal_dim, self.embed_dim)
        self.waypoint_embed = nn.Embedding(self.path_length, self.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.attn_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.decoder_layers
        )

        self.out_proj = nn.Linear(self.embed_dim, self.se2_dim)

        self._initialized = True

    def _compress_tokens(
        self, patch_tokens: torch.Tensor, B: int, T: int, V: int
    ) -> torch.Tensor:
        BTV, Np, D = patch_tokens.shape
        assert BTV == B * T * V, "patch_tokens batch dimension should be B*T*V"
        assert D == self.embed_dim, "patch token dim should match embed_dim"

        queries = self.compression_queries.unsqueeze(0).expand(BTV, -1, -1)
        compressed, _ = self.compression_attn(
            query=queries,
            key=patch_tokens,
            value=patch_tokens,
            need_weights=False,
        )
        compressed = self.compression_norm(compressed + queries)
        compressed = compressed.view(B, T, V, self.compressed_tokens, D)

        assert compressed.shape == (
            B,
            T,
            V,
            self.compressed_tokens,
            D,
        ), "compressed tokens should be [B, T, V, K, D]"
        return compressed

    def _add_factorized_positional_embeddings(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        B, T, V, K, D = tokens.shape
        assert T <= self.temporal_len, "T exceeds configured temporal_len"
        assert V <= self.num_views, "V exceeds configured num_views"
        assert K == self.compressed_tokens, "K should equal compressed_tokens"
        assert D == self.embed_dim, "token dim should match embed_dim"

        device = tokens.device
        temporal_ids = torch.arange(T, device=device)
        view_ids = torch.arange(V, device=device)
        spatial_ids = torch.arange(K, device=device)

        temporal_pos = self.temporal_embed(temporal_ids).view(1, T, 1, 1, D)
        view_pos = self.view_embed(view_ids).view(1, 1, V, 1, D)
        spatial_pos = self.spatial_embed(spatial_ids).view(1, 1, 1, K, D)
        return tokens + temporal_pos + view_pos + spatial_pos

    def _fuse_views(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, V, K, D = tokens.shape
        view_tokens = tokens.view(B * T, V * K, D)
        queries = self.view_queries.unsqueeze(0).expand(B * T, -1, -1)

        fused, _ = self.view_attn(
            query=queries,
            key=view_tokens,
            value=view_tokens,
            need_weights=False,
        )
        fused = self.view_norm(fused + queries)
        fused = fused.view(B, T, self.view_tokens, D)

        assert fused.shape == (
            B,
            T,
            self.view_tokens,
            D,
        ), "view fused tokens should be [B, T, K_view, D]"
        return fused

    def _fuse_temporal(self, view_fused: torch.Tensor) -> torch.Tensor:
        B, T, K_view, D = view_fused.shape
        current_tokens = view_fused[:, -1]
        history_tokens = view_fused.reshape(B, T * K_view, D)

        temporal_tokens, _ = self.temporal_attn(
            query=current_tokens,
            key=history_tokens,
            value=history_tokens,
            need_weights=False,
        )
        temporal_tokens = self.temporal_norm(temporal_tokens + current_tokens)

        assert temporal_tokens.shape == (
            B,
            self.view_tokens,
            D,
        ), "temporal tokens should be [B, K_view, D]"
        return temporal_tokens

    def _decode_path(
        self, temporal_tokens: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        B, K_view, D = temporal_tokens.shape
        assert K_view == self.view_tokens, "decoder memory should use K_view tokens"
        assert D == self.embed_dim, "decoder memory dim should match embed_dim"
        assert goal.shape == (B, self.goal_dim), "goal should be [B, goal_dim]"

        goal_emb = self.goal_proj(goal)
        waypoint_ids = torch.arange(self.path_length, device=goal.device)
        waypoint_pos = self.waypoint_embed(waypoint_ids)
        waypoint_pos = waypoint_pos.unsqueeze(0).expand(B, -1, -1)
        waypoint_queries = waypoint_pos + goal_emb.unsqueeze(1)

        assert waypoint_queries.shape == (
            B,
            self.path_length,
            D,
        ), "waypoint queries should be [B, Nw, D]"

        decoder_out = self.decoder(tgt=waypoint_queries, memory=temporal_tokens)
        path = self.out_proj(decoder_out)

        assert path.shape == (
            B,
            self.path_length,
            self.se2_dim,
        ), "path should be [B, path_length, se2_dim]"
        return path

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.setup()

        assert "image_seq" in batch, "batch must contain image_seq"
        assert "goal" in batch, "batch must contain goal"

        image_seq = batch["image_seq"]
        goal = batch["goal"]

        assert image_seq.ndim == 6, "image_seq should be [B, T, V, 3, H, W]"
        B, T, V, C, H, W = image_seq.shape
        assert C == 3, "image_seq channel dimension should be 3"
        assert (H, W) == self.image_size, "image_seq H,W should match image_size"
        assert T <= self.temporal_len, "T exceeds configured temporal_len"
        assert V <= self.num_views, "V exceeds configured num_views"
        assert goal.shape == (B, self.goal_dim), "goal should be [B, 3]"

        images = image_seq.reshape(B * T * V, C, H, W)
        assert images.shape == (
            B * T * V,
            3,
            H,
            W,
        ), "flattened images should be [B*T*V, 3, H, W]"

        features = self.backbone.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]
        assert patch_tokens.ndim == 3, "patch tokens should be [B*T*V, Np, D]"
        assert patch_tokens.shape[0] == B * T * V
        assert patch_tokens.shape[2] == self.embed_dim

        Np = patch_tokens.shape[1]
        expected_np = self.grid_h * self.grid_w
        assert Np == expected_np, "Np should match image_size and patch_size"

        patch_tokens = patch_tokens.view(B, T, V, Np, self.embed_dim)
        assert patch_tokens.shape == (
            B,
            T,
            V,
            Np,
            self.embed_dim,
        ), "patch tokens should be [B, T, V, Np, D]"

        compressed = self._compress_tokens(
            patch_tokens.view(B * T * V, Np, self.embed_dim), B, T, V
        )
        compressed = self._add_factorized_positional_embeddings(compressed)

        view_fused = self._fuse_views(compressed)
        temporal_tokens = self._fuse_temporal(view_fused)
        path = self._decode_path(temporal_tokens, goal)
        return path


if __name__ == "__main__":
    model = LimoNetFTV(pretrained=False, temporal_len=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    trainable_backbone_params = [
        name for name, p in model.backbone.named_parameters() if p.requires_grad
    ]
    print("Trainable backbone parameters:")
    for name in trainable_backbone_params:
        print(f"  {name}")
    if not trainable_backbone_params:
        print("  <none>")

    batch = {
        "image_seq": torch.randn(1, 2, 3, 3, 308, 476),
        "goal": torch.randn(1, 3),
    }
    out = model(batch)
    print("Output shape:", out.shape)  # expected (1, 50, 3)

    target = torch.randn_like(out)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    print("loss:", loss.item())

    grad_names = [
        "compression_queries",
        "view_queries",
        "goal_proj.weight",
        "out_proj.weight",
    ]
    params = dict(model.named_parameters())
    for name in grad_names:
        grad = params[name].grad
        print(f"{name} grad is not None: {grad is not None}")
