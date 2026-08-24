import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

num_bins = 7

_BACKBONE_OUT_DIM = {
    'clip_vit_b16': 512,
}


class EarthMoverDistance(nn.Module):
    def __init__(self, dim=-1):
        super(EarthMoverDistance, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        """
        Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
        """
        cdf_x = torch.cumsum(x, dim=self.dim)
        cdf_y = torch.cumsum(y, dim=self.dim)
        emd = torch.norm(cdf_x - cdf_y, p=2, dim=self.dim)
        return emd

earth_mover_distance = EarthMoverDistance()


class NIMA(nn.Module):
    """NIMA model with a CLIP ViT-B/16 backbone (2D, expects `[B, C, H, W]`)."""
    def __init__(self, num_bins_aesthetic, backbone='clip_vit_b16', dropout=0.0, feat_dim=256):
        super(NIMA, self).__init__()

        self.backbone_name = backbone

        if backbone == 'clip_vit_b16':
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            self.backbone = clip_model.visual.float()
            backbone_out_features = 512
        else:
            raise ValueError(f"Backbone '{backbone}' is not supported. Choose from: clip_vit_b16")

        self.feat_dim = feat_dim

        def _drop():
            return nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.feat_proj = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            _drop(),
            nn.Linear(512, feat_dim),
        )
        self.fc_aesthetic = nn.Linear(feat_dim, num_bins_aesthetic)

    def freeze_backbone(self):
        """Freeze entire backbone. Only fc_aesthetic remains trainable."""
        backbone = self.backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    def _set_frozen_modules_eval(self):
        """Ensure frozen backbone stays in eval mode during model.train()."""
        backbone = self.backbone
        if not any(p.requires_grad for p in backbone.parameters()):
            backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._set_frozen_modules_eval()
        return self

    def forward(self, x, return_feat=False):
        # Expects `[B, C, H, W]`.
        raw_feat = self.backbone(x)
        domain_feat = self.feat_proj(raw_feat)
        aesthetic_logits = self.fc_aesthetic(domain_feat)
        if return_feat:
            return aesthetic_logits, domain_feat, raw_feat
        return aesthetic_logits


# ─── PIAA Models ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.drop is not None:
            x = self.drop(x)
        return self.fc2(x)


class InternalInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(InternalInteraction, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, attribute_embeddings):
        batch_size, num_attributes, _ = attribute_embeddings.shape
        combined = attribute_embeddings.unsqueeze(2) * attribute_embeddings.unsqueeze(1)
        out = self.mlp(combined.view(batch_size, num_attributes * num_attributes, self.input_dim))
        aggregated_interactions = out.view(batch_size, num_attributes, num_attributes, self.input_dim).sum(dim=1)
        return aggregated_interactions


class ExternalInteraction(nn.Module):
    def __init__(self):
        super(ExternalInteraction, self).__init__()

    def forward(self, user_attributes, image_attributes):
        interaction_results = user_attributes.unsqueeze(2) * image_attributes.unsqueeze(1)
        aggregated_interactions_user = torch.sum(interaction_results, dim=2)
        aggregated_interactions_img = torch.sum(interaction_results, dim=1)
        return aggregated_interactions_user, aggregated_interactions_img


class Interfusion_GRU(nn.Module):
    def __init__(self, input_dim=64):
        super(Interfusion_GRU, self).__init__()
        self.gru = nn.GRUCell(input_dim, input_dim)

    def forward(self, initial_node, internal_interaction, external_interaction):
        num_attr = initial_node.shape[1]
        results = []
        for i in range(num_attr):
            fused_node = self.gru(initial_node[:, i], None)
            fused_node = self.gru(internal_interaction[:, i], fused_node)
            fused_node = self.gru(external_interaction[:, i], fused_node)
            results.append(fused_node)
        return torch.stack(results, dim=1)


class PIAA_ICI(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, genres, backbone_dict, input_dim=64, hidden_size=256, dropout=None):
        super(PIAA_ICI, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.genres = genres
        self.input_dim = input_dim

        self.nima_dict = nn.ModuleDict()
        for genre in genres:
            backbone_type = backbone_dict.get(genre, 'clip_vit_b16')
            self.nima_dict[genre] = NIMA(num_bins, backbone_type, dropout=dropout if dropout else 0.0)

        self.internal_interaction_img = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size, dropout=dropout if dropout else 0.0)
        self.internal_interaction_user = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size, dropout=dropout if dropout else 0.0)
        self.external_interaction = ExternalInteraction()

        self.interfusion_img = Interfusion_GRU(input_dim=input_dim)
        self.interfusion_user = Interfusion_GRU(input_dim=input_dim)

        _dropout = dropout if dropout else 0.0
        self.node_attr_user = MLP(num_pt, hidden_size, num_attr * input_dim, dropout=_dropout)
        self.node_attr_img = MLP(num_attr + input_dim, hidden_size, num_attr * input_dim, dropout=_dropout)

        self.attr_corr = nn.Linear(input_dim, 1)
        self.direct_fc = nn.Linear(num_bins, 1)

        self.backbone_image_proj = nn.ModuleDict()
        for genre in genres:
            backbone_type = backbone_dict.get(genre, 'clip_vit_b16')
            in_dim = _BACKBONE_OUT_DIM[backbone_type]
            self.backbone_image_proj[genre] = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dim),
            )

    def freeze_backbone(self):
        for genre, nima in self.nima_dict.items():
            for param in nima.backbone.parameters():
                param.requires_grad = False
            nima.backbone.eval()
            for param in nima.feat_proj.parameters():
                param.requires_grad = False
            nima.feat_proj.eval()
            for param in nima.fc_aesthetic.parameters():
                param.requires_grad = False
            nima.fc_aesthetic.eval()

    def _set_frozen_modules_eval(self):
        for genre, nima in self.nima_dict.items():
            if not any(p.requires_grad for p in nima.backbone.parameters()):
                nima.backbone.eval()
            if not any(p.requires_grad for p in nima.feat_proj.parameters()):
                nima.feat_proj.eval()
            if not any(p.requires_grad for p in nima.fc_aesthetic.parameters()):
                nima.fc_aesthetic.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._set_frozen_modules_eval()
        return self

    def forward(self, images, personal_traits, image_attributes, genre, return_feat=False):
        logit, _, raw_feat = self.nima_dict[genre](images, return_feat=True)
        prob = F.softmax(logit, dim=1)

        n_attr = image_attributes.shape[1]
        img_feat = self.backbone_image_proj[genre](raw_feat)
        img_input = torch.cat([image_attributes, img_feat], dim=1)
        attr_img = self.node_attr_img(img_input).view(-1, n_attr, self.input_dim)
        attr_user = self.node_attr_user(personal_traits).view(-1, n_attr, self.input_dim)

        internal_img = self.internal_interaction_img(attr_img)
        internal_user = self.internal_interaction_user(attr_user)
        aggregated_interactions_user, aggregated_interactions_img = self.external_interaction(attr_user, attr_img)

        fused_features_img = self.interfusion_img(attr_img, internal_img, aggregated_interactions_img)
        fused_features_user = self.interfusion_user(attr_user, internal_user, aggregated_interactions_user)

        I_ij = torch.sum(fused_features_img, dim=1, keepdim=False) + torch.sum(fused_features_user, dim=1, keepdim=False)
        interaction_outputs = self.attr_corr(I_ij)
        direct_outputs = self.direct_fc(prob)
        if return_feat:
            return interaction_outputs + direct_outputs, I_ij
        return interaction_outputs + direct_outputs


class PIAA_MIR(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, genres, backbone_dict, input_dim=64, hidden_size=256, dropout=None):
        super(PIAA_MIR, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.genres = genres

        self.nima_dict = nn.ModuleDict()
        for genre in genres:
            backbone_type = backbone_dict.get(genre, 'clip_vit_b16')
            self.nima_dict[genre] = NIMA(num_bins, backbone_type)

        primary_backbone = backbone_dict.get(genres[0], 'clip_vit_b16')
        in_dim = _BACKBONE_OUT_DIM[primary_backbone]
        self.backbone_image_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

        interaction_input_dim = (num_attr + input_dim) * num_pt
        _dropout = dropout if dropout else 0.0
        self.interaction_mlp = MLP(
            interaction_input_dim, 512, input_dim, dropout=_dropout)
        self.attr_corr = nn.Linear(input_dim, 1)
        self.direct_fc = nn.Linear(num_bins, 1)

    def freeze_backbone(self):
        for genre, nima in self.nima_dict.items():
            for param in nima.backbone.parameters():
                param.requires_grad = False
            nima.backbone.eval()
            for param in nima.feat_proj.parameters():
                param.requires_grad = False
            nima.feat_proj.eval()
            for param in nima.fc_aesthetic.parameters():
                param.requires_grad = False
            nima.fc_aesthetic.eval()

    def _set_frozen_modules_eval(self):
        for genre, nima in self.nima_dict.items():
            if not any(p.requires_grad for p in nima.backbone.parameters()):
                nima.backbone.eval()
            if not any(p.requires_grad for p in nima.feat_proj.parameters()):
                nima.feat_proj.eval()
            if not any(p.requires_grad for p in nima.fc_aesthetic.parameters()):
                nima.fc_aesthetic.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._set_frozen_modules_eval()
        return self

    def forward(self, images, personal_traits, image_attributes, genre, return_feat=False):
        logit, _, raw_feat = self.nima_dict[genre](images, return_feat=True)
        prob = F.softmax(logit, dim=1)

        img_feat = self.backbone_image_proj(raw_feat)
        img_input = torch.cat([image_attributes, img_feat], dim=1)

        A_ij = img_input.unsqueeze(2) * personal_traits.unsqueeze(1)
        A_flat = A_ij.view(images.size(0), -1)

        I_ij = self.interaction_mlp(A_flat)
        interaction_outputs = self.attr_corr(I_ij)
        direct_outputs = self.direct_fc(prob)
        if return_feat:
            return interaction_outputs + direct_outputs, I_ij
        return interaction_outputs + direct_outputs


# ─── Checkpoint helpers ───────────────────────────────────────────────────────
#
# During PIAA finetuning `freeze_backbone()` freezes every submodule under
# `nima_dict` (backbone / feat_proj / fc_aesthetic), and the optimizer only ever
# receives parameters with requires_grad=True. Those tensors therefore stay
# bit-identical to the pretrain checkpoint for every user, so storing them once
# per user costs ~346 MB of duplicated weights against ~7 MB that actually
# changed. `trainable_state()` keeps only the part that moved; the frozen half
# is restored from the pretrain checkpoint at load time.

FROZEN_STATE_PREFIX = 'nima_dict.'


def trainable_state(state_dict):
    """Return the entries of `state_dict` that PIAA finetuning can modify."""
    return {k: v for k, v in state_dict.items() if not k.startswith(FROZEN_STATE_PREFIX)}


def is_trainable_only_state(state_dict):
    """True when a checkpoint holds only the trainable half (no frozen backbone)."""
    return not any(k.startswith(FROZEN_STATE_PREFIX) for k in state_dict)


def build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args):
    """Instantiate PIAA_ICI or PIAA_MIR based on args.model_type."""
    if args.model_type == 'MIR':
        return PIAA_MIR(
            num_bins, num_attr, num_pt, genres, backbone_dict,
            dropout=args.dropout)
    else:
        return PIAA_ICI(
            num_bins, num_attr, num_pt, genres, backbone_dict,
            dropout=args.dropout)


def discover_folds(root_dir, version_prefix):
    """Discover all fold directories for a given version prefix (e.g., 'v3' -> ['v3_fold1', ...])."""
    split_dir = os.path.join(root_dir, 'split')
    folds = sorted([
        d for d in os.listdir(split_dir)
        if d.startswith(f'{version_prefix}_fold') and os.path.isdir(os.path.join(split_dir, d))
    ])
    return folds
