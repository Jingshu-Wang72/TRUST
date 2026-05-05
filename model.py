import torch
import torch.nn as nn
import torch.nn.functional as F


FUSED_FEATURE_DIM = 128


def kl_divergence(alpha: torch.Tensor, n_classes: int) -> torch.Tensor:
    beta = torch.ones((1, n_classes), device=alpha.device, dtype=alpha.dtype)
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    s_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_b = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_b_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)
    return torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(s_alpha)), dim=1, keepdim=True) + ln_b + ln_b_uni


def evidential_ce_loss(target: torch.Tensor, alpha: torch.Tensor, n_classes: int, epoch: int, annealing_epoch: int) -> torch.Tensor:
    target = target.long()
    strength = torch.sum(alpha, dim=1, keepdim=True)
    evidence = alpha - 1.0
    one_hot = F.one_hot(target, num_classes=n_classes).to(alpha.dtype)
    fit = torch.sum(one_hot * (torch.digamma(strength) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing = min(1.0, float(epoch) / float(max(1, annealing_epoch)))
    adjusted = evidence * (1.0 - one_hot) + 1.0
    regularizer = annealing * kl_divergence(adjusted, n_classes)
    return torch.mean(fit + regularizer)


class MLPBranch(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TRUST(nn.Module):
    def __init__(
        self,
        view_dims: list[int],
        n_classes: int,
        hidden: list[int] | None = None,
        top_k: int = 5,
        use_pseudo: bool = True,
        use_pseudo_in_fusion: bool = True,
        min_warmup_epochs: int = 5,
        stable_topk_steps: int = 3,
        max_warmup_epochs: int = 40,
    ) -> None:
        super().__init__()
        hidden = hidden or [128, 64]
        self.view_dims = list(view_dims)
        self.n_classes = int(n_classes)
        self.n_views = len(view_dims)
        self.top_k = min(int(top_k), self.n_views)
        self.use_pseudo = bool(use_pseudo)
        self.use_pseudo_in_fusion = bool(use_pseudo_in_fusion)
        self.min_warmup_epochs = int(min_warmup_epochs)
        self.stable_topk_steps = int(stable_topk_steps)
        self.max_warmup_epochs = int(max_warmup_epochs)

        self.projections = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, FUSED_FEATURE_DIM), nn.ReLU())
            for dim in self.view_dims
        ])
        self.pseudo_projection = nn.Sequential(
            nn.BatchNorm1d(self.n_views * FUSED_FEATURE_DIM),
            nn.Linear(self.n_views * FUSED_FEATURE_DIM, FUSED_FEATURE_DIM),
            nn.ReLU(),
        )
        self.view_branches = nn.ModuleList([
            MLPBranch(dim + (FUSED_FEATURE_DIM if self.use_pseudo else 0), hidden, self.n_classes)
            for dim in self.view_dims
        ])
        # Every base view, including MiniROCKET/MultiROCKET/HYDRA, uses this same evidence DNN.
        self.pseudo_branch = MLPBranch(FUSED_FEATURE_DIM, hidden, self.n_classes)

        self.selected_views: list[int] | None = None
        self.selection_step = 0
        self.stable_count = 0
        self.warmup_complete = False

    @staticmethod
    def sign_sqrt_l2(x: torch.Tensor) -> torch.Tensor:
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)
        return F.normalize(x, p=2, dim=1)

    def build_pseudo(self, views: list[torch.Tensor]) -> torch.Tensor:
        projected = [proj(view) for proj, view in zip(self.projections, views)]
        pseudo = self.pseudo_projection(torch.cat(projected, dim=1))
        return self.sign_sqrt_l2(pseudo)

    def ds_combine_two(self, alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        strength1 = torch.sum(alpha1, dim=1, keepdim=True)
        strength2 = torch.sum(alpha2, dim=1, keepdim=True)
        evidence1 = alpha1 - 1.0
        evidence2 = alpha2 - 1.0
        belief1 = evidence1 / strength1
        belief2 = evidence2 / strength2
        uncertainty1 = self.n_classes / strength1
        uncertainty2 = self.n_classes / strength2

        bb = torch.bmm(belief1.view(-1, self.n_classes, 1), belief2.view(-1, 1, self.n_classes))
        conflict = torch.sum(bb, dim=(1, 2), keepdim=False) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        denom = (1.0 - conflict).view(-1, 1).clamp_min(1e-8)
        belief = (belief1 * belief2 + belief1 * uncertainty2 + belief2 * uncertainty1) / denom
        uncertainty = (uncertainty1 * uncertainty2) / denom
        strength = self.n_classes / uncertainty.clamp_min(1e-8)
        return belief * strength + 1.0

    def ds_combine(self, alphas: list[torch.Tensor]) -> torch.Tensor:
        if not alphas:
            raise ValueError("Expected at least one alpha tensor for fusion.")
        fused = alphas[0]
        for alpha in alphas[1:]:
            fused = self.ds_combine_two(fused, alpha)
        return fused

    def update_selected_views(self, uncertainties: torch.Tensor, epoch: int) -> list[int]:
        current = torch.topk(uncertainties, k=self.top_k, largest=False).indices.tolist()
        self.selection_step += 1
        if self.selected_views == current:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.selected_views = list(current)

        if epoch >= self.max_warmup_epochs:
            self.warmup_complete = True
        if epoch >= self.min_warmup_epochs and self.stable_count >= self.stable_topk_steps:
            self.warmup_complete = True
        return current

    def forward(self, views: list[torch.Tensor], epoch: int | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        if len(views) != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {len(views)}")

        pseudo = self.build_pseudo(views) if self.use_pseudo else None
        enhanced = [torch.cat([view, pseudo], dim=1) if pseudo is not None else view for view in views]
        view_logits = [branch(view) for branch, view in zip(self.view_branches, enhanced)]
        view_alphas = [F.softplus(logit) + 1.0 for logit in view_logits]

        uncertainties = torch.stack([
            torch.mean(self.n_classes / torch.sum(alpha, dim=1)) for alpha in view_alphas
        ])

        if self.training and epoch is not None:
            self.update_selected_views(uncertainties.detach(), epoch)

        if not self.warmup_complete or self.selected_views is None:
            selected = list(range(self.n_views))
        else:
            selected = list(self.selected_views)

        fusion_alphas = [view_alphas[idx] for idx in selected]
        if pseudo is not None and self.use_pseudo_in_fusion:
            pseudo_alpha = F.softplus(self.pseudo_branch(pseudo)) + 1.0
            fusion_alphas.append(pseudo_alpha)
        fused_alpha = self.ds_combine(fusion_alphas)
        return view_alphas, fused_alpha
