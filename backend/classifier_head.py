import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# =========================================================================
# 1. LEGACY ARCHITECTURE (For Backward Compatibility)
# This is the original class, renamed. It will be used to load models
# trained before the temporal delta update. No changes are needed here.
# =========================================================================
class ClassifierLegacyLSTM(nn.Module):
    def __init__(self, in_features, out_features, seq_len=31):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.lin1, self.lin0 = nn.Linear(in_features, out_features), nn.Linear(in_features, 256)
        self.lin2 = nn.Linear(128, out_features)
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.lstm = nn.LSTM(256, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.hsl, self.sw = seq_len // 2, 5

    def forward_linear(self, x):
        x_proj = self.lin1(x)
        windowed_logits = x_proj[:, self.hsl - self.sw : self.hsl + self.sw + 1, :]
        return windowed_logits.mean(dim=1)

    def forward_lstm(self, x):
        lstm_out, _ = self.lstm(x)
        center_window_raw_lstm = lstm_out[:, self.hsl - self.sw : self.hsl + self.sw + 1, :]
        avg_latent = center_window_raw_lstm.mean(dim=1)
        logits = self.lin2(avg_latent)
        return logits, avg_latent

    def forward(self, x):
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        amount = random.randint(64, 256)
        rand_inds = torch.randperm(x.size(2))[:amount]
        x[:, :, rand_inds] = torch.randn_like(x[:, :, rand_inds]).to(x.device)
        linear_logits = self.forward_linear(x)
        x_lstm = self.lin0(x)
        x_lstm = x_lstm - x_lstm.mean(dim=1, keepdim=True)
        lstm_logits, rawm = self.forward_lstm(x_lstm)
        return lstm_logits, linear_logits, rawm

    def forward_nodrop(self, x):
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        linear_logits = self.forward_linear(x)
        x_lstm = self.lin0(x)
        x_lstm = x_lstm - x_lstm.mean(dim=1, keepdim=True)
        lstm_logits, _ = self.forward_lstm(x_lstm)
        return lstm_logits + linear_logits

# =========================================================================
# 2. NEW PRODUCTION ARCHITECTURE (Temporal Deltas)
# It will be the new default for all models trained going forward.
# =========================================================================
class ClassifierLSTMDeltas(nn.Module):
    """
    LSTM classification head with temporal deltas.
    Incorporates all colleague feedback for robustness, stability, and performance.
    """
    def __init__(self, in_features, out_features, seq_len=31, bottleneck_dim=128,
                 dropout_p=0.15, use_acceleration=True, ema_alpha=0.3, center_window_size=5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.seq_len, self.sw = seq_len, center_window_size
        self.hsl = seq_len // 2
        self.use_acceleration = use_acceleration
        self.ema_alpha = ema_alpha

        self.cls_bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), nn.GELU(), nn.Dropout(0.1))
        self.delta_bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), nn.GELU(), nn.Dropout(0.1))
        if self.use_acceleration:
            self.acc_bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), nn.GELU(), nn.Dropout(0.1))

        self.cls_ln = nn.LayerNorm(bottleneck_dim)
        self.delta_ln = nn.LayerNorm(bottleneck_dim)
        if self.use_acceleration:
            self.acc_ln = nn.LayerNorm(bottleneck_dim)

        augmented_features = bottleneck_dim * 3 if self.use_acceleration else bottleneck_dim * 2
        self.lin0 = nn.Sequential(
            nn.Linear(augmented_features, 256),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        
        self.gate = nn.Parameter(torch.tensor(0.2))

        self.attention_head = nn.Linear(128, 1)
        self.attention_temp = nn.Parameter(torch.tensor(1.0))

        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(128, out_features)
        self.lstm = nn.LSTM(256, 64, num_layers=1, batch_first=True, bidirectional=True)

    def _calculate_robust_deltas(self, x_seq):
        """Helper to compute smoothed, reflection-padded temporal deltas."""
        B, T, C = x_seq.shape
        x_fp32 = x_seq.float()
        x_smooth = torch.zeros_like(x_fp32)
        x_smooth[:, 0, :] = x_fp32[:, 0, :]
        for t in range(1, T):
            x_smooth[:, t, :] = torch.lerp(x_smooth[:, t - 1, :], x_fp32[:, t, :], self.ema_alpha)

        mode = 'reflect' if T >= 3 else 'replicate'
        padded = F.pad(x_smooth.permute(0, 2, 1), (2, 0), mode).permute(0, 2, 1)

        dx = padded[:, 1:] - padded[:, :-1]
        ddx = dx[:, 1:] - dx[:, :-1]
        
        return x_smooth.to(x_seq.dtype), dx[:, 1:].to(x_seq.dtype), ddx.to(x_seq.dtype)

    def forward_linear(self, x):
        """Processes the smoothed CLS stream with the linear branch."""
        L = x.size(1)
        l, r = max(0, self.hsl - self.sw), min(L, self.hsl + self.sw + 1)
        if l >= r:
            idx = min(max(0, L // 2), L - 1) if L > 0 else 0
            return self.lin1(x[:, idx, :])
        
        windowed_features = x[:, l:r, :]
        windowed_logits = self.lin1(windowed_features)
        return windowed_logits.mean(dim=1)

    def forward_lstm(self, x):
        """Processes the augmented stream with the LSTM and attention head."""
        lstm_out, _ = self.lstm(x)
        L = lstm_out.size(1)
        l, r = max(0, self.hsl - self.sw), min(L, self.hsl + self.sw + 1)
        if l >= r:
            idx = min(max(0, L // 2), L - 1) if L > 0 else 0
            return self.lin2(lstm_out[:, idx, :]), lstm_out[:, idx, :]

        center_window_raw_lstm = lstm_out[:, l:r, :]
        
        temp = F.softplus(self.attention_temp) + 1e-3
        scores = self.attention_head(center_window_raw_lstm).squeeze(-1) / temp
        attention_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attended_latent = (attention_weights * center_window_raw_lstm).sum(dim=1)
        
        logits = self.lin2(attended_latent)
        return logits, attended_latent

    def forward(self, x):
        """Unified forward pass. Mode is controlled externally by model.train() or model.eval()."""
        cls_stream, delta_stream, acc_stream = self._calculate_robust_deltas(x)
        linear_logits = self.forward_linear(cls_stream)

        cls_b = self.cls_ln(self.cls_bottleneck(cls_stream))
        delta_b = self.delta_ln(self.delta_bottleneck(delta_stream))

        if self.use_acceleration:
            acc_b = self.acc_ln(self.acc_bottleneck(acc_stream))
            x_augmented = torch.cat([cls_b, delta_b, acc_b], dim=-1)
        else:
            x_augmented = torch.cat([cls_b, delta_b], dim=-1)

        x_lstm = self.lin0(x_augmented)
        
        mean_fp32 = x_lstm.mean(dim=1, keepdim=True, dtype=torch.float32)
        x_lstm_centered = (x_lstm - mean_fp32).to(x_lstm.dtype)
        
        lstm_logits, rawm = self.forward_lstm(x_lstm_centered)

        final_logits = torch.lerp(linear_logits, lstm_logits, torch.sigmoid(self.gate))
        return final_logits, rawm