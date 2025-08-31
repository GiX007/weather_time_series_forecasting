# src/nn_models.py
#
# Neural network forecasting models for weather time series:
# - Vanilla RNN
# - Attention-based model
# - Transformer Encoder
#
import math, time
import numpy as np
import torch
import torch.nn as nn
from src.utils import r2_fn, count_trainable_params


class RNN(nn.Module):
    """
    Vanilla RNN for sequence-to-one forecasting. Works for univariate (input_dim=1) or multivariate (e.g., input_dim=3) inputs, predicting a single target (output_dim=1).
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        h, _ = self.rnn(x) # h shape: (batch, seq_len, hidden_dim)
        last_h_t = h[:, -1, :] # take output from the last time step (hidden state)
        out = self.fc(last_h_t) # project the last hidden state (summary of the entire sequence) to the target dimension (e.g., 1-step forecast)
        return out


class AttNN(nn.Module):
    """
    Minimal self-attention forecaster (sequence -> one) with input x of (B, S, input_dim) and output y of (B, 1).
    """
    def __init__(self, input_dim=1, model_dim=64, num_heads=1, num_layers=1, seq_len=24, dropout=0.0, pool="mean"):
        super().__init__()
        assert model_dim % num_heads == 0   # ensure heads divide model_dim evenly
        self.pool = pool  # "last" or "mean" pooling of the sequence

        # linear projection to model dim and simple learnable positional embeddings
        self.in_proj = nn.Linear(input_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, model_dim))

        # stack of attention blocks (MultiheadAttention + FFN)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True), # self-attention
                "norm1": nn.LayerNorm(model_dim),
                "ffn": nn.Sequential(nn.Linear(model_dim, model_dim * 2), nn.ReLU(), nn.Linear(model_dim * 2, model_dim)), # feed-forward net
                "norm2": nn.LayerNorm(model_dim),
            }) for _ in range(num_layers)
        ])

        # final output head
        self.out = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        x:  (B, S, input_dim), y:  (B, 1)
        """
        h = self.in_proj(x) # project to model_dim (B, S, E)
        h = h + self.pos[:, :h.size(1), :] # # add positional info (B, S, E)

        for blk in self.layers:

            # create causal mask: shape (S, S), True means "blocked"
            # L = h.size(1)
            # causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1)
            # attn_out, _ = blk["attn"](h, h, h, attn_mask=causal_mask) # apply self-attention with causal mask

            attn_out, _ = blk["attn"](h, h, h)  # bidirectional self-attention by default
            h = blk["norm1"](h + attn_out) # residual + norm
            f = blk["ffn"](h) # feed-forward
            h = blk["norm2"](h + f) # residual + norm

        # pool sequence into single vector
        pooled = h[:, -1, :] if self.pool == "last" else h.mean(dim=1) # (B, E)
        y = self.out(pooled) # map to output (B, 1)
        return y


def get_positional_encoding(seq_len, d_model):
    # standard sinusoidal positional encoding (fixed, not trainable)
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe # (L, d_model)

class TransformerEncoder(nn.Module):
    """
    Bidirectional Transformer-Encoder (sequence -> one) with input x of (N, L) or (N, L, input_dim) and output y of (N,) (single-step regression).
    """
    def __init__(self, seq_len, input_dim=1, d_model=64, num_heads=2, ffn_dim=128, num_layers=1, dropout=0.1, pool="mean"):
        super().__init__()
        self.seq_len = seq_len
        self.pool = pool # "cls" (default) or "mean"

        # project input_dim (1 or 3) to model dimension (d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional encoding (fixed, not trainable)
        pe = get_positional_encoding(seq_len, d_model)  # (L, d_model)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # (1, L, d_model)

        # special [CLS] token (learnable vector summarizing the whole sequence)
        # self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # transformer-encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True) # tensors as (N, L, D), activation="relu"
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # output head
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # accept (N, L) or (N, L, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (N, L, 1)

        h = self.input_proj(x) # (N, L, d_model)
        pos = self.pos_enc[:, :h.size(1), :] # (1, L, d_model) in case runtime L <= seq_len
        h = h + pos  # add positional info

        # bidirectional encoder (tokens see each other in both directions)
        h = self.encoder(h) # (N, L, d_model)

        # mean pooling instead of CLS
        if self.pool == "last":
            agg = h[:, -1, :] # (N, d_model)
        else:  # "mean"
            agg = h.mean(dim=1) # (N, d_model)

        y = self.output_layer(agg) # (N, 1)

        return y


def trainer(model, model_name, train_loader, val_loader, epochs=200, lr=1e-3, weight_decay=1e-4, patience=10, verbose_every=10, max_grad_norm=1.0):
    """
    Universal trainer for sequence->one regressors (RNN / Attention / Transformer-Encoder).

    Args:
      - model: an instance of nn_mod_att
      - train_loader: PyTorch DataLoader for training data
      - val_loader: PyTorch DataLoader for validation data
      - epochs: int, number of training epochs
      - lr: float, learning rate
      - weight_decay: float, L2 regularization strength
      - patience: int, early stopping patience
      - verbose_every: int, print training progress every n epochs
      - max_grad_norm: float, maximum gradient norm for gradient clipping
      - model__name: str, name of model

  Returns:
      - history: dictionary with training metrics, trained model, time, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, threshold=1e-4)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()

    history = {"train_loss": [], "val_loss": [], "train_r2": [],   "val_r2": []}

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    stopped_epoch = epochs

    header = f">>> Training {model_name} ..."
    with open("results/nn_training_processes.txt", "a", encoding="utf-8") as f:
        f.write("\n" + header + "\n")

    start_time = time.time()
    for epoch in range(epochs):
        # train step
        model.train()
        total_loss = 0.0
        tr_preds, tr_targets = [], []

        for X, y in train_loader:
            # X: (B, L) or (B, L, F), y: (B, 1) or (B,)
            X, y = X.to(device), y.to(device)
            if X.dim() == 2:  # (B, L) -> (B, L, 1)
                X = X.unsqueeze(-1)

            y_pred = model(X) # -> (B, 1)
            # y_pred = out[0] if isinstance(out, tuple) else out # in case of return (y, attn)
            y_pred = y_pred.squeeze(-1) # (B,1)->(B,)
            y = y.squeeze(-1) # (B, 1) -> (B,)

            loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            total_loss += loss.item()
            tr_preds.extend(y_pred.detach().cpu().numpy())
            tr_targets.extend(y.detach().cpu().numpy())

        avg_train_loss = total_loss / max(1, len(train_loader))
        tr_r2 = r2_fn(tr_targets, tr_preds)

        # val step
        model.eval()
        val_total_loss = 0.0
        va_preds, va_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                if X.dim() == 2:
                    X = X.unsqueeze(-1)

                y_pred = model(X)
                y_pred = y_pred.squeeze(-1)
                y = y.squeeze(-1)

                loss = loss_fn(y_pred, y)
                val_total_loss += loss.item()

                va_preds.extend(y_pred.detach().cpu().numpy())
                va_targets.extend(y.detach().cpu().numpy())

        avg_val_loss = val_total_loss / max(1, len(val_loader))
        va_r2 = r2_fn(va_targets, va_preds)

        # step the scheduler
        scheduler.step(avg_val_loss)
        # print("lr:", optimizer.param_groups[0]["lr"])

        # log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_r2"].append(tr_r2)
        history["val_r2"].append(va_r2)

        if (epoch + 1) % verbose_every == 0:
            line = f"Epoch {epoch+1}/{epochs} | Train: loss={avg_train_loss:.4f} | R2={tr_r2:.4f} | Val: loss={avg_val_loss:.4f}| R2={va_r2:.4f}"

            # append to the log file
            with open("results/nn_training_processes.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")

        # early stop on val loss
        if avg_val_loss < best_val_loss - 1e-8:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped_epoch = epoch + 1
                break

    total_time = time.time() - start_time
    total_params = count_trainable_params(model.parameters())
    # print(f"\nTotal training time: {total_time:.4f}s")
    # print(f"Total trainable parameters: {total_params}")

    # restore best weights
    model.load_state_dict(best_state)

    history.update({
        "params": best_state,
        "model": model,
        "total_training_time": round(total_time, 4),
        "total_params": total_params,
        "best_val_loss": float(best_val_loss),
        "stopped_epoch": stopped_epoch,
    })
    return history


# Univariate prediction with autoregression
@torch.no_grad()
def predict_nn_mod(model, test_loader, n_steps=1):
    """
    Rolling-forecast predictions for univariate models. Model must accept (B, L, 1) and return (B, 1).  This does a within-origin recursive rollout: at each step we feed the previous prediction back into the input window to produce the next step (closed loop). It does NOT refit or
    update the model across origins (like we did in ARs), each test window comes from the DataLoader.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        n_steps: int, number of steps ahead to predict (>=1)

    Returns:
        preds_last: (N,) last-step predictions
        trues_last: (N,) or None
        preds_rollout: (N, n_steps)
        trues_rollout: (N, n_steps) or None
    """
    model.eval()
    device = next(model.parameters()).device

    preds_last, trues_last = [], []
    preds_rollout, trues_rollout = [], []

    for batch in test_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            Xb, yb = batch
            yb = yb.to(device)
        else:
            Xb, yb = batch, None

        Xb = Xb.to(device) # (B, L) or (B, L, 1)
        if Xb.dim() == 2: # (B, L) -> (B, L, 1)
            Xb = Xb.unsqueeze(-1)

        B = Xb.size(0)

        for i in range(B):
            seq = Xb[i:i+1].clone() # (1, L, 1)
            step_preds = []

            for _ in range(n_steps):
                y_hat = model(seq) # (1, 1)
                y_val = float(y_hat.squeeze().item())
                step_preds.append(y_val)

                # autoregressive feedback (append prediction)
                next_in = seq.new_tensor([[[y_val]]]) # same dtype/device as seq, shape (1, 1, 1)
                seq = torch.cat([seq[:, 1:], next_in], dim=1) # (1, L, 1)

            preds_rollout.append(step_preds)
            preds_last.append(step_preds[-1])

            if yb is not None:
                yi = yb[i].view(-1) # (H,)
                if yi.numel() >= n_steps:
                    tgt = yi[:n_steps].detach().cpu().numpy().astype("float32").tolist()
                else:
                    tgt = yi.detach().cpu().numpy().astype("float32").tolist()
                trues_rollout.append(tgt)
                trues_last.append(float(tgt[-1]))

    preds_last = np.asarray(preds_last, dtype="float32")
    trues_last = np.asarray(trues_last, dtype="float32") if trues_last else None
    preds_rollout = np.asarray(preds_rollout, dtype="float32")
    trues_rollout = np.asarray(trues_rollout, dtype="float32") if trues_rollout else None

    return preds_last, trues_last, preds_rollout, trues_rollout


# Multivariate rolling predictions: X = (B, L, 3) as [SWDR, rh, T], target_idx is last dim
@torch.no_grad()
def predict_nn_mod_mv(model, test_loader, n_steps=1):
    """
    Rolling forecast for multivariate inputs with T as target (last dim). Only T is auto-regressed, exogenous features (SWDR, rh) are frozen by repeating their last observed values.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test data (X: (B, L, F), F features [SWDR, rh, T] and y: (B,) or (B, H) target T)
        n_steps: int, number of steps ahead to predict (>=1)

    Returns:
        preds_last: (N,) last-step predictions
        trues_last: (N,) or None
        preds_rollout: (N, n_steps)
        trues_rollout: (N, n_steps) or None
    """
    model.eval()
    device = next(model.parameters()).device

    preds_last, trues_last = [], []
    preds_rollout, trues_rollout = [], []

    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        B, L, F = Xb.shape
        tgt = F - 1 # index of target (last feature, T)
        exo_idx = [0, 1] # indices of exogenous features (SWDR, rh)

        for i in range(B):
            seq = Xb[i:i+1].clone() # one sample of (1, L, F), e.g. (1, seq_len=48, 3)
            step_preds = []
            last_exo = seq[:, -1, exo_idx].clone() # last observations of the sequence (1, F-1), we reuse them at each forecast step

            for _ in range(n_steps):
                y_hat = model(seq) # -> (1,1)
                y_val = float(y_hat.squeeze().item()) # scalar
                step_preds.append(y_val)

                # next timestep (1,1,F)
                next_vec = seq.new_zeros(1, 1, F)

                # keep SWDR & rh fixed by reusing their last observed values (test set)
                next_vec[:, :, exo_idx] = last_exo.unsqueeze(1)

                # insert the newly predicted T value at the target position (last feature)
                next_vec[:, :, tgt] = y_hat.reshape(1, 1)

                # slide window by dropping the first timestep ([seq[:, 1:]), add the next_vec vertically (time dim=1)
                seq = torch.cat([seq[:, 1:], next_vec], dim=1) # (1, L, F)

            preds_rollout.append(step_preds)
            preds_last.append(step_preds[-1])

            yi = yb[i].view(-1) # (H,) or (1,)
            tgt_true = (yi[:n_steps] if yi.numel() >= n_steps else yi).detach().cpu().numpy().astype("float32").tolist()
            trues_rollout.append(tgt_true)
            trues_last.append(float(tgt_true[-1]))

    preds_last = np.asarray(preds_last, dtype="float32")
    trues_last = np.asarray(trues_last, dtype="float32") if trues_last else None
    preds_rollout = np.asarray(preds_rollout, dtype="float32")
    trues_rollout = np.asarray(trues_rollout, dtype="float32") if trues_rollout else None

    return preds_last, trues_last, preds_rollout, trues_rollout
