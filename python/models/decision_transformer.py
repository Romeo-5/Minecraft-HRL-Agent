"""
decision_transformer.py

Decision Transformer for Minecraft skill-sequence planning.
(Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", 2021)

Architecture: GPT-style causal transformer that takes sequences of
(return-to-go, state, action) triples and predicts the next action.

Sized for our dataset (605 samples, max path length 15, 47-skill vocab):
  hidden_dim = 64
  n_layers   = 2
  n_heads    = 4
  dropout    = 0.1
  max_len    = 15   (sequence length in (RTG, state, action) triplets)

Input sequence for a path of length T (3T tokens total):
  [RTG_0, s_0, a_0,  RTG_1, s_1, a_1,  ...,  RTG_{T-1}, s_{T-1}, a_{T-1}]

Output:
  action_logits at each state-token position → shape (batch, T, action_dim)
  Loss: cross-entropy on predicted a_t vs ground truth a_t (masked for padding)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_encoder import STATE_DIM, ACTION_DIM, PAD_ACTION


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal (autoregressive) mask."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float, max_tokens: int):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv   = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask (lower-triangular): shape (1, 1, max_tokens, max_tokens)
        mask = torch.tril(torch.ones(max_tokens, max_tokens)).view(1, 1, max_tokens, max_tokens)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale           # (B, n_heads, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)        # (B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float, max_tokens: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout, max_tokens)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Minecraft skill planning.

    Args:
        state_dim:  Dimension of the state vector (default 41)
        act_dim:    Number of discrete skills (default 47)
        hidden_dim: Transformer hidden size (default 64)
        n_layers:   Number of transformer blocks (default 2)
        n_heads:    Number of attention heads (default 4)
        dropout:    Dropout probability (default 0.1)
        max_len:    Max sequence length in (RTG, state, action) triplets (default 15)
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        act_dim:    int = ACTION_DIM,
        hidden_dim: int = 64,
        n_layers:   int = 2,
        n_heads:    int = 4,
        dropout:    float = 0.1,
        max_len:    int = 15,
    ):
        super().__init__()

        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.hidden_dim = hidden_dim
        self.max_len    = max_len
        max_tokens      = 3 * max_len   # 3 tokens per timestep

        # ── Input projections ────────────────────────────────────────────────
        self.embed_rtg    = nn.Linear(1, hidden_dim)             # RTG scalar
        self.embed_state  = nn.Linear(state_dim, hidden_dim)     # state vector
        self.embed_action = nn.Embedding(act_dim + 1, hidden_dim,
                                         padding_idx=PAD_ACTION)  # +1 for pad token

        # Learned positional embeddings (one per token position)
        self.pos_emb = nn.Embedding(max_tokens, hidden_dim)

        self.drop = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(hidden_dim)

        # ── Transformer blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout, max_tokens)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        # ── Output head ──────────────────────────────────────────────────────
        # Applied to the output at state-token positions to predict next action
        self.action_head = nn.Linear(hidden_dim, act_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        states:        torch.Tensor,   # (B, T, state_dim)
        actions:       torch.Tensor,   # (B, T) — long, PAD_ACTION for padding
        rtgs:          torch.Tensor,   # (B, T, 1)
        attention_mask: torch.Tensor = None,  # (B, T) — True where valid
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            action_logits: (B, T, act_dim) — raw logits at each state-token position.
                           Compute loss only on non-padding positions.
        """
        B, T, _ = states.shape
        device = states.device

        # ── Embed each modality ──────────────────────────────────────────────
        rtg_emb    = self.embed_rtg(rtgs)                     # (B, T, H)
        state_emb  = self.embed_state(states)                 # (B, T, H)
        action_emb = self.embed_action(actions)               # (B, T, H)

        # ── Interleave into (RTG, state, action) token sequence ─────────────
        # Token layout: [r0, s0, a0, r1, s1, a1, ..., r_{T-1}, s_{T-1}, a_{T-1}]
        # Shape: (B, 3T, H)
        tokens = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (B, T, 3, H)
        tokens = tokens.reshape(B, 3 * T, self.hidden_dim)

        # ── Positional embeddings ────────────────────────────────────────────
        pos_ids = torch.arange(3 * T, device=device).unsqueeze(0)      # (1, 3T)
        tokens  = self.drop(self.ln_in(tokens + self.pos_emb(pos_ids)))

        # ── Transformer ──────────────────────────────────────────────────────
        x = tokens
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)   # (B, 3T, H)

        # ── Extract state-token outputs → action predictions ─────────────────
        # State tokens are at positions 1, 4, 7, ... (index 1 in each triplet)
        state_positions = torch.arange(1, 3 * T, 3, device=device)    # [1, 4, 7, ...]
        state_out = x[:, state_positions, :]                           # (B, T, H)

        action_logits = self.action_head(state_out)                    # (B, T, act_dim)
        return action_logits

    @torch.no_grad()
    def predict(
        self,
        state:           torch.Tensor,   # (state_dim,) current state
        rtg:             float,          # current return-to-go scalar
        history_states:  list,           # list of past state tensors (may be empty)
        history_actions: list,           # list of past action ints (may be empty)
        temperature:     float = 1.0,
        device:          str = "cpu",
    ) -> int:
        """
        Autoregressively predict the next skill index given history.

        Args:
            state:           Current game state vector (41,)
            rtg:             Current return-to-go (scalar)
            history_states:  List of previous state tensors (oldest first)
            history_actions: List of previous action indices (oldest first)
            temperature:     Sampling temperature (1.0 = greedy-ish, lower = greedier)
            device:          Torch device string

        Returns:
            int: Predicted skill index (into SKILL_VOCAB)
        """
        self.eval()
        T = len(history_actions) + 1
        T = min(T, self.max_len)

        # Build full history (trim to max_len)
        all_states  = history_states[-(T-1):]  + [state]
        all_actions = history_actions[-(T-1):]
        # Pad actions to length T (last position is unknown — use PAD)
        all_actions = all_actions + [PAD_ACTION]

        # RTG is constant for now (we don't recompute per step here)
        all_rtgs = [rtg] * T

        # Build tensors
        states_t  = torch.stack(all_states).unsqueeze(0).to(device)            # (1, T, 41)
        actions_t = torch.tensor(all_actions, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)
        rtgs_t    = torch.tensor(all_rtgs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, T, 1)

        logits = self.forward(states_t, actions_t, rtgs_t)  # (1, T, act_dim)
        # Take prediction at the last timestep
        last_logits = logits[0, -1, :] / temperature         # (act_dim,)
        probs = F.softmax(last_logits, dim=-1)
        return int(torch.argmax(probs).item())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
