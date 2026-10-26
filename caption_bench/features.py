"""
caption_bench/features.py
Per-object features computed from one captured generation.

For an object whose first sub-token is generated at index j:

  Logit baselines (from the logits that produced token j)
    nll       -log p(token j)
    entropy   entropy of the next-token distribution at j

  Internal Confidence baseline (Jiang et al., ICLR 2025)
    ic[l]     max over image patches of p(token j) when the layer-l image
              hidden state is decoded with the LM head ("logit lens")

  Latent alignment (LatentTokenGuard and its variants)
    For every text layer lt and image layer li, the cosine similarity between
    the object token's state at lt and each image patch state at li, reduced to
    STAT_NAMES. Two text positions are kept:
      "at"   the state of token j itself (seq_states[j+1])
      "pre"  the state that predicted token j (seq_states[j])
    The max statistic at the object token is the object-level form used by
    ContextualLens (Phukan et al., NAACL 2025) and by the original
    LatentTokenGuard; top-k means, the global cosine and the spread
    statistics are the variants studied here.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from caption_bench.capture import Capture, LlavaCapturer, token_at_char
from caption_bench.chair import Mention

TOPK = (4, 8, 16, 32)
STAT_NAMES = ["max"] + [f"top{k}" for k in TOPK] + ["mean", "std", "global"]


def logit_features(logits: torch.Tensor, token_ids: List[int], positions: List[int]):
    """NLL and entropy at the given generation steps."""
    logp = F.log_softmax(logits[positions].float(), dim=-1)                 # (M, V)
    ids = torch.tensor([token_ids[j] for j in positions], device=logp.device)
    nll = -logp.gather(1, ids[:, None]).squeeze(1)
    entropy = -(logp.exp() * logp).sum(-1)
    return nll.cpu().numpy(), entropy.cpu().numpy()


def internal_confidence(capturer: LlavaCapturer, image_states: torch.Tensor,
                        token_ids: List[int], chunk: int = 128) -> np.ndarray:
    """
    (L+1, M): for each layer, max over image patches of the logit-lens
    probability of each requested token id.
    """
    norm, head = capturer.final_norm(), capturer.lm_head()
    ids = torch.tensor(token_ids, device=image_states.device)
    last = image_states.shape[0] - 1
    out = torch.empty(image_states.shape[0], len(token_ids))
    with torch.no_grad():
        for layer in range(image_states.shape[0]):
            # HF already applies the final norm to the last returned state
            states = image_states[layer]
            best = torch.full((len(token_ids),), -1.0, device=states.device)
            for s in range(0, states.shape[0], chunk):
                h = states[s:s + chunk]
                h = h if layer == last else norm(h)
                probs = F.softmax(head(h).float(), dim=-1)[:, ids]           # (chunk, M)
                best = torch.maximum(best, probs.max(0).values)
            out[layer] = best.cpu()
    return out.numpy()


def similarity_stats(text_states: torch.Tensor, image_states: torch.Tensor) -> np.ndarray:
    """
    text_states (Lt, d), image_states (Li, P, d) → (Lt, Li, len(STAT_NAMES)).
    """
    t = F.normalize(text_states.float(), dim=-1)
    v = F.normalize(image_states.float(), dim=-1)
    sims = torch.einsum("td,lpd->tlp", t, v)                                  # (Lt, Li, P)

    k_max = min(max(TOPK), sims.shape[-1])
    top = sims.topk(k_max, dim=-1).values                                     # sorted desc
    cum = top.cumsum(-1)
    stats = [top[..., 0]]
    stats += [cum[..., min(k, k_max) - 1] / min(k, k_max) for k in TOPK]
    stats += [sims.mean(-1), sims.std(-1)]

    g = F.normalize(image_states.float().mean(1), dim=-1)                     # (Li, d)
    stats.append(t @ g.T)                                                     # (Lt, Li)
    return torch.stack(stats, dim=-1).cpu().numpy()


def mention_features(capturer: LlavaCapturer, cap: Capture,
                     mentions: List[Mention]) -> Dict[str, np.ndarray]:
    """
    Features for every mention, in order. Returns arrays with leading dim M.
    Mentions whose first token is the last generated token have no "at"
    state; their "at" grid is filled with NaN.
    """
    positions = [token_at_char(cap.token_starts, m.start) for m in mentions]
    first_ids = [cap.token_ids[j] for j in positions]
    T = cap.seq_states.shape[0]

    nll, entropy = logit_features(cap.logits, cap.token_ids, positions)
    ic = internal_confidence(capturer, cap.image_states, first_ids).T          # (M, L+1)

    grids_at, grids_pre = [], []
    for j in positions:
        grids_pre.append(similarity_stats(cap.seq_states[j], cap.image_states))
        if j + 1 < T:
            grids_at.append(similarity_stats(cap.seq_states[j + 1], cap.image_states))
        else:
            grids_at.append(np.full_like(grids_pre[-1], np.nan))

    return {
        "token_index": np.array(positions, dtype=np.int32),
        "token_id": np.array(first_ids, dtype=np.int64),
        "nll": nll.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "ic": ic.astype(np.float32),
        "grid_at": np.stack(grids_at).astype(np.float32),
        "grid_pre": np.stack(grids_pre).astype(np.float32),
    }
