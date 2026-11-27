"""Frozen document encoders (Section 3.2.1).

* Semantic: google-bert/bert-base-chinese (revision 8f23c25b).  Title + cleaned
  body are split into non-overlapping blocks of at most 510 content tokens;
  [CLS]/[SEP] are added for inference but excluded from mean pooling; block
  vectors are weighted by content-token count -> one 768-d document vector.
* Sentiment (CSI): yiyanghkust/finbert-tone-chinese (revision e91b1a3), label
  order 0 = neutral, 1 = positive, 2 = negative, no fine-tuning.  Chunk
  probabilities are weighted by content-token count -> document probabilities.

Both models stay frozen: they run under ``torch.no_grad`` and never enter an optimiser.
The ``Hashing*`` classes are deterministic stand-ins used only by the synthetic smoke test.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..utils import get_logger

LOG = get_logger(__name__)


def _join(title: str, body: str) -> str:
    title, body = (title or "").strip(), (body or "").strip()
    return f"{title}\n{body}" if title and body else (title or body)


class _BlockEncoder:
    """Shared block construction and batched inference for BERT-style encoders."""

    def __init__(self, tokenizer, model, device, max_content: int = 510, batch_size: int = 16):
        import torch

        self.torch = torch
        self.tok = tokenizer
        self.tok.model_max_length = 10**9          # tokenise whole documents; blocks are cut manually
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_content = int(max_content)
        self.batch_size = int(batch_size)

    def blocks(self, titles: Sequence[str], bodies: Sequence[str]):
        """Yield (doc_index, content_token_ids) for every non-overlapping block."""
        for d, (t, b) in enumerate(zip(titles, bodies)):
            ids = self.tok(_join(t, b), add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if not ids:
                continue
            for start in range(0, len(ids), self.max_content):
                yield d, ids[start:start + self.max_content]

    def _batches(self, items):
        items = sorted(items, key=lambda x: len(x[1]))          # length bucketing for efficiency
        for s in range(0, len(items), self.batch_size):
            chunk = items[s:s + self.batch_size]
            width = max(len(ids) for _, ids in chunk) + 2
            input_ids = np.full((len(chunk), width), self.tok.pad_token_id, dtype=np.int64)
            attn = np.zeros((len(chunk), width), dtype=np.int64)
            content = np.zeros((len(chunk), width), dtype=np.float32)
            for r, (_, ids) in enumerate(chunk):
                seq = [self.tok.cls_token_id] + list(ids) + [self.tok.sep_token_id]
                input_ids[r, : len(seq)] = seq
                attn[r, : len(seq)] = 1
                content[r, 1 : 1 + len(ids)] = 1.0           # special tokens excluded from pooling
            yield chunk, input_ids, attn, content


class FrozenSemanticEncoder(_BlockEncoder):
    dim = 768

    @classmethod
    def from_pretrained(cls, name: str, revision: str | None, device, max_content=510, batch_size=16):
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModel.from_pretrained(name, revision=revision)
        return cls(tok, model, device, max_content, batch_size)

    def encode(self, titles: Sequence[str], bodies: Sequence[str]) -> np.ndarray:
        torch = self.torch
        n = len(titles)
        acc = np.zeros((n, self.dim), dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            for chunk, ids, attn, content in self._batches(list(self.blocks(titles, bodies))):
                out = self.model(input_ids=torch.from_numpy(ids).to(self.device),
                                 attention_mask=torch.from_numpy(attn).to(self.device)).last_hidden_state
                mask = torch.from_numpy(content).to(self.device).unsqueeze(-1)
                counts = mask.sum(1)
                vec = ((out * mask).sum(1) / counts.clamp_min(1.0)).double().cpu().numpy()
                for r, (d, block) in enumerate(chunk):
                    acc[d] += len(block) * vec[r]                # content-token weighting
                    weight[d] += len(block)
        ok = weight > 0
        acc[ok] /= weight[ok, None]
        return acc.astype(np.float32)


class FrozenSentimentEncoder(_BlockEncoder):
    """Returns document probabilities in the fixed order (neutral, positive, negative)."""

    @classmethod
    def from_pretrained(cls, name: str, revision: str | None, device, label_index: dict, max_content=510, batch_size=16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision)
        LOG.info("sentiment id2label=%s; configured order=%s", getattr(model.config, "id2label", None), label_index)
        enc = cls(tok, model, device, max_content, batch_size)
        enc.order = [int(label_index["neutral"]), int(label_index["positive"]), int(label_index["negative"])]
        return enc

    def encode(self, titles: Sequence[str], bodies: Sequence[str]) -> np.ndarray:
        torch = self.torch
        n = len(titles)
        acc = np.zeros((n, 3), dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            for chunk, ids, attn, _ in self._batches(list(self.blocks(titles, bodies))):
                logits = self.model(input_ids=torch.from_numpy(ids).to(self.device),
                                    attention_mask=torch.from_numpy(attn).to(self.device)).logits
                prob = torch.softmax(logits.double(), dim=-1).cpu().numpy()[:, self.order]
                for r, (d, block) in enumerate(chunk):
                    acc[d] += len(block) * prob[r]
                    weight[d] += len(block)
        ok = weight > 0
        acc[ok] /= weight[ok, None]
        acc[~ok] = np.array([1.0, 0.0, 0.0])
        return acc.astype(np.float32)


# --------------------------------------------------------------------------- smoke-test stand-ins
class HashingSemanticEncoder:
    """Deterministic character n-gram hashing features (smoke tests only; NOT the paper encoder)."""

    dim = 768

    def encode(self, titles: Iterable[str], bodies: Iterable[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer

        hv = HashingVectorizer(analyzer="char", ngram_range=(1, 2), n_features=self.dim, alternate_sign=True, norm="l2")
        return hv.transform([_join(t, b) for t, b in zip(titles, bodies)]).toarray().astype(np.float32)


class LexiconSentimentEncoder:
    """Small polarity lexicon -> (neutral, positive, negative) probabilities (smoke tests only)."""

    POS = ["上涨", "利好", "增长", "回暖", "活跃", "积极", "提升", "成交放量", "创新高", "支持"]
    NEG = ["下跌", "利空", "下降", "低迷", "萎缩", "风险", "违规", "处罚", "收紧", "创新低"]

    def encode(self, titles: Iterable[str], bodies: Iterable[str]) -> np.ndarray:
        out = []
        for t, b in zip(titles, bodies):
            text = _join(t, b)
            pos = sum(text.count(w) for w in self.POS)
            neg = sum(text.count(w) for w in self.NEG)
            logits = np.array([1.0, 0.6 * pos, 0.6 * neg])
            e = np.exp(logits - logits.max())
            out.append(e / e.sum())
        return np.asarray(out, dtype=np.float32)
