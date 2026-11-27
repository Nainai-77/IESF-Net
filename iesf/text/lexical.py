""""Lexical content" representation (Table 12): character n-gram TF-IDF + truncated SVD.

Character n-grams avoid a word segmenter for Chinese.  The vocabulary, IDF
weights and SVD basis are fitted on documents available within the training
period of the market only, then applied to every document.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class LexicalEncoder:
    def __init__(self, analyzer="char", ngram_range=(1, 2), max_features=50000, min_df=3, svd_dim=128, seed=2026):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram_range), max_features=max_features,
                                   min_df=min_df, sublinear_tf=True)
        self.svd_dim = int(svd_dim)
        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=seed)
        self.fitted_dim = self.svd_dim

    def fit(self, texts: Sequence[str]) -> "LexicalEncoder":
        x = self.vec.fit_transform(texts)
        n_comp = int(min(self.svd_dim, max(1, x.shape[1] - 1), max(1, x.shape[0] - 1)))
        if n_comp != self.svd_dim:
            from sklearn.decomposition import TruncatedSVD

            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.svd.random_state)
        self.svd.fit(x)
        self.fitted_dim = n_comp
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        z = self.svd.transform(self.vec.transform(texts))
        out = np.zeros((len(texts), self.svd_dim), dtype=np.float32)
        out[:, : z.shape[1]] = z
        return out
