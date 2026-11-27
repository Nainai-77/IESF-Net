"""Synthetic raw data in the project schema — for the end-to-end smoke test only.

Produces ``raw/prices/{market}_observed_prices.csv`` and ``raw/documents.jsonl``
with irregular trading calendars, multiscale price dynamics with transient
reversals, news with polarity words and policy notices that contain category
anchors and operative expressions.  No result obtained on these data says
anything about the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_dir, write_jsonl

POLICY_TEMPLATES = [
    ("Allocation", "主管部门印发《碳排放权配额分配方案》", "根据方案，{y}年度配额总量{dir}，免费分配比例{dir2}，自{eff}起施行。"),
    ("Compliance", "关于做好碳排放配额清缴履约工作的通知", "重点排放单位应当于{eff}前完成配额清缴，未足额清缴的将依法处罚。"),
    ("Offsets", "关于国家核证自愿减排量抵消机制的通知", "控排企业可使用CCER抵消，抵消比例{dir}至不超过百分之十，自{eff}起实施。"),
    ("Coverage", "主管部门发布纳入碳排放权交易管理的企业名单", "新增行业纳入配额管理，纳入标准调整为年综合能耗一万吨标准煤，自{eff}起执行。"),
    ("Trading", "交易中心发布《碳排放权交易规则》修订通知", "交易中心决定调整涨跌幅限制，协议转让方式同步优化，自{eff}起实施。"),
    ("MRV", "关于开展温室气体排放报告核查工作的通知", "主管部门要求第三方核查机构加强数据质量管理，现场检查应当覆盖全部企业。"),
    ("Finance", "交易中心推出碳排放权质押融资业务", "交易中心联合银行开展配额质押融资试点，支持企业盘活碳资产，自{eff}起启动。"),
]
NEWS_POS = ["碳价上涨，市场交投活跃", "成交放量，碳价回暖", "利好政策推动碳价提升"]
NEWS_NEG = ["碳价下跌，市场低迷", "成交萎缩，碳价创新低", "市场风险上升，碳价下降"]
NEWS_NEU = ["今日碳市场成交情况公布", "碳排放权交易市场运行平稳", "碳市场周度数据发布"]


def make_market(name: str, product: str, n_obs: int, rng: np.random.Generator, start="2018-01-02"):
    days = pd.bdate_range(start, periods=int(n_obs * 1.25))
    keep = np.sort(rng.choice(len(days), size=n_obs, replace=False))
    dates = days[keep]
    t = np.arange(n_obs)
    trend = 30.0 + np.cumsum(rng.normal(0.02, 0.35, n_obs))
    cycles = 2.0 * np.sin(2 * np.pi * t / 45.0) + 1.0 * np.sin(2 * np.pi * t / 14.0 + 1.0)
    transient = np.zeros(n_obs)
    eps = rng.standard_t(4, n_obs) * 0.4
    for i in range(1, n_obs):
        transient[i] = -0.4 * transient[i - 1] + eps[i]          # partly reversed at the next observation
    close = np.maximum(np.round(trend + cycles + transient, 2), 1.0)
    volume = np.round(rng.lognormal(10.0, 1.0, n_obs))
    turnover = np.round(volume * close * rng.uniform(0.95, 1.05, n_obs))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "market": name, "product": product,
                         "close": close, "volume": volume, "turnover": turnover})


def make_documents(prices: dict[str, pd.DataFrame], rng: np.random.Generator) -> list[dict]:
    docs, k = [], 0
    for mk, df in prices.items():
        dates = pd.to_datetime(df["date"])
        change = np.concatenate([[0.0], np.diff(df["close"].to_numpy())])
        for i, d in enumerate(dates):
            if rng.random() < 0.12:
                continue                                         # dates without new text
            for _ in range(rng.poisson(1.5) + 1):
                pool = NEWS_POS if change[i] > 0.3 else NEWS_NEG if change[i] < -0.3 else NEWS_NEU
                title = pool[rng.integers(len(pool))]
                hh, mm = rng.integers(8, 22), rng.integers(0, 60)
                docs.append({"document_id": f"syn{k:07d}", "title": f"{mk}{title}",
                             "body": f"{title}。{mk}市场收盘价{df['close'].iloc[i]:.2f}元/吨。",
                             "publish_datetime": f"{d.date()} {hh:02d}:{mm:02d}:00", "date_only_publish_flag": 0,
                             "markets": [mk], "source_type": "synthetic_news", "publisher": "合成新闻"})
                k += 1
            if rng.random() < 0.025:
                cat, title, body = POLICY_TEMPLATES[rng.integers(len(POLICY_TEMPLATES))]
                eff = (d + pd.Timedelta(days=int(rng.integers(-10, 90)))).strftime("%Y年%m月%d日").replace("年0", "年").replace("月0", "月")
                text = body.format(y=d.year, dir=rng.choice(["增加", "减少"]), dir2=rng.choice(["提高", "降低"]), eff=eff)
                for rep in range(1 + int(rng.random() < 0.4)):          # repeat reports must not re-trigger memory
                    docs.append({"document_id": f"syn{k:07d}", "title": title, "body": text,
                                 "publish_datetime": f"{(d + pd.Timedelta(days=rep)).date()}", "date_only_publish_flag": 1,
                                 "markets": [mk], "source_type": "synthetic_policy",
                                 "publisher": "交易中心" if cat in ("Trading", "Finance") else "生态环境厅"})
                    k += 1
    return docs


def write_synthetic(cfg: dict, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    raw = Path(cfg["paths"]["raw_dir"]).resolve()
    from ..config import PROJECT_ROOT
    smoke_root = (PROJECT_ROOT / "data_smoke").resolve()
    if not raw.is_relative_to(smoke_root) or smoke_root != PROJECT_ROOT.resolve() / "data_smoke":
        raise ValueError("Synthetic data may only be written under the repository data_smoke/ directory.")
    if not set(cfg["markets"]).issubset({"SynA", "SynB"}):
        raise ValueError("Synthetic generation requires the smoke markets SynA/SynB.")
    ensure_dir(raw / "prices")
    prices = {}
    for j, (mk, mcfg) in enumerate(cfg["markets"].items()):
        n = int(cfg["reserve"]) + int(sum(mcfg["splits"]))
        prices[mk] = make_market(mk, mcfg.get("product", mk), n, rng, start=f"{2017 + j}-01-02")
        prices[mk].to_csv(raw / "prices" / f"{mk}_observed_prices.csv", index=False)
    write_jsonl(make_documents(prices, rng), raw / "documents.jsonl")
