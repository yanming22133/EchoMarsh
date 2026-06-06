"""
EchoMarsh 社区情绪模块 (Community Sentiment Engine)
====================================================
从东方财富免费接口获取全网关注度数据，量化社区情绪。

数据源（全部免费，通过 akshare）：
  1. stock_hot_rank_em()        — 东财实时热度 Top 100
  2. stock_hot_up_em()          — 东财热度飙升榜（突然被大量关注的票）
  3. stock_comment_em()         — 东财个股综合评分（上涨概率/成本支撑度等）

输出：
  {code: SentimentResult} 字典，每只股票的社区情绪得分（0~15分）
"""

import os
import time
import traceback
import pandas as pd

# 绕过代理
os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import akshare as ak

# 东方财富部分 POST 端点需要浏览器 TLS 指纹
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None


class SentimentResult:
    """单只股票的社区情绪评分结果"""
    def __init__(self, code):
        self.code = code
        self.hot_rank = None          # 热度排名（1=最热，None=不在榜）
        self.hot_rank_score = 0.0     # 热度排名得分（0~8）
        self.surge_rank = None        # 飙升排名
        self.surge_score = 0.0        # 飙升得分（0~7）
        self.comment_score = None     # 东财综合评分（原始值）
        self.comment_up_prob = None   # 上涨概率
        self.total_score = 0.0        # 汇总得分（0~15）
        self.tags = []                # 标签列表 ['热榜Top30', '飙升']

    def __repr__(self):
        tags_str = ','.join(self.tags) if self.tags else '无'
        return f"<Sentiment {self.code} score={self.total_score:.1f} tags=[{tags_str}]>"


class CommunitySentimentEngine:
    """
    社区情绪引擎。
    一次性拉取全市场数据，构建查询字典，避免逐票请求。
    """

    def __init__(self):
        self._hot_rank_dict = {}      # {code: rank}
        self._surge_dict = {}         # {code: rank}
        self._comment_dict = {}       # {code: {score, up_prob, ...}}
        self._loaded = False

    def load(self):
        """拉取所有社区情绪数据（收盘后调用一次即可）"""
        print("[社区情绪] 开始拉取东财社区数据...")
        self._load_hot_rank()
        time.sleep(0.5)
        self._load_surge_rank()
        time.sleep(0.5)
        self._load_comment_scores()
        self._loaded = True
        print(f"[社区情绪] 数据加载完成: 热榜 {len(self._hot_rank_dict)} 只, "
              f"飙升 {len(self._surge_dict)} 只, 评分 {len(self._comment_dict)} 只")

    def _load_hot_rank(self):
        """东财实时热度排行 (curl_cffi 绕过 TLS 指纹检测)"""
        try:
            df = None
            if curl_requests:
                try:
                    url = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"
                    payload = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
                               "marketType": "", "pageNo": 1, "pageSize": 100}
                    r = curl_requests.post(url, json=payload, impersonate="chrome131", timeout=10)
                    data = r.json().get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        if 'sc' in df.columns:
                            df['代码'] = df['sc'].str.extract(r'(\d{6})')
                except Exception:
                    pass
            if df is None:
                df = ak.stock_hot_rank_em()
            if df is not None and not df.empty:
                code_col = self._find_column(df, ['股票代码', '代码'])
                if code_col:
                    for idx, row in df.iterrows():
                        code = str(row[code_col]).zfill(6)
                        self._hot_rank_dict[code] = idx + 1
                    print(f"  [热度榜] 成功加载 {len(self._hot_rank_dict)} 只")
        except Exception as e:
            print(f"  [热度榜] 加载失败: {e}")

    def _load_surge_rank(self):
        """东财热度飙升榜 (curl_cffi 绕过 TLS 指纹检测)"""
        try:
            df = None
            if curl_requests:
                try:
                    url = "https://emappdata.eastmoney.com/stockrank/getAllRiseList"
                    payload = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
                               "marketType": "", "pageNo": 1, "pageSize": 100}
                    r = curl_requests.post(url, json=payload, impersonate="chrome131", timeout=10)
                    data = r.json().get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        if 'sc' in df.columns:
                            df['代码'] = df['sc'].str.extract(r'(\d{6})')
                except Exception:
                    pass
            if df is None:
                df = ak.stock_hot_up_em()
            if df is not None and not df.empty:
                code_col = self._find_column(df, ['股票代码', '代码'])
                if code_col:
                    for idx, row in df.iterrows():
                        code = str(row[code_col]).zfill(6)
                        self._surge_dict[code] = idx + 1
                    print(f"  [飙升榜] 成功加载 {len(self._surge_dict)} 只")
        except Exception as e:
            print(f"  [飙升榜] 加载失败: {e}")

    def _load_comment_scores(self):
        """东财个股综合评分"""
        try:
            df = ak.stock_comment_em()
            if df is not None and not df.empty:
                code_col = self._find_column(df, ['代码', '股票代码'])
                score_col = self._find_column(df, ['综合得分', '综合评分'])
                up_prob_col = self._find_column(df, ['上涨概率', '上涨'])
                if code_col:
                    for _, row in df.iterrows():
                        code = str(row[code_col]).zfill(6)
                        entry = {}
                        if score_col:
                            try:
                                entry['score'] = float(row[score_col])
                            except (ValueError, TypeError):
                                entry['score'] = None
                        if up_prob_col:
                            try:
                                entry['up_prob'] = float(row[up_prob_col])
                            except (ValueError, TypeError):
                                entry['up_prob'] = None
                        self._comment_dict[code] = entry
                    print(f"  [综合评分] 成功加载 {len(self._comment_dict)} 只")
                else:
                    print(f"  [综合评分] 未找到代码列，可用列: {list(df.columns)}")
        except Exception as e:
            print(f"  [综合评分] 加载失败: {e}")
            traceback.print_exc()

    def score(self, code: str) -> SentimentResult:
        """
        查询单只股票的社区情绪得分。

        打分逻辑（满分 15 分）：
          - 热榜 Top 30  → +8分
          - 热榜 31~50   → +6分
          - 热榜 51~100  → +4分
          - 飙升榜 Top 10 → +7分
          - 飙升榜 11~30  → +5分
          - 飙升榜 31~50  → +3分
          - 上限 cap 在 15 分
        """
        code = str(code).zfill(6)
        result = SentimentResult(code)

        # 热度排名打分
        rank = self._hot_rank_dict.get(code)
        if rank is not None:
            result.hot_rank = rank
            if rank <= 30:
                result.hot_rank_score = 8.0
                result.tags.append(f'热榜Top{rank}')
            elif rank <= 50:
                result.hot_rank_score = 6.0
                result.tags.append(f'热榜Top{rank}')
            elif rank <= 100:
                result.hot_rank_score = 4.0
                result.tags.append('热榜')

        # 飙升排名打分
        surge = self._surge_dict.get(code)
        if surge is not None:
            result.surge_rank = surge
            if surge <= 10:
                result.surge_score = 7.0
                result.tags.append('飙升Top10')
            elif surge <= 30:
                result.surge_score = 5.0
                result.tags.append('飙升')
            elif surge <= 50:
                result.surge_score = 3.0
                result.tags.append('飙升')

        # 东财综合评分（附加信息，不直接加分，用于推荐理由）
        comment = self._comment_dict.get(code)
        if comment:
            result.comment_score = comment.get('score')
            result.comment_up_prob = comment.get('up_prob')

        # 汇总（上限 15）
        result.total_score = min(result.hot_rank_score + result.surge_score, 15.0)

        return result

    def batch_score(self, codes: list) -> dict:
        """批量查询多只股票的社区情绪"""
        if not self._loaded:
            self.load()
        return {code: self.score(code) for code in codes}

    # ============ 预留扩展接口 ============

    def xueqiu_score(self, code: str) -> float:
        """
        [预留] 雪球社区情绪打分。
        TODO: 未来接入雪球评论 NLP 分析。
        """
        return 0.0

    def weibo_score(self, code: str) -> float:
        """
        [预留] 微博/B站舆情打分。
        TODO: 未来接入社交媒体 NLP 分析。
        """
        return 0.0

    @staticmethod
    def _find_column(df, candidates):
        """在 DataFrame 中查找第一个匹配的列名"""
        for c in candidates:
            if c in df.columns:
                return c
        return None


if __name__ == "__main__":
    engine = CommunitySentimentEngine()
    engine.load()

    # 测试几只热门股
    test_codes = ['000001', '600519', '300750', '601398']
    results = engine.batch_score(test_codes)
    for code, r in results.items():
        print(f"  {code}: {r}")
