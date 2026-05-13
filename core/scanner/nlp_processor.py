import numpy as np

class NLPProcessor:
    def __init__(self, use_deep_learning=False):
        """
        初始化 NLP 情感处理引擎。
        :param use_deep_learning: 如果为 True，则加载本地大模型(如 FinBERT)，否则使用轻量级词典/规则
        """
        self.use_deep_learning = use_deep_learning
        if self.use_deep_learning:
            print("Loading FinBERT / Roberta model on RTX 4070... (Mocking)")
            # 这里将在未来集成 transformers 库
            # self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            # self.model = BertForSequenceClassification.from_pretrained(...)
        else:
            print("Using lightweight NLP rule engine.")

    def analyze_sentiment(self, text_list):
        """
        分析一组文本的情绪，返回综合情绪得分 [-1, 1]。
        1 代表极度狂热看多，-1 代表极度恐慌看空。
        """
        if not text_list:
            return 0.0
            
        # Mock 实现
        total_score = 0
        for text in text_list:
            if "满仓" in text or "打板" in text or "十倍" in text:
                total_score += 0.8
            elif "跌停" in text or "快跑" in text or "杀猪" in text:
                total_score -= 0.8
            else:
                total_score += np.random.uniform(-0.1, 0.2)
                
        # 归一化到 [-1, 1]
        avg_score = total_score / len(text_list)
        return max(min(avg_score, 1.0), -1.0)

    def detect_shill_divergence(self, sentiment_score, main_net_inflow_ratio):
        """
        核心防割韭菜逻辑 (Divergence Checker)。
        量价背离检查：如果散户情绪极度狂热，但主力资金在大举流出，则判定为“杀猪盘”或诱多出货。
        
        :param sentiment_score: 论坛/B站舆情的情绪得分 [-1, 1]
        :param main_net_inflow_ratio: 主力资金净流入占比（负数代表净流出）
        :return: bool (True 表示高危噪音，应该拉黑)
        """
        # 阈值可调：如果情绪分 > 0.6 (非常狂热)，但主力净流出比例低于 -5%
        if sentiment_score > 0.6 and main_net_inflow_ratio < -5.0:
            print(f"[警告] 检测到情绪与资金严重背离！情绪分={sentiment_score:.2f}, 主力流入={main_net_inflow_ratio}% (可能是恶意接盘推荐)")
            return True
        return False
