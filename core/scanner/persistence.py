"""
EchoMarsh 数据持久化层 (Persistence Layer)
==========================================
使用 SQLite 本地数据库存储：
  - 每日推荐记录
  - 持仓记录
  - 交易记录（买入/卖出）
  - 每日账户快照（总资产/盈亏）

数据库文件: data/echomarsh.db
"""

import os
import sqlite3
from datetime import datetime, date
from typing import List, Optional
from dataclasses import dataclass, field


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'echomarsh.db')


@dataclass
class PositionRecord:
    code: str
    name: str
    volume: int           # 股数
    cost_price: float     # 成本价
    buy_date: str         # YYYYMMDD
    current_price: float = 0.0
    pnl_pct: float = 0.0  # 浮动盈亏%
    note: str = ''


@dataclass
class TradeRecord:
    trade_date: str       # YYYYMMDD
    trade_time: str       # HH:MM:SS
    code: str
    name: str
    direction: str        # BUY / SELL
    price: float
    volume: int
    amount: float         # 成交金额
    reason: str = ''      # 交易理由


class PersistenceManager:
    """数据持久化管理器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.abspath(DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()
        c = conn.cursor()

        # 每日推荐记录
        c.execute('''CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rec_date TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            industry TEXT,
            lianban INTEGER,
            total_score REAL,
            top_reason TEXT,
            market_level TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')

        # 持仓记录（当前持仓）
        c.execute('''CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            name TEXT,
            volume INTEGER,
            cost_price REAL,
            buy_date TEXT,
            note TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')

        # 交易记录（历史所有交易）
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            trade_time TEXT,
            code TEXT NOT NULL,
            name TEXT,
            direction TEXT NOT NULL,
            price REAL,
            volume INTEGER,
            amount REAL,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')

        # 每日模型预测记录（用于对比每日分数变化）
        c.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_date TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            industry TEXT,
            factor_score REAL,
            model_ret_pred REAL,
            model_limit_up_prob REAL,
            combined_score REAL,
            lianban INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pred_date, code)
        )''')

        # 每日账户快照
        c.execute('''CREATE TABLE IF NOT EXISTS daily_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snap_date TEXT NOT NULL UNIQUE,
            total_asset REAL,
            cash REAL,
            position_value REAL,
            daily_pnl REAL,
            total_pnl_pct REAL,
            position_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')

        conn.commit()
        conn.close()

    # ==================== 推荐记录 ====================

    def save_recommendations(self, rec_date: str, scored_stocks: list, market_level: str):
        """保存当日推荐结果"""
        conn = self._get_conn()
        c = conn.cursor()
        # 先清除当日旧记录
        c.execute("DELETE FROM recommendations WHERE rec_date=?", (rec_date,))
        for s in scored_stocks:
            top_reasons = '; '.join([r.detail for r in s.top_reasons(3)])
            c.execute("""INSERT INTO recommendations 
                        (rec_date, code, name, industry, lianban, total_score, top_reason, market_level)
                        VALUES (?,?,?,?,?,?,?,?)""",
                      (rec_date, s.code, s.name, s.industry, s.lianban,
                       s.total_score, top_reasons, market_level))
        conn.commit()
        conn.close()
        print(f"[存储] 已保存 {len(scored_stocks)} 条推荐记录 ({rec_date})")

    def get_recommendations(self, rec_date: str) -> list:
        """获取指定日期的推荐记录"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT code, name, industry, lianban, total_score, top_reason, market_level "
                  "FROM recommendations WHERE rec_date=? ORDER BY total_score DESC", (rec_date,))
        rows = c.fetchall()
        conn.close()
        return [{'code': r[0], 'name': r[1], 'industry': r[2], 'lianban': r[3],
                 'score': r[4], 'reason': r[5], 'market': r[6]} for r in rows]

    def get_recent_recommendations(self, days: int = 5) -> dict:
        """获取最近 N 天的推荐记录"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT DISTINCT rec_date FROM recommendations ORDER BY rec_date DESC LIMIT ?", (days,))
        dates = [r[0] for r in c.fetchall()]
        conn.close()
        return {d: self.get_recommendations(d) for d in dates}

    # ==================== 持仓管理 ====================

    def add_position(self, code: str, name: str, volume: int, cost_price: float,
                     buy_date: str = None, note: str = ''):
        """新增或更新持仓"""
        conn = self._get_conn()
        c = conn.cursor()
        buy_date = buy_date or datetime.now().strftime('%Y%m%d')
        c.execute("""INSERT OR REPLACE INTO positions 
                    (code, name, volume, cost_price, buy_date, note, updated_at)
                    VALUES (?,?,?,?,?,?,?)""",
                  (code, name, volume, cost_price, buy_date, note,
                   datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        print(f"[持仓] 已记录: {code} {name} x{volume} @ {cost_price}")

    def remove_position(self, code: str):
        """清除持仓（卖出后）"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM positions WHERE code=?", (code,))
        conn.commit()
        conn.close()

    def get_positions(self) -> List[PositionRecord]:
        """获取当前所有持仓"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT code, name, volume, cost_price, buy_date, note FROM positions")
        rows = c.fetchall()
        conn.close()
        return [PositionRecord(code=r[0], name=r[1], volume=r[2],
                               cost_price=r[3], buy_date=r[4], note=r[5]) for r in rows]

    def update_position_price(self, code: str, current_price: float):
        """更新持仓的当前价格（不存表，仅运行时用）"""
        pass  # 实时价格不入库，每次查询时实时计算

    # ==================== 每日预测记录 ====================

    def save_predictions(self, pred_date: str, predictions: list):
        """保存当日所有股票预测结果"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM predictions WHERE pred_date=?", (pred_date,))
        for p in predictions:
            c.execute("""INSERT OR REPLACE INTO predictions
                        (pred_date, code, name, industry, factor_score,
                         model_ret_pred, model_limit_up_prob, combined_score, lianban)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                      (pred_date, p.get('code'), p.get('name'), p.get('industry'),
                       p.get('factor_score'), p.get('model_ret_pred'),
                       p.get('model_limit_up_prob'), p.get('combined_score'), p.get('lianban', 0)))
        conn.commit()
        conn.close()
        print(f"[存储] 已保存 {len(predictions)} 条预测记录 ({pred_date})")

    def get_predictions(self, pred_date: str) -> list:
        """获取指定日期的预测结果"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""SELECT code, name, industry, factor_score, model_ret_pred,
                     model_limit_up_prob, combined_score, lianban
                     FROM predictions WHERE pred_date=? ORDER BY combined_score DESC""", (pred_date,))
        rows = c.fetchall()
        conn.close()
        return [{'code': r[0], 'name': r[1], 'industry': r[2],
                 'factor_score': r[3], 'model_ret_pred': r[4],
                 'model_limit_up_prob': r[5], 'combined_score': r[6], 'lianban': r[7]} for r in rows]

    def get_prediction_dates(self, limit: int = 10) -> list:
        """获取最近的预测日期列表"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT DISTINCT pred_date FROM predictions ORDER BY pred_date DESC LIMIT ?", (limit,))
        dates = [r[0] for r in c.fetchall()]
        conn.close()
        return dates

    def get_prediction_with_changes(self, today: str, prev: str = None) -> list:
        """
        获取今日预测并与前一日期对比 score 变化。
        返回: [{code, name, industry, score, prev_score, change, lianban}, ...]
        """
        today_records = self.get_predictions(today)
        if not prev:
            dates = self.get_prediction_dates(2)
            if len(dates) < 2:
                return today_records
            prev = dates[1]

        prev_map = {}
        for r in self.get_predictions(prev):
            prev_map[r['code']] = r['combined_score']

        result = []
        for r in today_records:
            code = r['code']
            prev_score = prev_map.get(code)
            change = r['combined_score'] - prev_score if prev_score is not None else None
            result.append({**r, 'prev_score': prev_score, 'change': change})

        result.sort(key=lambda x: x['combined_score'], reverse=True)
        return result

    # ==================== 交易记录 ====================

    def add_trade(self, code: str, name: str, direction: str, price: float,
                  volume: int, reason: str = '', trade_date: str = None):
        """记录一笔交易"""
        conn = self._get_conn()
        c = conn.cursor()
        trade_date = trade_date or datetime.now().strftime('%Y%m%d')
        trade_time = datetime.now().strftime('%H:%M:%S')
        amount = price * volume
        c.execute("""INSERT INTO trades 
                    (trade_date, trade_time, code, name, direction, price, volume, amount, reason)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                  (trade_date, trade_time, code, name, direction, price, volume, amount, reason))
        conn.commit()
        conn.close()
        print(f"[交易] {direction} {code} {name} x{volume} @ {price} ({reason})")

    def get_trades(self, trade_date: str = None, days: int = None) -> List[dict]:
        """获取交易记录"""
        conn = self._get_conn()
        c = conn.cursor()
        if trade_date:
            c.execute("SELECT * FROM trades WHERE trade_date=? ORDER BY trade_time", (trade_date,))
        elif days:
            c.execute(f"SELECT * FROM trades ORDER BY trade_date DESC, trade_time DESC LIMIT {days * 10}")
        else:
            c.execute("SELECT * FROM trades ORDER BY trade_date DESC, trade_time DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        cols = ['id', 'trade_date', 'trade_time', 'code', 'name', 'direction',
                'price', 'volume', 'amount', 'reason', 'created_at']
        return [dict(zip(cols, r)) for r in rows]

    # ==================== 每日快照 ====================

    def save_daily_snapshot(self, snap_date: str, total_asset: float, cash: float,
                           position_value: float, daily_pnl: float,
                           total_pnl_pct: float, position_count: int):
        """保存每日账户快照"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO daily_snapshots
                    (snap_date, total_asset, cash, position_value, daily_pnl,
                     total_pnl_pct, position_count)
                    VALUES (?,?,?,?,?,?,?)""",
                  (snap_date, total_asset, cash, position_value, daily_pnl,
                   total_pnl_pct, position_count))
        conn.commit()
        conn.close()

    def get_equity_curve(self, days: int = 30) -> list:
        """获取资金曲线数据"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT snap_date, total_asset, daily_pnl, total_pnl_pct "
                  "FROM daily_snapshots ORDER BY snap_date DESC LIMIT ?", (days,))
        rows = c.fetchall()
        conn.close()
        return [{'date': r[0], 'asset': r[1], 'daily_pnl': r[2], 'total_pnl': r[3]}
                for r in reversed(rows)]


if __name__ == "__main__":
    pm = PersistenceManager()
    print(f"数据库路径: {pm.db_path}")

    # 测试持仓
    pm.add_position('002552', '宝鼎科技', 500, 12.50, '20260512', '4连板龙头')
    pm.add_position('600530', '交大昂立', 800, 8.30, '20260512', '3连板食品')

    positions = pm.get_positions()
    print(f"当前持仓: {len(positions)} 只")
    for p in positions:
        print(f"  {p.code} {p.name} x{p.volume} @ {p.cost_price}")

    # 测试交易记录
    pm.add_trade('002552', '宝鼎科技', 'BUY', 12.50, 500, '12因子评分52分，4连板龙头')
    trades = pm.get_trades()
    print(f"交易记录: {len(trades)} 条")
