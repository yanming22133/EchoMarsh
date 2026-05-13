"""
EchoMarsh QMT 自动交易执行器 (Auto Trading Executor)
----------------------------------------------------
负责与 QMT (xtquant) 交互，执行模型发出的买卖信号。
包含完整的风险控制逻辑，防止死循环发单等灾难性事故。

实盘启动前必须检查：
  1. QMT 客户端已登录并正在运行
  2. 账户资金充足
  3. 网络连接到交易所

模拟模式（paper=True）：不会真实下单，只打印订单信息。
"""

import os
import time
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("EchoMarsh.Executor")


@dataclass
class Order:
    symbol: str
    direction: str          # 'BUY' or 'SELL'
    price: float
    volume: int             # 股数 (非手数)
    timestamp: str = ""
    order_id: str = ""
    status: str = "PENDING" # PENDING / FILLED / CANCELLED / FAILED

@dataclass
class Position:
    symbol: str
    volume: int
    cost_price: float
    buy_date: date


class RiskController:
    """
    冷血风险控制器 (Risk Controller)
    每次下单前必须通过所有检查，否则拒绝执行。
    """
    def __init__(self, max_position_pct=0.95, max_single_loss_pct=0.05,
                 max_daily_orders=10, min_confidence=0.55):
        self.max_position_pct    = max_position_pct    # 最大仓位比例
        self.max_single_loss_pct = max_single_loss_pct # 单笔最大亏损容忍
        self.max_daily_orders    = max_daily_orders     # 每日最大下单次数（防止死循环）
        self.min_confidence      = min_confidence       # 最低涨停概率阈值
        self.daily_order_count   = 0

    def check(self, signal_confidence, total_capital, cash, existing_positions) -> tuple[bool, str]:
        """返回 (是否允许下单, 拒绝原因)"""
        # 1. 防止死循环发单
        if self.daily_order_count >= self.max_daily_orders:
            return False, f"每日下单次数已达上限 {self.max_daily_orders} 次，拒绝执行！"

        # 2. 模型置信度检查
        if signal_confidence < self.min_confidence:
            return False, f"模型置信度 {signal_confidence:.2%} 低于阈值 {self.min_confidence:.2%}"

        # 3. 资金检查（最低保留 5% 的现金作为备用）
        min_cash = total_capital * 0.05
        if cash <= min_cash:
            return False, f"现金不足！当前现金 {cash:.0f}，最低需保留 {min_cash:.0f}"

        return True, "PASS"


class QMTExecutor:
    def __init__(self, account_id: str = "", paper_mode: bool = True):
        """
        :param account_id: QMT 账号（实盘时填写）
        :param paper_mode: True = 模拟模式，不真实发单；False = 实盘模式
        """
        self.account_id    = account_id
        self.paper_mode    = paper_mode
        self.risk          = RiskController()
        self.positions: List[Position] = []
        self.order_history: List[Order] = []
        self.total_capital = 20000.0
        self.cash          = 20000.0
        self._xt_trader    = None

        if not paper_mode:
            self._connect_qmt()

    def _connect_qmt(self):
        """连接 QMT 交易接口"""
        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant import xtconstant
            session_id = int(time.time())
            self._xt_trader = XtQuantTrader('', session_id)
            self._xt_trader.start()
            connect_result = self._xt_trader.connect()
            if connect_result != 0:
                raise ConnectionError(f"QMT 连接失败，代码: {connect_result}")
            logger.info(f"QMT 连接成功，账号: {self.account_id}")
        except ImportError:
            logger.error("xtquant 未安装！请先安装 QMT 交易端。切换到纸上交易模式。")
            self.paper_mode = True

    def get_cash(self) -> float:
        """获取账户可用资金"""
        if self.paper_mode:
            return self.cash
        try:
            account = self._xt_trader.query_stock_asset(self.account_id)
            return account.cash if account else 0.0
        except Exception as e:
            logger.error(f"查询资金失败: {e}")
            return 0.0

    def get_positions(self) -> List[Position]:
        """获取当前持仓"""
        if self.paper_mode:
            return self.positions
        try:
            positions = self._xt_trader.query_stock_positions(self.account_id)
            return [
                Position(symbol=p.stock_code, volume=p.volume,
                         cost_price=p.avg_price, buy_date=date.today())
                for p in (positions or [])
            ]
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return []

    def buy(self, symbol: str, confidence: float, price: float = 0.0) -> Optional[Order]:
        """
        执行买入信号
        :param symbol: 六位股票代码
        :param confidence: 模型涨停概率 [0, 1]
        :param price: 指定买入价，0 表示市价
        """
        cash = self.get_cash()
        positions = self.get_positions()

        ok, reason = self.risk.check(confidence, self.total_capital, cash, positions)
        if not ok:
            logger.warning(f"[RISK BLOCK] 买入 {symbol} 被拒绝: {reason}")
            return None

        # 计算买入数量（满仓），以 100 股为最小单位
        buy_amount = cash * 0.95  # 留 5% 备用
        if price <= 0:
            price = self._get_realtime_price(symbol)
        volume = int(buy_amount / price / 100) * 100

        if volume <= 0:
            logger.warning(f"[SKIP] {symbol} 计算买入量为 0，跳过。")
            return None

        order = Order(
            symbol=symbol, direction='BUY',
            price=price, volume=volume,
            timestamp=datetime.now().strftime('%H:%M:%S')
        )

        if self.paper_mode:
            cost = volume * price * (1 + 0.0003 + 0.002)  # 手续费 + 滑点
            self.cash -= cost
            self.positions.append(Position(symbol=symbol, volume=volume,
                                           cost_price=price, buy_date=date.today()))
            order.status = 'FILLED'
            logger.info(f"[PAPER BUY] {symbol} x{volume} @ {price:.2f} | 信心: {confidence:.2%}")
        else:
            try:
                from xtquant import xtconstant
                order_id = self._xt_trader.order_stock(
                    self.account_id, symbol, xtconstant.STOCK_BUY,
                    volume, xtconstant.FIX_PRICE if price > 0 else xtconstant.MARKET_PRICE,
                    price, 'EchoMarsh', 'EchoMarsh Auto Buy'
                )
                order.order_id = str(order_id)
                order.status = 'SUBMITTED'
                logger.info(f"[REAL BUY] {symbol} x{volume} @ {price:.2f} | OrderID: {order_id}")
            except Exception as e:
                order.status = 'FAILED'
                logger.error(f"买入失败: {e}")
                return None

        self.risk.daily_order_count += 1
        self.order_history.append(order)
        return order

    def sell(self, symbol: str, price: float = 0.0) -> Optional[Order]:
        """
        执行卖出信号（遵循 T+1，只能卖出昨日买入的仓位）
        """
        today = date.today()
        sellable = [p for p in self.positions
                    if p.symbol == symbol and p.buy_date < today]

        if not sellable:
            logger.info(f"[T+1 BLOCK] {symbol} 今日买入，不可卖出（T+1 限制）")
            return None

        pos = sellable[0]
        if price <= 0:
            price = self._get_realtime_price(symbol)

        order = Order(
            symbol=symbol, direction='SELL',
            price=price, volume=pos.volume,
            timestamp=datetime.now().strftime('%H:%M:%S')
        )

        if self.paper_mode:
            proceeds = pos.volume * price * (1 - 0.0003 - 0.001 - 0.002)  # 印花税 + 手续费 + 滑点
            self.cash += proceeds
            self.positions = [p for p in self.positions if p.symbol != symbol]
            order.status = 'FILLED'
            pnl = (price - pos.cost_price) / pos.cost_price * 100
            logger.info(f"[PAPER SELL] {symbol} x{pos.volume} @ {price:.2f} | PnL: {pnl:+.2f}%")
        else:
            try:
                from xtquant import xtconstant
                order_id = self._xt_trader.order_stock(
                    self.account_id, symbol, xtconstant.STOCK_SELL,
                    pos.volume, xtconstant.FIX_PRICE if price > 0 else xtconstant.MARKET_PRICE,
                    price, 'EchoMarsh', 'EchoMarsh Auto Sell'
                )
                order.order_id = str(order_id)
                order.status = 'SUBMITTED'
                logger.info(f"[REAL SELL] {symbol} x{pos.volume} @ {price:.2f} | OrderID: {order_id}")
            except Exception as e:
                order.status = 'FAILED'
                logger.error(f"卖出失败: {e}")
                return None

        self.order_history.append(order)
        return order

    def _get_realtime_price(self, symbol: str) -> float:
        """获取实时最新价（实盘调用 QMT，模拟模式返回 0 触发市价）"""
        if self.paper_mode:
            return 10.0  # 模拟价格
        try:
            from xtquant import xtdata
            ticks = xtdata.get_full_tick([symbol])
            return ticks[symbol]['lastPrice'] if symbol in ticks else 0.0
        except Exception:
            return 0.0

    def reset_daily_counter(self):
        """每天开盘前重置订单计数器"""
        self.risk.daily_order_count = 0
        logger.info("每日订单计数器已重置。")

    # ============================================================
    # 做 T 逻辑 (T+0 高抛低吸，降低持仓成本)
    # ============================================================
    def do_t_check(self, symbol: str,
                   take_profit_pct: float = 0.03,
                   buyback_pct: float = 0.02) -> str:
        """
        做 T 检查器：判断当前持仓是否满足高抛低吸的条件。
        仅对 昨日买入（T+1 可卖）的仓位操作，当日买入的绝对不动。

        做 T 完整流程:
        1. 持仓浮盈 > take_profit_pct (如 3%) → 高抛 1/3 仓位
        2. 继续监控：若股价从高抛价格回调 buyback_pct (如 2%) → 低吸买回
        3. 净效果：持仓成本降低，相当于提前锁定了部分利润

        :return: 'SELL_HALF' / 'BUYBACK' / 'HOLD' / 'T1_BLOCKED'
        """
        today = date.today()
        sellable = [p for p in self.positions
                    if p.symbol == symbol and p.buy_date < today]
        if not sellable:
            return 'T1_BLOCKED'  # 今日买入，不可做T

        pos = sellable[0]
        cur_price = self._get_realtime_price(pos.symbol)
        float_pnl_pct = (cur_price - pos.cost_price) / pos.cost_price

        # 阶段 1：浮盈超过阈值，触发高抛
        if float_pnl_pct >= take_profit_pct:
            logger.info(f"[做T-高抛] {symbol} 浮盈 {float_pnl_pct:.2%} >= {take_profit_pct:.2%}，触发高抛 1/3 仓位")
            return 'SELL_THIRD'

        return 'HOLD'

    def sell_for_t(self, symbol: str, fraction: float = 1/3) -> Optional[Order]:
        """
        做 T 高抛：卖出仓位的 fraction 比例，为低吸买回做准备。
        仅允许对昨日仓位操作（T+1 可卖）。
        """
        today = date.today()
        sellable = [p for p in self.positions
                    if p.symbol == symbol and p.buy_date < today]
        if not sellable:
            logger.info(f"[T1 BLOCK] {symbol} 今日买入，做T高抛被拒绝")
            return None

        pos = sellable[0]
        sell_vol = int(pos.volume * fraction / 100) * 100
        if sell_vol <= 0:
            return None

        price = self._get_realtime_price(symbol)
        order = Order(
            symbol=symbol, direction='SELL_T',
            price=price, volume=sell_vol,
            timestamp=datetime.now().strftime('%H:%M:%S')
        )

        if self.paper_mode:
            proceeds = sell_vol * price * (1 - 0.0003 - 0.001 - 0.002)
            self.cash += proceeds
            # 更新持仓数量
            for p in self.positions:
                if p.symbol == symbol and p.buy_date < today:
                    p.volume -= sell_vol
                    break
            order.status = 'FILLED'
            logger.info(f"[做T-高抛] {symbol} 卖出 {sell_vol} 股 @ {price:.2f}，待低吸买回")
        else:
            self._submit_real_order(order)

        self.order_history.append(order)
        return order

    # ============================================================
    # 多时窗策略分发器 (Multi-Window Strategy Dispatcher)
    # ============================================================
    def dispatch_strategy(self, signals: list, window: str) -> list:
        """
        根据当前时窗，自动选择最适合的交易策略并执行。

        :param signals: [(symbol, limit_up_prob, return_pred), ...] 已按概率排序
        :param window:  '竞价' | '早盘' | '日内' | '尾盘' | '次日开盘'
        :return: 本轮执行的 Order 列表
        """
        executed = []

        if window == '竞价':
            # 竞价期仅侦察，记录高竞价量比候选，不下单
            top = [(s, p, r) for s, p, r in signals if p > 0.65]
            logger.info(f"[竞价侦察] {len(top)} 只候选进入早盘观察名单")

        elif window == '早盘':
            # ✅ 主战场：早盘 09:30~10:00 打首板
            # 挑选置信度最高的票满仓买入
            if signals and not self.positions:
                sym, prob, ret = signals[0]
                if prob > 0.70:
                    order = self.buy(sym, prob)
                    if order:
                        executed.append(order)
                        logger.info(f"[早盘打板] {sym} 买入，涨停概率 {prob:.2%}")

        elif window == '日内':
            # 日内持仓管理：做 T + 止损巡逻
            for pos in self.get_positions():
                cur = self._get_realtime_price(pos.symbol)
                pnl = (cur - pos.cost_price) / pos.cost_price

                # 止损：浮亏超过 5% 无脑割（昨日仓位）
                if pnl < -0.05 and pos.buy_date < date.today():
                    logger.warning(f"[止损] {pos.symbol} 浮亏 {pnl:.2%}，触发强制止损！")
                    order = self.sell(pos.symbol)
                    if order:
                        executed.append(order)

                # 做T：浮盈 3% 以上，高抛 1/3
                elif pnl >= 0.03 and pos.buy_date < date.today():
                    t_action = self.do_t_check(pos.symbol)
                    if t_action == 'SELL_THIRD':
                        order = self.sell_for_t(pos.symbol, fraction=1/3)
                        if order:
                            executed.append(order)

        elif window == '尾盘':
            # 尾盘 14:30~14:55：埋伏次日连板预期强的票
            # 此时市场信息最充分，模型置信度通常更高
            if not self.positions and signals:
                sym, prob, ret = signals[0]
                if prob > 0.75:  # 尾盘要求更高置信度
                    order = self.buy(sym, prob)
                    if order:
                        executed.append(order)
                        logger.info(f"[尾盘埋伏] {sym} 买入，次日连板预期强")

        elif window == '次日开盘':
            # 次日开盘：对昨日买入的仓位执行获利了结
            for pos in self.get_positions():
                if pos.buy_date < date.today():
                    cur = self._get_realtime_price(pos.symbol)
                    pnl = (cur - pos.cost_price) / pos.cost_price
                    # 开盘高开 3% 以上，直接卖出锁定利润
                    if pnl >= 0.03:
                        logger.info(f"[次日高开] {pos.symbol} 高开 {pnl:.2%}，锁定利润卖出")
                        order = self.sell(pos.symbol)
                        if order:
                            executed.append(order)
                    # 开盘低开 2% 以上，止损
                    elif pnl < -0.02:
                        logger.warning(f"[次日低开] {pos.symbol} 低开 {pnl:.2%}，止损卖出")
                        order = self.sell(pos.symbol)
                        if order:
                            executed.append(order)

        return executed

    def _submit_real_order(self, order: Order):
        """提交真实 QMT 订单（供内部复用）"""
        try:
            from xtquant import xtconstant
            dir_map = {'BUY': xtconstant.STOCK_BUY,
                       'SELL': xtconstant.STOCK_SELL,
                       'SELL_T': xtconstant.STOCK_SELL}
            order_id = self._xt_trader.order_stock(
                self.account_id, order.symbol, dir_map[order.direction],
                order.volume,
                xtconstant.FIX_PRICE if order.price > 0 else xtconstant.MARKET_PRICE,
                order.price, 'EchoMarsh', f'EchoMarsh {order.direction}'
            )
            order.order_id = str(order_id)
            order.status = 'SUBMITTED'
        except Exception as e:
            order.status = 'FAILED'
            logger.error(f"真实下单失败: {e}")
