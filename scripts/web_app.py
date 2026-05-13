"""
EchoMarsh 本地量化门户 Web 界面
基于 Streamlit 构建的资产管理、大盘监控与策略推荐终端。

用法:
    streamlit run scripts/web_app.py
"""

import os
import sys
import pandas as pd
import streamlit as st
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 初始化页面配置 (必须在第一个 Streamlit 命令)
st.set_page_config(page_title="EchoMarsh 鸣泽量化", page_icon="⚡", layout="wide")

# 尝试导入核心引擎
try:
    from core.scanner.persistence import PersistenceManager
    from core.scanner.global_market import GlobalMarketEngine, PortfolioAdvisor
    from core.scanner.daily_scanner import DailyScannerEngine
except ImportError as e:
    st.error(f"核心模块导入失败，请检查项目路径或依赖。错误信息: {e}")
    st.stop()


# 页面全局样式
st.markdown("""
<style>
    .metric-value { font-size: 24px; font-weight: bold; }
    .metric-label { font-size: 14px; color: #888; }
    .up-red { color: #ff4b4b; font-weight: bold; }
    .down-green { color: #00d26a; font-weight: bold; }
    .stDataFrame { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    db_path = os.path.join(project_root, 'data', 'echomarsh.db')
    return PersistenceManager(db_path)

db = get_db()
global_market = GlobalMarketEngine()

# --- 侧边栏导航 ---
st.sidebar.title("⚡ EchoMarsh")
st.sidebar.markdown("鸣泽量化管理终端")
page = st.sidebar.radio("导航菜单", ["📊 市场看板", "📝 每日推荐", "💼 持仓与交易管理"])


# ==========================================
# 页面 1：市场看板
# ==========================================
if page == "📊 市场看板":
    st.title("📊 全球市场看板")
    st.markdown("实时追踪外围指数与重要财经快讯")
    
    if st.button("🔄 刷新外围数据"):
        st.cache_data.clear()

    @st.cache_data(ttl=300)
    def load_market_data():
        try:
            return global_market.fetch_global_indices()
        except Exception as e:
            st.error(f"获取指数失败: {e}")
            return []

    indices = load_market_data()
    
    if indices:
        cols = st.columns(len(indices))
        for i, idx in enumerate(indices):
            with cols[i]:
                name = idx.get('name', 'N/A')
                price = idx.get('close', 0.0)
                pct = idx.get('pct_change', 0.0)
                
                # A股习惯：红涨绿跌
                color_class = "up-red" if pct > 0 else "down-green" if pct < 0 else ""
                sign = "+" if pct > 0 else ""
                
                st.markdown(f"<div class='metric-label'>{name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value {color_class}'>{price:.2f} <span style='font-size:16px;'>({sign}{pct:.2f}%)</span></div>", unsafe_allow_html=True)
    else:
        st.warning("暂无外围指数数据。")

    st.markdown("---")
    st.subheader("📰 财联社 24 小时电报")
    
    @st.cache_data(ttl=600)
    def load_news():
        try:
            return global_market.fetch_cls_telegraph(limit=15)
        except Exception as e:
            st.error(f"获取新闻失败: {e}")
            return []

    news_list = load_news()
    for news in news_list:
        time_str = news.get('time', '')
        title = news.get('title', '')
        content = news.get('content', '')
        
        with st.expander(f"🕒 {time_str} | **{title}**"):
            st.write(content)


# ==========================================
# 页面 2：每日推荐
# ==========================================
elif page == "📝 每日推荐":
    st.title("📝 每日量化推荐")
    st.markdown("基于 12 因子模型的 A 股高胜率标的扫描与历史查询")
    
    rec_tab1, rec_tab2, rec_tab3 = st.tabs(["🚀 今日扫描", "📚 历史记录", "📈 分数对比"])
    
    with rec_tab1:
        date_str = st.date_input("选择扫描日期", datetime.today()).strftime('%Y%m%d')
        board = st.selectbox("板块过滤", ["主板 (00/60)", "全市场"], index=0)
        board_code = "main" if "主板" in board else "all"
        
        if st.button("🚀 开始扫描 (耗时较长)"):
            with st.spinner("正在获取全市场数据及进行多因子评分..."):
                try:
                    engine = DailyScannerEngine(board_filter=board_code)
                    scored_stocks, market = engine.scan(date_str)
                    
                    if not scored_stocks:
                        st.warning("今日无符合条件的推荐标的。")
                    else:
                        st.success(f"扫描完成！大盘情绪: {market.emoji} {market.label_cn} | 找到 {len(scored_stocks)} 只标的。")
                        
                        # 存入数据库
                        db.save_recommendations(date_str, scored_stocks, market.label_cn)
                        
                        # 准备表格数据
                        table_data = []
                        for s in scored_stocks:
                            reasons = " | ".join([r.detail for r in s.top_reasons(2)])
                            tags = ",".join(s.sentiment_tags) if s.sentiment_tags else "-"
                            
                            table_data.append({
                                "代码": s.code,
                                "名称": s.name,
                                "行业": s.industry,
                                "综合分": f"{s.total_score:.1f}",
                                "连板": s.lianban,
                                "情绪标签": tags,
                                "核心理由": reasons
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # 详细理由
                        st.subheader("💡 详细标的分析")
                        for s in scored_stocks:
                            with st.expander(f"[{s.total_score:.0f}分] {s.name} ({s.code}) - {s.industry}"):
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown("**✅ 推荐理由:**")
                                    for r in s.top_reasons(5):
                                        st.markdown(f"- {r.detail}")
                                with c2:
                                    st.markdown("**⚠️ 风险提示:**")
                                    warnings = s.risk_warnings()
                                    if warnings:
                                        for w in warnings:
                                            st.markdown(f"- <span style='color:red;'>{w.detail}</span>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("- 暂无明显风险指标")
                                        
                except Exception as e:
                    st.error(f"扫描过程中发生错误: {e}")

    with rec_tab2:
        st.subheader("历史推荐记录查询")
        try:
            recent_recs = db.get_recent_recommendations(days=10)
            if not recent_recs:
                st.info("暂无历史推荐记录。")
            else:
                selected_date = st.selectbox("选择历史日期", list(recent_recs.keys()))
                records = recent_recs[selected_date]
                if records:
                    st.write(f"**{selected_date}** 推荐标的 (共 {len(records)} 只):")
                    df_history = pd.DataFrame(records)
                    st.dataframe(df_history, use_container_width=True)
                else:
                    st.warning("该日期无推荐记录。")
        except Exception as e:
            st.error(f"读取历史记录失败: {e}")

    with rec_tab3:
        st.subheader("每日分数对比")
        st.markdown("查看每日预测分数变化，对比前一日涨跌。")

        dates = db.get_prediction_dates(limit=10)
        if len(dates) < 1:
            st.info("暂无预测数据。请先运行 daily_update.py。")
        else:
            selected = st.selectbox("选择对比日期", dates, index=0)
            prev_dates = [d for d in dates if d < selected]
            compare_date = prev_dates[0] if prev_dates else None

            if compare_date:
                st.write(f"**{selected}** vs **{compare_date}**")
                changes = db.get_prediction_with_changes(selected, compare_date)
                if changes:
                    rows = []
                    for r in changes[:30]:
                        chg = r.get('change')
                        if chg is not None:
                            arrow = "+" if chg > 0 else ("-" if chg < 0 else "=")
                            chg_display = f"{arrow} {chg:+.1f}"
                        else:
                            chg_display = "新"
                        rows.append({
                            "代码": r['code'], "名称": r['name'],
                            "综合分": r['combined_score'],
                            "前日分": r.get('prev_score', '-'),
                            "变化": chg_display, "连板": r.get('lianban', 0),
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    st.caption("+ 分数上升  - 分数下降  = 持平  新=新入选")
                else:
                    st.warning("该日无预测数据。")
            else:
                st.info("仅有一日数据，无可对比的前日记录。")
                records = db.get_predictions(selected)
                if records:
                    st.dataframe(pd.DataFrame(records), use_container_width=True)


# ==========================================
# 页面 3：持仓与交易管理
# ==========================================
elif page == "💼 持仓与交易管理":
    st.title("💼 持仓与交易管理")
    
    tab1, tab2, tab3 = st.tabs(["📊 当前持仓与建议", "➕ 记录交易", "📜 历史记录"])
    
    # --- Tab 1: 当前持仓 ---
    with tab1:
        st.subheader("当前持仓快照")
        positions = db.get_positions()
        
        if not positions:
            st.info("当前无活动持仓。")
        else:
            advisor = PortfolioAdvisor()
            
            for p in positions:
                # p 是 PositionRecord
                code, name, vol, cost = p.code, p.name, p.volume, p.cost_price
                
                # 获取实时价格 (尝试从 akshare 拿最新)
                try:
                    import akshare as ak
                    spot = ak.stock_zh_a_spot_em()
                    row = spot[spot['代码'] == code]
                    if not row.empty:
                        current_price = float(row.iloc[0]['最新价'])
                        pct_change = float(row.iloc[0]['涨跌幅'])
                    else:
                        current_price = cost # fallback
                        pct_change = 0.0
                except:
                    current_price = cost
                    pct_change = 0.0
                
                # 计算盈亏
                profit_amount = (current_price - cost) * vol
                profit_pct = (current_price / cost - 1) * 100 if cost > 0 else 0
                
                # 顾问建议
                advice = advisor.advise_position(code, name, cost, current_price, vol)
                
                color = "up-red" if profit_amount > 0 else "down-green" if profit_amount < 0 else ""
                
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                    col1.markdown(f"**{name}** ({code})")
                    col1.caption(f"持股: {vol} 股 | 成本: {cost:.3f}")
                    
                    col2.metric("现价", f"{current_price:.2f}", f"{pct_change:.2f}%")
                    
                    # Streamlit metric delta is green for up by default. In A-shares, red is up.
                    # We use custom HTML for A-share color logic
                    col3.markdown(f"浮动盈亏:<br><span class='{color}'>{profit_amount:.2f} ({profit_pct:.2f}%)</span>", unsafe_allow_html=True)
                    
                    if "SELL" in advice['action']:
                        adv_color = "red"
                    elif "HOLD" in advice['action']:
                        adv_color = "orange"
                    else:
                        adv_color = "green"
                    
                    col4.markdown(f"**智能建议: <span style='color:{adv_color};'>{advice['action']}</span>**", unsafe_allow_html=True)
                    col4.caption(advice['reason'])
                    
                st.markdown("---")

    # --- Tab 2: 记录交易 ---
    with tab2:
        st.subheader("录入买卖记录")
        with st.form("trade_form"):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                action = st.selectbox("交易方向", ["BUY", "SELL"])
                code = st.text_input("股票代码 (如: 000001)", max_chars=6)
                name = st.text_input("股票名称 (如: 平安银行)")
            with t_col2:
                price = st.number_input("成交价格", min_value=0.01, value=10.00, step=0.01)
                volume = st.number_input("成交数量 (股)", min_value=100, value=100, step=100)
                reason = st.text_input("交易备注 (如: 打板买入 / 止损出局)")
            
            date = st.date_input("交易日期")
            
            submit = st.form_submit_button("💾 记录交易")
            if submit:
                if len(code) == 6 and name:
                    try:
                        date_str = date.strftime('%Y%m%d')
                        db.add_trade(code, name, action, price, volume, reason, date_str)
                        st.success(f"已成功记录 {action} {name} {volume}股！")
                    except Exception as e:
                        st.error(f"记录失败: {e}")
                else:
                    st.error("请填写正确的代码和名称。")

    # --- Tab 3: 历史记录 ---
    with tab3:
        st.subheader("近期交易记录")
        try:
            trades = db.get_trades(days=5)
            
            if trades:
                df_trades = pd.DataFrame(trades)
                st.dataframe(df_trades[['trade_date', 'trade_time', 'code', 'name', 'direction', 'price', 'volume', 'amount', 'reason']], use_container_width=True)
            else:
                st.info("暂无交易记录。")
        except Exception as e:
            st.error(f"读取记录失败: {e}")
