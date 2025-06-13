import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import openai
import google.generativeai as genai

# --- 1. 初始化與設定 ---

# 從 .env 檔案載入環境變數
load_dotenv()

# 初始化 Flask 應用
app = Flask(__name__)
# 確保 JSON 回應能正確顯示中文
app.config['JSON_AS_ASCII'] = False

# --- API 金鑰與客戶端設定 ---

# 從環境變數讀取金鑰
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")

# 檢查必要的 Gemini 金鑰
if not gemini_api_key:
    raise ValueError("錯誤: 必須在 .env 檔案中設定 GEMINI_API_KEY")

# 初始化幣安客戶端，包含錯誤處理與回退機制
binance_client = None
try:
    # 優先使用提供的 API 金鑰連接到測試網路
    if binance_api_key and binance_api_secret:
        client = Client(binance_api_key, binance_api_secret, testnet=True)
        client.ping()
        binance_client = client
        print("成功連接到幣安測試網路 (使用 API Key)。")
    else:
        # 如果沒有提供金鑰，嘗試使用公共端點連接
        client = Client("", "", testnet=True)
        client.ping()
        binance_client = client
        print("成功連接到幣安公共 API (無需 API Key)。")
except Exception as e:
    print(f"警告: 連接幣安 API 失敗: {e}。所有數據請求將使用模擬數據。")

# 初始化 AI 客戶端
deepseek_client = None
if deepseek_api_key:
    deepseek_client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
else:
    print("警告: 未設定 DEEPSEEK_API_KEY，DeepSeek 相關功能將不可用。")

genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- 2. 輔助函式 (Helpers) ---

def create_mock_data() -> pd.DataFrame:
    """當無法獲取真實數據時，生成模擬數據。"""
    print("回退機制: 正在生成模擬數據...")
    dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=365, freq='D'))
    data = pd.DataFrame({'Date': dates})
    data['Open'] = np.random.normal(loc=50000, scale=1500, size=365).cumsum() + 50000
    data['High'] = data['Open'] + np.random.uniform(0, 2000, 365)
    data['Low'] = data['Open'] - np.random.uniform(0, 2000, 365)
    data['Close'] = (data['High'] + data['Low']) / 2
    data['Volume'] = np.random.normal(loc=1000, scale=200, size=365)
    data.set_index('Date', inplace=True)
    return data

def get_market_data_with_indicators(symbol: str) -> pd.DataFrame:
    """從幣安獲取數據並使用策略計算技術指標。"""
    data = None
    if binance_client:
        try:
            start_date = (datetime.now() - timedelta(days=500)).strftime("%d %b, %Y") # 多取一些數據以確保指標計算有足夠前期數據
            klines = binance_client.get_historical_klines(symbol.upper(), Client.KLINE_INTERVAL_1DAY, start_date)
            if not klines: raise ValueError("API 返回空的 K線數據")

            cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
            df = pd.DataFrame(klines, columns=cols)
            df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
            df.set_index('Date', inplace=True)
            
            # 轉換數據類型為數值
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            data = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            print(f"從幣安獲取 '{symbol}' 數據失敗: {e}")
            data = create_mock_data()
    else:
        data = create_mock_data()

    # 定義一個包含所有所需指標的策略
    MyStrategy = ta.Strategy(
        name="常用指標組合",
        description="計算 SMA(20, 50), RSI(14), 和 MACD(12, 26, 9)",
        ta=[
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50},
            {"kind": "rsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        ]
    )
    
    # 一行程式碼應用策略，自動計算並附加所有指標
    data.ta.strategy(MyStrategy)
    
    # 清理包含 NaN 的行 (通常是數據開頭的部分)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    return data

def call_ai_provider(prompt: str, provider: str) -> str:
    """根據提供者名稱呼叫對應的 AI 模型。"""
    if provider == 'deepseek':
        if not deepseek_client:
            return "錯誤: DeepSeek API 未被正確設定。"
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"呼叫 DeepSeek API 時發生錯誤: {e}"
    
    elif provider == 'gemini':
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"呼叫 Gemini API 時發生錯誤: {e}"
            
    return "錯誤: 無效的 AI 提供者。"


# --- 3. API 端點 (Routes) ---

@app.route('/')
def index():
    return "<h2>市場羅盤 API v2.0</h2><p>已成功啟動。請使用 <code>/indicators</code> 或 <code>/analyze</code> 端點。</p>"

@app.route('/indicators', methods=['GET'])
def get_indicators_endpoint():
    """僅獲取數據和指標。"""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "必須提供 'symbol' 參數"}), 400
        
    data = get_market_data_with_indicators(symbol)
    if data is None or data.empty:
        return jsonify({"error": f"找不到 '{symbol}' 的數據或計算失敗。"}), 404
        
    # 回傳最新的 100 筆數據
    return jsonify(json.loads(data.tail(100).to_json(orient='records')))

@app.route('/analyze', methods=['GET'])
def analyze_market_endpoint():
    """獲取數據、指標，並交由指定的 AI 模型進行分析。"""
    symbol = request.args.get('symbol')
    provider = request.args.get('provider', 'gemini').lower() # 預設使用 gemini

    if not symbol:
        return jsonify({"error": "必須提供交易對 'symbol' 參數"}), 400
    if provider not in ['deepseek', 'gemini']:
        return jsonify({"error": "提供者 'provider' 必須是 'deepseek' 或 'gemini'"}), 400
    if provider == 'deepseek' and not deepseek_client:
        return jsonify({"error": "DeepSeek API 未配置，請使用 'gemini' 作為提供者"}), 400

    data = get_market_data_with_indicators(symbol)
    if data is None or data.empty:
        return jsonify({"error": f"找不到 '{symbol}' 的數據，無法進行分析。"}), 404
    
    latest_data = data.iloc[-1]
    
    # 建立全新的、高度結構化的詳細分析提示詞模板
    prompt = f"""
    您是一位專精於加密貨幣的資深技術分析師，擅長撰寫結構清晰、極具洞察力的市場報告。

    請根據以下提供的 **{symbol.upper()}** 最新單日K線數據，並結合您對市場的理解，生成一份詳細的市場分析報告。請嚴格遵循下方的七大模塊結構，並使用簡體中文回答。

    ---
    ### **提供給您的最新數據指標:**
    - **日期:** {latest_data['Date']}
    - **收盤價:** {latest_data['Close']:.4f}
    - **成交量:** {latest_data['Volume']:.2f}
    - **20日均線 (SMA_20):** {latest_data.get('SMA_20', 'N/A'):.4f}
    - **50日均線 (SMA_50):** {latest_data.get('SMA_50', 'N/A'):.4f}
    - **14日相對強弱指數 (RSI_14):** {latest_data.get('RSI_14', 'N/A'):.2f}
    - **MACD 指標 (12, 26, 9):**
        - MACD線 (快線): {latest_data.get('MACD_12_26_9', 'N/A'):.4f}
        - 訊號線 (慢線): {latest_data.get('MACDs_12_26_9', 'N/A'):.4f}
        - 柱狀圖 (動能): {latest_data.get('MACDh_12_26_9', 'N/A'):.4f}
    ---

    ### **分析報告生成要求 (请严格遵循此结构):**

    ### 一、六大致命信号
    *请建立一个 Markdown 表格分析以下指标。如果某些数据（如买盘深度、OBV、持仓结构）无法从提供的数据中直接获取，请基于您的模型知识进行推断或标记为「需实时数据」。*
    | **指标** | **当前值/状态** | **含义与风险** |
    |---|---|---|
    | **价格** | {latest_data['Close']:.4f} | (请分析价格走势，例如是否接近关键支撑/阻力) |
    | **成交量** | {latest_data['Volume']:.1f} | (请分析成交量变化，是放量还是缩量，代表什么) |
    | **买盘深度** | (推断或标记) | (请分析订单簿的健康状况) |
    | **OBV** | (推断或标记) | (请分析资金流向) |
    | **L.S结构** | (推断或标记) | (请分析多空持仓比例结构) |
    | **O.I.数据** | (推断或标记) | (请分析未平仓合约数据及其风险) |

    ---

    ### 二、技术面：(请判断当前通道趋势，如“下跌通道加速”)
    #### 关键价格锚点
    *请列出并分析当前最重要的几个价格水平。*
    ```
    (例如: 8.000 ──┤ 周线阻力)
    (例如: {latest_data['Close']:.4f} ──┤ 当前价)
    (例如: 6.500 ──┤ 关键支撑)
    ```
    - **均线系统**: (请分析 MA(20), MA(50) 的排列和交叉情况，例如是否形成“瀑布式下压”)
    - **终极警告**: (基于以上分析，给出一个最关键的风险提示)

    ---

    ### 三、主力行为解密：(请描述当前主力可能的意图，例如“请君入瓮”)
    #### (例如：四步猎杀陷阱)
    *请使用 Mermaid `graph TB` 流程图分析主力可能的行为路径。*
    ```mermaid
    graph TB
    A[...] --> B[...]
    B --> C[...]
    C --> D[...]
    ```
    **当前阶段**: (请结合数据分析目前处于哪个阶段)

    ---

    ### 四、行情推演与概率
    *请建立一个 Markdown 表格，推演未来最可能发生的三种情景。*
    | **场景** | **概率** | **路径** | **触发条件** |
    |---|---|---|---|
    | **(情景1)** | (例如: 75%) | (描述价格可能路径) | (描述触发条件) |
    | **(情景2)** | (例如: 20%) | (描述价格可能路径) | (描述触发条件) |
    | **(情景3)** | (例如: 5%) | (描述价格可能路径) | (描述触发条件) |

    ---

    ### 五、紧急生存指南
    #### （1）持有多单者
    - **立即行动**: (给出明确操作建议)
    #### （2）持有空单者
    - **止盈策略**: (给出明确操作建议)
    #### （3）空仓者
    - **操作建议**: (给出明确操作建议，例如“绝对禁止抄底”或在何处开空)

    ---

    ### 六、末日观测锚点
    *请列出 2-3 个需要密切关注的关键指标或事件。*
    1. **(观测点1)**: (描述内容)
    2. **(观测点2)**: (描述内容)

    ---

    ### 七、血泪教训总结
    *请总结 2-3 个从当前行情中可以学到的关键教训。*
    1. **(教训1)**: (描述内容)
    2. **(教训2)**: (描述内容)

    > ☠️ **终极警告**:
    > (给出一个总结性的、最强烈的风险警告)
    > **生存法则**:
    > *"(用一句话总结生存法则)"*
    """

    analysis_text = call_ai_provider(prompt, provider)

    response_data = {
        "symbol": symbol.upper(),
        "analysis_provider": provider,
        "latest_indicators": json.loads(latest_data.to_json()),
        "ai_analysis": analysis_text
    }
    
    response = jsonify(response_data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    return response

# --- 4. 啟動應用 ---
if __name__ == '__main__':
    # 建議使用 gunicorn 等生產級伺服器部署，debug=True 僅適用於開發環境
    app.run(host='0.0.0.0', port=5001, debug=True)

