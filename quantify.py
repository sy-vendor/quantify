import os
from flask import Flask, request, jsonify, make_response
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from datetime import datetime, timedelta
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import json  # 添加在文件顶部的导入部分

# --- 1. 初始化与设置 ---

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化 Flask 应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 确保 JSON 响应支持中文
app.json.ensure_ascii = False  # 确保 JSON 响应支持中文

# 从环境变量读取 API 密钥
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")

if not gemini_api_key:
    raise ValueError("请在 .env 文件中设置 GEMINI_API_KEY")

# 初始化币安客户端
try:
    # 使用测试网络
    binance_client = Client(binance_api_key, binance_api_secret, testnet=True)
    # 测试连接
    binance_client.ping()
    print("成功连接到币安测试网络")
except Exception as e:
    print(f"警告: 币安 API 初始化失败: {e}")
    print("尝试使用公共 API 端点...")
    try:
        # 尝试使用公共 API 端点
        binance_client = Client("", "", testnet=True)
        binance_client.ping()
        print("成功连接到币安公共 API")
    except Exception as e:
        print(f"警告: 公共 API 连接也失败: {e}")
        binance_client = None

# 初始化 DeepSeek 客户端 (使用 OpenAI 套件)
# if deepseek_api_key:
#     deepseek_client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
# else:
deepseek_client = None

# 设置 Gemini API
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')


# --- 2. 辅助函数 (Helpers) ---

def get_market_data_with_indicators(symbol: str) -> pd.DataFrame:
    """从币安获取数据并计算技术指标，返回一个 DataFrame。"""
    if not binance_client:
        print("使用模拟数据...")
        # 生成模拟数据
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        data = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Open': np.random.normal(50000, 1000, 365),
            'High': np.random.normal(51000, 1000, 365),
            'Low': np.random.normal(49000, 1000, 365),
            'Close': np.random.normal(50000, 1000, 365),
            'Volume': np.random.normal(1000, 100, 365)
        })
    else:
        try:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%d %b, %Y")
            klines = binance_client.get_historical_klines(symbol.upper(), Client.KLINE_INTERVAL_1DAY, start_date)
            if not klines: 
                raise ValueError(f"无法获取 {symbol} 的历史数据")

            columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
            data = pd.DataFrame(klines, columns=columns)
            
            data = data[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data['Date'] = pd.to_datetime(data['Open time'], unit='ms').dt.strftime('%Y-%m-%d')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col])
            data.drop('Open time', axis=1, inplace=True)
        except Exception as e:
            print(f"获取真实数据失败: {e}")
            print("使用模拟数据...")
            # 生成模拟数据
            dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
            data = pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'Open': np.random.normal(50000, 1000, 365),
                'High': np.random.normal(51000, 1000, 365),
                'Low': np.random.normal(49000, 1000, 365),
                'Close': np.random.normal(50000, 1000, 365),
                'Volume': np.random.normal(1000, 100, 365)
            })

    # 计算技术指标
    # 使用 pandas_ta 的 ta 扩展来计算指标
    sma20 = data.ta.sma(length=20)
    sma50 = data.ta.sma(length=50)
    rsi = data.ta.rsi(length=14)
    macd = data.ta.macd(fast=12, slow=26, signal=9)
    
    # 打印所有指标的列名，用于调试
    print("SMA20 columns:", sma20.columns.tolist())
    print("SMA50 columns:", sma50.columns.tolist())
    print("RSI columns:", rsi.columns.tolist())
    print("MACD columns:", macd.columns.tolist())
    
    # 将指标添加到数据框
    data['SMA_20'] = sma20.iloc[:, 0]  # 使用第一列
    data['SMA_50'] = sma50.iloc[:, 0]  # 使用第一列
    data['RSI_14'] = rsi.iloc[:, 0]    # 使用第一列
    data['MACD'] = macd.iloc[:, 0]     # MACD 线
    data['MACD_Signal'] = macd.iloc[:, 1]  # 信号线
    
    data.dropna(inplace=True)
    return data

def call_deepseek(prompt: str) -> str:
    """调用 DeepSeek API。"""
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用 DeepSeek 时发生错误: {e}"

def call_gemini(prompt: str) -> str:
    """调用 Gemini API。"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"调用 Gemini 时发生错误: {e}"


# --- 3. API 端点 (Routes) ---

@app.route('/')
def index():
    return "市场数据与AI分析 API 已启动。请使用 /indicators 或 /analyze 端点。"

@app.route('/indicators', methods=['GET'])
def get_indicators_endpoint():
    """仅获取数据和指标。"""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "必须提供 'symbol' 参数"}), 400
        
    data = get_market_data_with_indicators(symbol)
    if data is None or data.empty:
        return jsonify({"error": f"找不到 '{symbol}' 的数据或计算失败。"}), 404
        
    result_df = data.tail(100)
    return jsonify(result_df.to_dict(orient='records'))

@app.route('/analyze', methods=['GET'])
def analyze_market_endpoint():
    """获取数据、指标，并交由指定的 AI 模型进行分析。"""
    symbol = request.args.get('symbol')
    provider = request.args.get('provider', 'gemini').lower() # 默认使用 gemini

    if not symbol:
        return jsonify({"error": "必须提供交易对 'symbol' 参数"}), 400
    if provider not in ['deepseek', 'gemini']:
        return jsonify({"error": "提供者 'provider' 必须是 'deepseek' 或 'gemini'"}), 400
    if provider == 'deepseek' and not deepseek_client:
        return jsonify({"error": "DeepSeek API 未配置，请使用 'gemini' 作为提供者"}), 400

    # 1. 获取数据
    data = get_market_data_with_indicators(symbol)
    if data is None or data.empty:
        return jsonify({"error": f"找不到 '{symbol}' 的数据，无法进行分析。"}), 404
    
    # 2. 准备 Prompt
    latest_data = data.iloc[-1]
    prompt = f"""
    您是一位专精于加密货币的资深技术分析师。
    请根据以下提供的 {symbol.upper()} 最新单日K线数据，提供简洁、专业的技术面分析。

    **最新数据指标:**
    - **日期:** {latest_data['Date']}
    - **收盘价:** {latest_data['Close']:.4f}
    - **20日均线 (SMA_20):** {latest_data['SMA_20']:.4f}
    - **50日均线 (SMA_50):** {latest_data['SMA_50']:.4f}
    - **14日相对强弱指数 (RSI_14):** {latest_data['RSI_14']:.2f}
    - **MACD 指标:** MACD线({latest_data['MACD']:.4f}), 信号线({latest_data['MACD_Signal']:.4f})

    **分析要求 (请用简体中文回答):**
    1. **市场趋势:** 根据价格与MA均线的关系，判断目前是处于多头、空头还是盘整趋势？
    2. **市场动能:** RSI值显示市场是处于超买、超卖还是中性区间？
    3. **趋势信号:** MACD指标目前呈现了什么信号（例如：黄金交叉、死亡交叉）？
    4. **综合评论:** 综合以上指标，给出一个简短的整体技术面前景总结。
    """

    # 3. 根据 provider 选择调用的函数
    analysis_text = ""
    if provider == 'deepseek':
        analysis_text = call_deepseek(prompt)
    else: # provider == 'gemini'
        analysis_text = call_gemini(prompt)

    # 4. 组合最终的回应
    # 使用 \r\n 作为换行符
    formatted_analysis = analysis_text.replace('\n', '\r\n')
    
    response_data = {
        "symbol": symbol.upper(),
        "analysis_provider": provider,
        "latest_indicators": latest_data.to_dict(),
        "ai_analysis": formatted_analysis
    }
    
    # 使用 jsonify 并设置响应头
    response = jsonify(response_data)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    
    return response


# --- 4. 启动应用 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
