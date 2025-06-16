import os
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.json.ensure_ascii = False

# 初始化 Gemini AI
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def generate_mock_data(symbol, interval, limit):
    """生成模拟的K线数据"""
    logger.info(f"生成模拟数据: {symbol}, {interval}, {limit}")
    
    # 生成时间序列
    end_time = datetime.now()
    if interval == '1d':
        delta = timedelta(days=1)
    elif interval == '4h':
        delta = timedelta(hours=4)
    elif interval == '1h':
        delta = timedelta(hours=1)
    else:
        delta = timedelta(days=1)
    
    timestamps = [end_time - delta * i for i in range(limit)]
    timestamps.reverse()
    
    # 生成价格数据
    base_price = 1000  # 基础价格
    volatility = 0.02  # 波动率
    
    prices = []
    current_price = base_price
    for _ in range(limit):
        change = np.random.normal(0, volatility)
        current_price *= (1 + change)
        prices.append(current_price)
    
    # 生成成交量
    volumes = np.random.normal(1000, 200, limit)
    volumes = np.abs(volumes)  # 确保成交量为正
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'volume': volumes
    })
    
    return df

def get_market_data(symbol, interval, limit):
    """获取市场数据（使用模拟数据）"""
    logger.info(f"获取市场数据: {symbol}, {interval}, {limit}")
    return generate_mock_data(symbol, interval, limit)

def calculate_indicators(df, extra_indicators=None):
    """计算技术指标"""
    try:
        # 基础指标
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        
        # 额外指标
        if extra_indicators:
            for indicator in extra_indicators:
                if indicator == 'BB':
                    bollinger = ta.bbands(df['close'])
                    df['BB_Upper'] = bollinger['BBU_20_2.0']
                    df['BB_Middle'] = bollinger['BBM_20_2.0']
                    df['BB_Lower'] = bollinger['BBL_20_2.0']
                elif indicator == 'ATR':
                    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])
                elif indicator == 'OBV':
                    df['OBV'] = ta.obv(df['close'], df['volume'])
                    
        return df
    except Exception as e:
        logger.error(f"计算技术指标失败: {str(e)}")
        raise

def analyze_market_data(df, symbol):
    """分析市场数据"""
    try:
        # 准备数据
        latest_data = df.iloc[-1].to_dict()
        latest_data['symbol'] = symbol
        
        # 使用 Gemini 进行分析
        prompt = f"""
请分析以下加密货币市场数据并提供交易建议：

交易对：{symbol}
最新价格：{latest_data['close']:.2f}
20日均线：{latest_data['SMA_20']:.2f}
50日均线：{latest_data['SMA_50']:.2f}
RSI(14)：{latest_data['RSI_14']:.2f}
MACD：{latest_data['MACD']:.2f}
MACD信号线：{latest_data['MACD_Signal']:.2f}

请提供：
1. 市场趋势分析
2. 技术指标解读
3. 交易建议
4. 风险提示
"""
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"AI 分析失败: {str(e)}")
        return f"""
由于 AI 服务暂时不可用，以下是基于技术指标的基本分析：

交易对：{symbol}
最新价格：{latest_data['close']:.2f}
20日均线：{latest_data['SMA_20']:.2f}
50日均线：{latest_data['SMA_50']:.2f}
RSI(14)：{latest_data['RSI_14']:.2f}
MACD：{latest_data['MACD']:.2f}
MACD信号线：{latest_data['MACD_Signal']:.2f}

请注意：这是基于模拟数据的分析，仅供参考。
"""

@app.route('/analyze', methods=['GET'])
def analyze():
    """分析接口"""
    try:
        # 获取参数
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1d')
        limit = int(request.args.get('limit', 100))
        extra_indicators = request.args.get('extra_indicators', '').split(',') if request.args.get('extra_indicators') else None
        
        # 获取市场数据
        df = get_market_data(symbol, interval, limit)
        
        # 计算技术指标
        df = calculate_indicators(df, extra_indicators)
        
        # 分析数据
        analysis = analyze_market_data(df, symbol)
        
        # 准备响应数据
        response_data = {
            'symbol': symbol,
            'interval': interval,
            'latest_data': df.iloc[-1].to_dict(),
            'ai_analysis': analysis
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"分析请求失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
