# Quantify 量化分析API

## 项目简介
本项目是一个基于 Flask 的加密货币市场数据与AI技术分析API，支持获取币安市场K线数据、技术指标（SMA、RSI、MACD），并可调用 Gemini 或 DeepSeek 大模型自动生成专业的技术面分析。

## 主要功能
- 获取币安市场历史K线数据（支持模拟数据）
- 计算主流技术指标（SMA、RSI、MACD）
- 提供 `/indicators` 接口返回最新100条数据与指标
- 提供 `/analyze` 接口，结合AI大模型自动生成技术分析报告（简体中文）

## 安装依赖
建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

## 环境变量配置
请在项目根目录下创建 `.env` 文件，内容示例：

```
GEMINI_API_KEY=你的Gemini API Key
BINANCE_API_KEY=你的Binance API Key（可选）
BINANCE_API_SECRET=你的Binance API Secret（可选）
DEEPSEEK_API_KEY=你的DeepSeek API Key（可选）
```

> 仅使用 Gemini 时，Binance 和 DeepSeek 的 Key 可留空。

## 如何运行

```bash
python quantify.py
```

默认服务运行在 `http://127.0.0.1:5001`。

## API接口说明

### 1. 获取技术指标
- **接口**：`/indicators`
- **方法**：GET
- **参数**：`symbol`（如 BTCUSDT）
- **示例**：
  ```
  http://127.0.0.1:5001/indicators?symbol=BTCUSDT
  ```
- **返回**：最新100条K线与技术指标（JSON数组）

### 2. AI技术分析
- **接口**：`/analyze`
- **方法**：GET
- **参数**：
  - `symbol`（如 BTCUSDT，必填）
  - `provider`（可选，默认 gemini，可选 deepseek）
- **示例**：
  ```
  http://127.0.0.1:5001/analyze?symbol=BTCUSDT&provider=gemini
  ```
- **返回**：包含最新指标与AI生成的技术分析（JSON对象）

## 常见问题与注意事项
- 若无法访问币安API，系统将自动使用模拟数据。
- 若端口被占用，请关闭占用5001端口的进程或修改代码中的端口号。
- 返回的 `ai_analysis` 字段为纯文本，前端展示时请自行处理换行符（如将 `\n` 替换为 `<br>`）。
- 仅供学习与研究使用，勿用于实际投资决策。

---

如有问题欢迎反馈！ 
