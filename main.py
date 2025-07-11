import requests
import json
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app untuk API endpoint
app = Flask(__name__)

class TradingBot:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        
    def get_market_data(self, symbol: str, interval: str = "5min", count: int = 50) -> Optional[pd.DataFrame]:
        """Ambil data market dari Twelvedata"""
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': count
            }
            
            response = self.session.get(f"{self.base_url}/time_series", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'values' not in data:
                logger.error(f"No data received for {symbol}")
                return None
                
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def get_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Ambil RSI terkini"""
        try:
            params = {
                'symbol': symbol,
                'interval': '5min',
                'time_period': period,
                'apikey': self.api_key
            }
            
            response = self.session.get(f"{self.base_url}/rsi", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'values' in data and len(data['values']) > 0:
                return float(data['values'][0]['rsi'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting RSI: {e}")
            return None
    
    def get_macd(self, symbol: str) -> Optional[Dict]:
        """Ambil MACD terkini"""
        try:
            params = {
                'symbol': symbol,
                'interval': '5min',
                'apikey': self.api_key
            }
            
            response = self.session.get(f"{self.base_url}/macd", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'values' in data and len(data['values']) > 0:
                latest = data['values'][0]
                return {
                    'macd': float(latest['macd']),
                    'signal': float(latest['macd_signal']),
                    'histogram': float(latest['macd_hist'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting MACD: {e}")
            return None
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Hitung level support dan resistance"""
        if df is None or len(df) < 20:
            return {'support': None, 'resistance': None}
        
        # Ambil 20 candle terakhir
        recent_data = df.tail(20)
        
        # Support = low terendah dari 20 candle
        support = recent_data['low'].min()
        
        # Resistance = high tertinggi dari 20 candle
        resistance = recent_data['high'].max()
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    def generate_signal(self, symbol: str) -> Dict:
        """Generate sinyal trading berdasarkan indikator"""
        try:
            # Ambil data market
            df = self.get_market_data(symbol)
            if df is None:
                return {'error': 'Failed to get market data'}
            
            current_price = df['close'].iloc[-1]
            
            # Ambil indikator
            rsi = self.get_rsi(symbol)
            macd_data = self.get_macd(symbol)
            sr_levels = self.calculate_support_resistance(df)
            
            if rsi is None or macd_data is None:
                return {'error': 'Failed to get indicators'}
            
            # Logika sinyal
            signal = self.analyze_signal(rsi, macd_data, current_price, sr_levels)
            
            if signal['action'] != 'HOLD':
                # Hitung entry, TP, SL
                trade_levels = self.calculate_trade_levels(
                    current_price, 
                    signal['action'], 
                    sr_levels
                )
                signal.update(trade_levels)
            
            signal.update({
                'symbol': symbol,
                'current_price': current_price,
                'rsi': rsi,
                'macd': macd_data,
                'support': sr_levels['support'],
                'resistance': sr_levels['resistance'],
                'timestamp': datetime.now().isoformat()
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {'error': str(e)}
    
    def analyze_signal(self, rsi: float, macd_data: Dict, price: float, sr_levels: Dict) -> Dict:
        """Analisis sinyal berdasarkan multiple indikator"""
        
        # Kondisi BUY
        buy_conditions = [
            rsi < 30,  # RSI oversold
            macd_data['macd'] > macd_data['signal'],  # MACD bullish
            sr_levels['support'] and price > sr_levels['support'] * 1.001  # Di atas support
        ]
        
        # Kondisi SELL
        sell_conditions = [
            rsi > 70,  # RSI overbought
            macd_data['macd'] < macd_data['signal'],  # MACD bearish
            sr_levels['resistance'] and price < sr_levels['resistance'] * 0.999  # Di bawah resistance
        ]
        
        # Hitung kekuatan sinyal
        buy_strength = sum(buy_conditions) / len(buy_conditions)
        sell_strength = sum(sell_conditions) / len(sell_conditions)
        
        # Minimal 2 dari 3 kondisi harus terpenuhi
        if buy_strength >= 0.67:
            return {
                'action': 'BUY',
                'confidence': buy_strength,
                'reason': f"RSI: {rsi:.1f}, MACD: Bullish, Price above support"
            }
        elif sell_strength >= 0.67:
            return {
                'action': 'SELL',
                'confidence': sell_strength,
                'reason': f"RSI: {rsi:.1f}, MACD: Bearish, Price below resistance"
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': "Not enough confluence for signal"
            }
    
    def calculate_trade_levels(self, current_price: float, action: str, sr_levels: Dict) -> Dict:
        """Hitung level entry, TP, SL untuk pending order"""
        
        if action == 'BUY':
            # Buy Limit di support atau sedikit di atas current price
            entry_price = min(
                current_price * 0.9995,  # 0.05% di bawah current
                sr_levels['support'] * 1.001 if sr_levels['support'] else current_price * 0.9995
            )
            
            # SL di bawah support
            sl_price = sr_levels['support'] * 0.999 if sr_levels['support'] else entry_price * 0.995
            
            # TP dengan RR 1:2
            risk = entry_price - sl_price
            tp_price = entry_price + (risk * 2)
            
            return {
                'entry_price': round(entry_price, 2),
                'tp_price': round(tp_price, 2),
                'sl_price': round(sl_price, 2),
                'order_type': 'BUY_LIMIT',
                'pips_tp': round((tp_price - entry_price) * 10000, 0),
                'pips_sl': round((entry_price - sl_price) * 10000, 0),
                'risk_reward': '1:2'
            }
            
        elif action == 'SELL':
            # Sell Limit di resistance atau sedikit di bawah current price
            entry_price = max(
                current_price * 1.0005,  # 0.05% di atas current
                sr_levels['resistance'] * 0.999 if sr_levels['resistance'] else current_price * 1.0005
            )
            
            # SL di atas resistance
            sl_price = sr_levels['resistance'] * 1.001 if sr_levels['resistance'] else entry_price * 1.005
            
            # TP dengan RR 1:2
            risk = sl_price - entry_price
            tp_price = entry_price - (risk * 2)
            
            return {
                'entry_price': round(entry_price, 2),
                'tp_price': round(tp_price, 2),
                'sl_price': round(sl_price, 2),
                'order_type': 'SELL_LIMIT',
                'pips_tp': round((entry_price - tp_price) * 10000, 0),
                'pips_sl': round((sl_price - entry_price) * 10000, 0),
                'risk_reward': '1:2'
            }
        
        return {}

# Inisialisasi bot dengan API key
bot = TradingBot(api_key="7182acfcc3cb43aba4482f1849f5b84e")

# Daftar pair yang didukung
SUPPORTED_PAIRS = {
    'BTCUSD': 'BTC/USD',
    'EURUSD': 'EUR/USD', 
    'XAUUSD': 'XAU/USD',
    'ETHUSD': 'ETH/USD',
    'USDJPY': 'USD/JPY'
}

@app.route('/signal/<symbol>', methods=['GET'])
def get_signal(symbol):
    """Endpoint untuk ambil sinyal trading"""
    try:
        # Convert symbol format jika perlu
        if '/' in symbol:
            symbol = symbol.replace('/', '')
        
        # Validasi pair yang didukung
        if symbol.upper() not in SUPPORTED_PAIRS:
            return jsonify({
                'error': f'Pair {symbol} tidak didukung',
                'supported_pairs': list(SUPPORTED_PAIRS.keys())
            }), 400
        
        signal = bot.generate_signal(symbol.upper())
        return jsonify(signal)
        
    except Exception as e:
        logger.error(f"Error in get_signal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'trading-bot'
    })

@app.route('/signals/all', methods=['GET'])
def get_all_signals():
    """Endpoint untuk ambil semua sinyal dari 5 pair sekaligus"""
    try:
        results = {}
        for symbol in SUPPORTED_PAIRS.keys():
            results[symbol] = bot.generate_signal(symbol)
        
        return jsonify({
            'signals': results,
            'timestamp': datetime.now().isoformat(),
            'total_pairs': len(SUPPORTED_PAIRS)
        })
        
    except Exception as e:
        logger.error(f"Error in get_all_signals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/signals/multiple', methods=['POST'])
def get_multiple_signals():
    """Endpoint untuk ambil multiple sinyal sekaligus"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        # Validasi symbols
        valid_symbols = []
        for symbol in symbols:
            if '/' in symbol:
                symbol = symbol.replace('/', '')
            if symbol.upper() in SUPPORTED_PAIRS:
                valid_symbols.append(symbol.upper())
        
        if not valid_symbols:
            return jsonify({
                'error': 'Tidak ada pair yang valid',
                'supported_pairs': list(SUPPORTED_PAIRS.keys())
            }), 400
        
        results = {}
        for symbol in valid_symbols:
            results[symbol] = bot.generate_signal(symbol)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in get_multiple_signals: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Untuk development
    app.run(debug=True, host='0.0.0.0', port=5000)
