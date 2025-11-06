#!/usr/bin/env python3
"""
Standalone Trade Visualization CLI
Визуализация результатов торговой стратегии с использованием существующего движка проекта

Usage:
    python visualize_cli.py --csv_in data/market.csv --csv_out results/ --top 5 --dates 2025.05.01-2025.10.25
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
import glob
import importlib.util

class BacktestEngineImporter:
    """Автоматический импорт модулей из проекта"""
    
    def __init__(self, script_dir):
        self.script_dir = Path(script_dir)
        self.project_root = self._find_project_root()
        
    def _find_project_root(self):
        """Ищет корневую папку проекта"""
        current = self.script_dir
        
        # Ищем папку с backtest_engine.py или config.py
        for _ in range(5):  # Ищем до 5 уровней вверх
            if (current / 'backtest_engine.py').exists():
                return current
            if (current / 'config.py').exists():
                return current
            if (current / 'src').exists() and (current / 'src' / 'backtest_engine.py').exists():
                return current / 'src'
            parent = current.parent
            if parent == current:
                break
            current = parent
        
        # Если не нашли, используем текущую директорию
        return self.script_dir
    
    def import_module_from_file(self, module_name, file_path):
        """Импорт модуля из файла"""
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            print(f"Warning: Could not import {module_name} from {file_path}: {e}")
        return None
    
    def load_backtest_engine(self):
        """Загружает модуль backtest_engine"""
        possible_paths = [
            self.project_root / 'backtest_engine.py',
            self.project_root / 'src' / 'backtest_engine.py',
            self.project_root / 'engine' / 'backtest_engine.py',
        ]
        
        for path in possible_paths:
            if path.exists():
                return self.import_module_from_file('backtest_engine', path)
        
        return None


class TradeVisualizer:
    """Визуализатор трейдов с использованием проектного движка"""
    
    def __init__(self, csv_in_path, csv_out_path, backtest_engine=None):
        self.csv_in_path = Path(csv_in_path)
        self.csv_out_path = Path(csv_out_path)
        self.backtest_engine = backtest_engine
        
        # Загрузка данных
        self.market_data = self._load_market_data()
        self.optimization_results = self._load_optimization_results()
        
        print(f"✓ Market data loaded: {len(self.market_data)} bars")
        print(f"✓ Optimization results loaded: {len(self.optimization_results['combinations'])} combinations")
    
    def _load_market_data(self):
        """Загрузка рыночных данных"""
        if not self.csv_in_path.exists():
            raise FileNotFoundError(f"Market data file not found: {self.csv_in_path}")
        
        df = pd.read_csv(self.csv_in_path)
        
        # Определяем колонку с датой/временем
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        timestamp_col = None
        for col in df.columns:
            if col.lower() in timestamp_cols:
                timestamp_col = col
                break
        
        if timestamp_col:
            # Проверяем, является ли колонка числовой (Unix timestamp)
            if pd.api.types.is_numeric_dtype(df[timestamp_col]):
                # Пробуем парсить как Unix timestamp в секундах
                df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df[timestamp_col])
        else:
            # Предполагаем, что первая колонка - timestamp
            if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
        
        # Нормализуем названия колонок
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Проверяем наличие OHLCV
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in market data")
        
        df = df.set_index('timestamp').sort_index()
        return df
    
    def _find_csv_files(self, path):
        """Находит CSV файлы в указанной папке"""
        if path.is_file():
            return [path]
        elif path.is_dir():
            return list(path.glob('*.csv'))
        else:
            # Попробуем как pattern
            return list(Path('.').glob(str(path)))
    
    def _load_optimization_results(self):
        """Загрузка результатов оптимизации"""
        csv_files = self._find_csv_files(self.csv_out_path)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {self.csv_out_path}")
        
        # Берем первый файл (или самый свежий)
        csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"Using optimization results: {csv_file.name}")
        
        return self._parse_optimization_csv(csv_file)
    
    def _parse_optimization_csv(self, csv_path):
        """Парсинг CSV с результатами оптимизации"""
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Парсинг Fixed Parameters
        fixed_params = {}
        in_fixed = False
        
        for i, line in enumerate(lines):
            if 'Fixed Parameters' in line:
                in_fixed = True
                continue
            
            if in_fixed:
                if line.strip() and ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) == 2 and 'MA Type' not in line:
                        fixed_params[parts[0].strip()] = parts[1].strip()
                    else:
                        break
        
        # Парсинг комбинаций
        data_start = None
        for i, line in enumerate(lines):
            if 'MA Type' in line and 'MA Length' in line:
                data_start = i
                break
        
        if data_start is None:
            raise ValueError("Could not find combinations table in CSV")
        
        df = pd.read_csv(csv_path, skiprows=data_start)
        df.columns = df.columns.str.strip()
        
        return {
            'fixed_params': fixed_params,
            'combinations': df
        }
    
    def _run_backtest(self, params, date_range=None):
        """
        Запускает бэктест с использованием движка проекта
        Если движок недоступен, использует упрощенную симуляцию
        """
        if self.backtest_engine and hasattr(self.backtest_engine, 'run_strategy'):
            return self._run_project_backtest(params, date_range)
        else:
            return self._run_simple_backtest(params, date_range)
    
    def _run_project_backtest(self, params, date_range):
        """Запуск бэктеста через движок проекта"""
        print("Using project backtest engine")

        # Подготовка данных
        data = self.market_data.copy()

        # Нормализация названий колонок для backtest_engine
        data_normalized = pd.DataFrame()
        data_normalized['Close'] = data['close']
        data_normalized['High'] = data['high']
        data_normalized['Low'] = data['low']
        data_normalized['Open'] = data['open']
        data_normalized['Volume'] = data.get('volume', pd.Series(0, index=data.index))
        data_normalized.index = data.index

        # Создание объекта StrategyParams
        fixed = self.optimization_results['fixed_params']

        strategy_params = self.backtest_engine.StrategyParams(
            use_backtester=True,
            use_date_filter=date_range is not None,
            start=date_range[0] if date_range else None,
            end=date_range[1] if date_range else None,
            ma_type=params.get('MA Type', 'EMA'),
            ma_length=int(float(params.get('MA Length', 50))),
            close_count_long=int(float(params.get('Close Count Long', 3))),
            close_count_short=int(float(params.get('Close Count Short', 3))),
            stop_long_atr=float(params.get('Stop Long X', 1.0)),
            stop_long_rr=float(fixed.get('stopLongRR', 3)),
            stop_long_lp=int(float(params.get('Stop Long LP', 2))),
            stop_short_atr=float(params.get('Stop Short X', 1.0)),
            stop_short_rr=float(fixed.get('stopShortRR', 3)),
            stop_short_lp=int(float(params.get('Stop Short LP', 2))),
            stop_long_max_pct=float(params.get('Stop Long Max %', 0)),
            stop_short_max_pct=float(params.get('Stop Short Max %', 0)),
            stop_long_max_days=int(float(params.get('Stop Long Max Days', 0))),
            stop_short_max_days=int(float(params.get('Stop Short Max Days', 0))),
            trail_rr_long=float(fixed.get('trailRRLong', 1)),
            trail_rr_short=float(fixed.get('trailRRShort', 1)),
            trail_ma_long_type=params.get('Tr L Type', 'T3'),
            trail_ma_long_length=int(float(params.get('Tr L Len', 100))),
            trail_ma_long_offset=float(params.get('Tr L Off', 0.0)),
            trail_ma_short_type=params.get('Tr S Type', 'T3'),
            trail_ma_short_length=int(float(params.get('Tr S Len', 100))),
            trail_ma_short_offset=float(params.get('Tr S Off', 0.0)),
            risk_per_trade_pct=1.0,
            contract_size=1.0,
            commission_rate=0.0005,
        )

        # Запуск стратегии
        result = self.backtest_engine.run_strategy(data_normalized, strategy_params)

        # Преобразование результатов в формат для визуализации
        trades = []
        for trade_record in result.trades:
            # Получаем trail history для визуализации
            trade_direction = trade_record.direction
            trades.append({
                'type': trade_direction,
                'entry_time': trade_record.entry_time,
                'entry_price': trade_record.entry_price,
                'exit_time': trade_record.exit_time,
                'exit_price': trade_record.exit_price,
                'initial_stop': trade_record.entry_price,  # Упрощение
                'trail_history': []  # Будет пустым, так как engine не возвращает историю
            })

        # Пересчет индикаторов для графика (используем данные с фильтром по датам если нужно)
        if date_range:
            start, end = date_range
            data = data[start:end]
            if len(data) == 0:
                print(f"  ⚠ Warning: No data in date range {start} to {end}")
                return [], data, 0.0, 0.0

        # Расчет MA и trail для визуализации
        data['ma'] = self._calculate_ma(data, params.get('MA Type', 'EMA'),
                                        int(float(params.get('MA Length', 50))))
        data['trail_long'] = self._calculate_ma(data, params.get('Tr L Type', 'T3'),
                                                int(float(params.get('Tr L Len', 100)))) * \
                            (1 + float(params.get('Tr L Off', 0.0)) / 100)
        data['trail_short'] = self._calculate_ma(data, params.get('Tr S Type', 'T3'),
                                                 int(float(params.get('Tr S Len', 100)))) * \
                             (1 + float(params.get('Tr S Off', 0.0)) / 100)

        return trades, data, result.net_profit_pct, result.max_drawdown_pct
    
    def _run_simple_backtest(self, params, date_range):
        """Упрощенная симуляция трейдов"""
        data = self.market_data.copy()

        # Применяем фильтр по датам
        if date_range:
            start, end = date_range
            data = data[start:end]
            if len(data) == 0:
                print(f"  ⚠ Warning: No data in date range {start} to {end}")
                print(f"    Available data range: {self.market_data.index.min()} to {self.market_data.index.max()}")
                return [], data, 0.0, 0.0
        
        # Извлекаем параметры
        ma_type = params.get('MA Type', 'EMA')
        ma_length = int(float(params.get('MA Length', 50)))
        close_long = int(float(params.get('Close Count Long', 3)))
        close_short = int(float(params.get('Close Count Short', 3)))
        
        # Расчет основного MA
        data['ma'] = self._calculate_ma(data, ma_type, ma_length)
        
        # Трейлинг стопы
        tr_l_type = params.get('Tr L Type', 'T3')
        tr_l_len = int(float(params.get('Tr L Len', 100)))
        tr_l_off = float(params.get('Tr L Off', 0.0))
        
        tr_s_type = params.get('Tr S Type', 'T3')
        tr_s_len = int(float(params.get('Tr S Len', 100)))
        tr_s_off = float(params.get('Tr S Off', 0.0))
        
        data['trail_long'] = self._calculate_ma(data, tr_l_type, tr_l_len) * (1 + tr_l_off / 100)
        data['trail_short'] = self._calculate_ma(data, tr_s_type, tr_s_len) * (1 + tr_s_off / 100)
        
        # Симуляция трейдов
        trades = []
        position = None
        closes_above = 0
        closes_below = 0
        
        min_bars = max(ma_length, tr_l_len, tr_s_len) + 10
        
        for i in range(min_bars, len(data)):
            price = data['close'].iloc[i]
            ma = data['ma'].iloc[i]
            timestamp = data.index[i]
            
            # Подсчет закрытий
            if price > ma:
                closes_above += 1
                closes_below = 0
            else:
                closes_below += 1
                closes_above = 0
            
            # Вход в позицию
            if position is None:
                if closes_above >= close_long:
                    # Long entry
                    stop = data['trail_long'].iloc[i]
                    position = {
                        'type': 'long',
                        'entry_time': timestamp,
                        'entry_price': price,
                        'entry_index': i,
                        'stop_price': stop,
                        'initial_stop': stop,
                        'trail_history': [(timestamp, stop)]
                    }
                elif closes_below >= close_short:
                    # Short entry
                    stop = data['trail_short'].iloc[i]
                    position = {
                        'type': 'short',
                        'entry_time': timestamp,
                        'entry_price': price,
                        'entry_index': i,
                        'stop_price': stop,
                        'initial_stop': stop,
                        'trail_history': [(timestamp, stop)]
                    }
            
            # Управление открытой позицией
            elif position:
                if position['type'] == 'long':
                    # Обновление трейлинг стопа
                    new_stop = data['trail_long'].iloc[i]
                    if new_stop > position['stop_price']:
                        position['stop_price'] = new_stop
                        position['trail_history'].append((timestamp, new_stop))
                    
                    # Проверка выхода
                    if price <= position['stop_price']:
                        position['exit_time'] = timestamp
                        position['exit_price'] = price  # Реальная цена закрытия свечи
                        position['exit_index'] = i
                        position['exit_reason'] = 'stop'
                        trades.append(position)
                        position = None
                        
                else:  # short
                    # Обновление трейлинг стопа
                    new_stop = data['trail_short'].iloc[i]
                    if new_stop < position['stop_price']:
                        position['stop_price'] = new_stop
                        position['trail_history'].append((timestamp, new_stop))
                    
                    # Проверка выхода
                    if price >= position['stop_price']:
                        position['exit_time'] = timestamp
                        position['exit_price'] = price  # Реальная цена закрытия свечи
                        position['exit_index'] = i
                        position['exit_reason'] = 'stop'
                        trades.append(position)
                        position = None
        
        # Закрываем открытую позицию
        if position:
            position['exit_time'] = data.index[-1]
            position['exit_price'] = data['close'].iloc[-1]
            position['exit_index'] = len(data) - 1
            position['exit_reason'] = 'end'
            trades.append(position)

        # Расчет Net Profit для упрощенной симуляции
        if len(trades) > 0:
            total_pnl = 0
            for t in trades:
                if t['type'] == 'long':
                    pnl = (t['exit_price'] - t['entry_price']) / t['entry_price']
                else:  # short
                    pnl = (t['entry_price'] - t['exit_price']) / t['entry_price']
                total_pnl += pnl
            net_profit_pct = total_pnl * 100
            max_dd_pct = 0.0  # Упрощенная версия не считает DD
        else:
            net_profit_pct = 0.0
            max_dd_pct = 0.0

        return trades, data, net_profit_pct, max_dd_pct
    
    def _calculate_ma(self, data, ma_type, length):
        """Расчет скользящей средней"""
        ma_type = ma_type.upper()
        
        if ma_type == 'SMA':
            return data['close'].rolling(window=length).mean()
        elif ma_type == 'EMA':
            return data['close'].ewm(span=length, adjust=False).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return data['close'].rolling(window=length).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        elif ma_type == 'DEMA':
            ema1 = data['close'].ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == 'T3':
            alpha = 0.7
            ema1 = data['close'].ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            ema4 = ema3.ewm(span=length, adjust=False).mean()
            ema5 = ema4.ewm(span=length, adjust=False).mean()
            ema6 = ema5.ewm(span=length, adjust=False).mean()
            c1 = -alpha**3
            c2 = 3*alpha**2 + 3*alpha**3
            c3 = -6*alpha**2 - 3*alpha - 3*alpha**3
            c4 = 1 + 3*alpha + alpha**3 + 3*alpha**2
            return c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3
        elif ma_type == 'VWMA':
            if 'volume' in data.columns:
                return (data['close'] * data['volume']).rolling(window=length).sum() / \
                       data['volume'].rolling(window=length).sum()
            else:
                return data['close'].rolling(window=length).mean()
        elif ma_type in ['ALMA', 'KAMA', 'TMA']:
            # Упрощенные версии
            return data['close'].ewm(span=length, adjust=False).mean()
        else:
            return data['close'].rolling(window=length).mean()
    
    def _plot_candlesticks(self, ax, data, width=None, colorup='#26a69a', colordown='#ef5350'):
        """
        Рисует японские свечи на графике

        Args:
            ax: matplotlib axis
            data: DataFrame с колонками open, high, low, close и DatetimeIndex
            width: ширина свечи в днях (если None, вычисляется автоматически)
            colorup: цвет растущей свечи (зеленый)
            colordown: цвет падающей свечи (красный)
        """
        # Конвертируем datetime в числовой формат для matplotlib
        dates = mdates.date2num(data.index)

        # Автоматически вычисляем ширину свечи на основе временного интервала
        if width is None and len(dates) > 1:
            # Берем среднее расстояние между свечами
            time_interval = np.median(np.diff(dates))
            # Ширина свечи = 40% от интервала (чтобы оставить пространство между свечами)
            width = time_interval * 0.4

        for i, (idx, row) in enumerate(data.iterrows()):
            date = dates[i]
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # Определяем цвет свечи
            color = colorup if close_price >= open_price else colordown

            # Рисуем фитиль (high-low линия) - более тонкий, как в TradingView
            ax.plot([date, date], [low_price, high_price],
                   color=color, linewidth=0.8, solid_capstyle='butt', zorder=1)

            # Рисуем тело свечи
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)

            if height == 0:  # Доджи - цена открытия = цене закрытия
                height = (high_price - low_price) * 0.01  # Тонкая линия

            rect = Rectangle((date - width/2, bottom), width, height,
                           facecolor=color, edgecolor=color,
                           linewidth=0.5, zorder=2)
            ax.add_patch(rect)

    def _format_params_text(self, params, fixed_params):
        """Форматирование параметров для отображения на графике"""
        lines = []

        # Fixed parameters
        lines.append("Fixed Parameters:")
        for key, value in fixed_params.items():
            lines.append(f"  {key}: {value}")

        lines.append("\nCombination:")
        # Ключевые параметры комбинации
        key_params = [
            'MA Type', 'MA Length',
            'Close Count Long', 'Close Count Short',
            'Stop Long X', 'Stop Long LP',
            'Stop Short X', 'Stop Short LP',
            'Stop Long Max %', 'Stop Short Max %',
            'Stop Long Max Days', 'Stop Short Max Days',
            'Tr L Type', 'Tr L Len', 'Tr L Off',
            'Tr S Type', 'Tr S Len', 'Tr S Off'
        ]

        for param in key_params:
            if param in params:
                lines.append(f"  {param}: {params[param]}")

        return '\n'.join(lines)
    
    def visualize_combination(self, combo_index, date_range=None, output_dir='./charts'):
        """Визуализация одной комбинации"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        combinations = self.optimization_results['combinations']
        fixed_params = self.optimization_results['fixed_params']
        
        if combo_index >= len(combinations):
            raise ValueError(f"Combination index {combo_index} out of range (max: {len(combinations)-1})")
        
        params = combinations.iloc[combo_index].to_dict()

        # Запуск бэктеста
        trades, data, net_profit_pct, max_dd_pct = self._run_backtest(params, date_range)

        # Вывод информации о сделках
        expected_trades = int(params.get('Trades', 0))
        print(f"  Generated {len(trades)} trades (CSV shows: {expected_trades})")

        # Проверка на пустые данные
        if len(data) == 0:
            raise ValueError("No market data available for the specified date range")

        # Создание графика
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[3, 1, 0.1], width_ratios=[3, 3, 3, 1], 
                              hspace=0.3, wspace=0.05)
        
        ax_price = fig.add_subplot(gs[0, :3])
        ax_equity = fig.add_subplot(gs[1, :3])
        ax_params = fig.add_subplot(gs[:2, 3])
        
        # График цены - японские свечи (TradingView style colors)
        self._plot_candlesticks(ax_price, data, colorup='#d1d4dc', colordown='#787b86')

        # Скользящая средняя (фиолетовый ненасыщенный цвет)
        ax_price.plot(data.index, data['ma'],
                     label=f"{params.get('MA Type', 'MA')} {int(float(params.get('MA Length', 50)))}",
                     linewidth=1.5, color='mediumpurple', alpha=0.8, zorder=3)
        ax_price.plot(data.index, data['trail_long'], 
                     label=f"Trail Long ({params.get('Tr L Type', 'T3')} {int(float(params.get('Tr L Len', 100)))})",
                     linewidth=1, color='green', alpha=0.6, linestyle='--', zorder=2)
        ax_price.plot(data.index, data['trail_short'], 
                     label=f"Trail Short ({params.get('Tr S Type', 'T3')} {int(float(params.get('Tr S Len', 100)))})",
                     linewidth=1, color='red', alpha=0.6, linestyle='--', zorder=2)
        
        # Трейды
        for trade in trades:
            if trade['type'] == 'long':
                # Entry - только треугольник вверх (уменьшенный на 30%)
                ax_price.scatter(trade['entry_time'], trade['entry_price'],
                               color='green', marker='^', s=105, zorder=5,
                               edgecolors='darkgreen', linewidths=1.5)
                # Exit - только оранжевый крестик
                ax_price.scatter(trade['exit_time'], trade['exit_price'],
                               color='orange', marker='x', s=80, zorder=6, linewidths=2)
                # Initial stop
                ax_price.hlines(trade['initial_stop'], trade['entry_time'], trade['exit_time'],
                              colors='red', linestyles='dotted', alpha=0.5, linewidth=1, zorder=3)
                # Trailing stop path
                if len(trade['trail_history']) > 1:
                    trail_times = [t[0] for t in trade['trail_history']]
                    trail_prices = [t[1] for t in trade['trail_history']]
                    ax_price.plot(trail_times, trail_prices, color='orange',
                                linewidth=2, alpha=0.8, zorder=4)
            else:  # short
                # Entry - только треугольник вниз (уменьшенный на 30%)
                ax_price.scatter(trade['entry_time'], trade['entry_price'],
                               color='red', marker='v', s=105, zorder=5,
                               edgecolors='darkred', linewidths=1.5)
                # Exit - только оранжевый крестик
                ax_price.scatter(trade['exit_time'], trade['exit_price'],
                               color='orange', marker='x', s=80, zorder=6, linewidths=2)
                # Initial stop
                ax_price.hlines(trade['initial_stop'], trade['entry_time'], trade['exit_time'],
                              colors='green', linestyles='dotted', alpha=0.5, linewidth=1, zorder=3)
                # Trailing stop path
                if len(trade['trail_history']) > 1:
                    trail_times = [t[0] for t in trade['trail_history']]
                    trail_prices = [t[1] for t in trade['trail_history']]
                    ax_price.plot(trail_times, trail_prices, color='cyan',
                                linewidth=2, alpha=0.8, zorder=4)
        
        # Заголовок - используем результаты бэктеста, а не CSV
        sharpe = params.get('Sharpe', 'N/A')

        title = f"Combo #{combo_index + 1} - {params.get('MA Type', 'N/A')} {params.get('MA Length', 'N/A')}"
        title += f" | Net Profit: {net_profit_pct:.2f}%"
        title += f" | Trades: {len(trades)}"
        title += f" | Max DD: {max_dd_pct:.2f}%"
        if sharpe != 'N/A':
            title += f" | Sharpe: {sharpe:.2f}"
        
        ax_price.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax_price.set_ylabel('Price', fontsize=11)
        ax_price.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
        ax_price.grid(True, alpha=0.3, linestyle='--')
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Equity curve
        if len(trades) > 0:
            equity = [100]
            trade_times = [data.index[0]]

            for trade in trades:
                if trade['type'] == 'long':
                    pnl_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
                else:
                    pnl_pct = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * 100

                equity.append(equity[-1] * (1 + pnl_pct / 100))
                trade_times.append(trade['exit_time'])
        else:
            # Нет трейдов - плоская линия
            equity = [100, 100]
            trade_times = [data.index[0], data.index[-1]]
        
        ax_equity.plot(trade_times, equity, linewidth=2.5, color='darkgreen', zorder=2)
        ax_equity.fill_between(trade_times, 100, equity, alpha=0.3, 
                              color='green' if equity[-1] >= 100 else 'red', zorder=1)
        ax_equity.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax_equity.set_ylabel('Equity (%)', fontsize=11)
        ax_equity.set_xlabel('Date', fontsize=11)
        ax_equity.grid(True, alpha=0.3, linestyle='--')
        ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_equity.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Параметры справа
        ax_params.axis('off')
        params_text = self._format_params_text(params, fixed_params)
        ax_params.text(0.05, 0.95, params_text, transform=ax_params.transAxes,
                      fontsize=8, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Сохранение
        filename = f"combo_{combo_index + 1}_{params.get('MA Type', 'MA')}_{int(float(params.get('MA Length', 50)))}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: {filepath.name}")
        return str(filepath)
    
    def visualize_top_n(self, n=5, date_range=None, output_dir='./charts'):
        """Визуализация топ N комбинаций"""
        combinations = self.optimization_results['combinations']
        n = min(n, len(combinations))
        
        print(f"\n{'='*60}")
        print(f"Generating charts for top {n} combinations...")
        print(f"{'='*60}\n")
        
        saved_files = []
        for i in range(n):
            print(f"[{i+1}/{n}] Processing combination #{i+1}...")
            try:
                filepath = self.visualize_combination(i, date_range, output_dir)
                saved_files.append(filepath)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\n{'='*60}")
        print(f"✓ Generated {len(saved_files)} charts in {output_dir}")
        print(f"{'='*60}\n")
        
        return saved_files


def parse_date_range(date_str):
    """Парсинг диапазона дат"""
    if not date_str:
        return None
    
    try:
        start_str, end_str = date_str.split('-')
        start = pd.Timestamp(start_str.replace('.', '-'))
        end = pd.Timestamp(end_str.replace('.', '-'))
        return (start, end)
    except Exception as e:
        print(f"Warning: Could not parse date range '{date_str}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Standalone Trade Visualization CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Базовое использование (CSV из текущей папки)
  python visualize_cli.py --csv_in market_data.csv
  
  # С указанием всех параметров
  python visualize_cli.py --csv_in data/LINKUSDT.csv --csv_out results/ --top 10 --dates 2025.05.01-2025.10.25
  
  # Топ 3 комбинации за июнь
  python visualize_cli.py --csv_in market.csv --csv_out results/ --top 3 --dates 2025.06.01-2025.06.30
        """
    )
    
    parser.add_argument('--csv_in', type=str, required=True,
                       help='Path to market data CSV file')
    
    parser.add_argument('--csv_out', type=str, default=None,
                       help='Path to optimization results CSV (file or directory). '
                            'If not specified, uses current directory')
    
    parser.add_argument('--top', type=int, default=5,
                       help='Number of top combinations to visualize (default: 5)')
    
    parser.add_argument('--dates', type=str, default=None,
                       help='Date range in format: 2025.05.01-2025.10.25')
    
    parser.add_argument('--output', type=str, default='./charts',
                       help='Output directory for charts (default: ./charts)')
    
    args = parser.parse_args()
    
    # Определяем пути
    csv_in = Path(args.csv_in)
    if not csv_in.exists():
        print(f"Error: Market data file not found: {csv_in}")
        sys.exit(1)
    
    csv_out = Path(args.csv_out) if args.csv_out else Path.cwd()
    
    # Парсим диапазон дат
    date_range = parse_date_range(args.dates)
    if args.dates and date_range:
        print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    # Пытаемся загрузить движок проекта
    script_dir = Path(__file__).parent
    importer = BacktestEngineImporter(script_dir)
    backtest_engine = importer.load_backtest_engine()
    
    if backtest_engine:
        print("✓ Loaded project backtest engine")
    else:
        print("⚠ Project backtest engine not found, using simplified simulation")
    
    # Создаем визуализатор
    try:
        visualizer = TradeVisualizer(csv_in, csv_out, backtest_engine)
    except Exception as e:
        print(f"Error initializing visualizer: {e}")
        sys.exit(1)
    
    # Генерируем графики
    try:
        visualizer.visualize_top_n(n=args.top, date_range=date_range, output_dir=args.output)
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
