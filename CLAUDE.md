# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **standalone Python CLI tool** for visualizing cryptocurrency trading strategy results. It generates professional multi-panel charts showing price action, moving averages, trailing stops, trade entries/exits, and equity curves from backtesting optimization results.

**Key characteristics:**
- Pure Python (no Node.js, no build tooling)
- Self-contained with automatic project integration via dynamic module discovery
- Designed to work standalone OR integrate with an existing trading strategy project

## Commands

### Running the Visualization Tool

```bash
# Basic usage (required: market data CSV)
python visualize_cli.py --csv_in market_data.csv

# With optimization results directory
python visualize_cli.py --csv_in data/LINKUSDT_15m.csv --csv_out results/

# Visualize top N combinations
python visualize_cli.py --csv_in market.csv --csv_out results/ --top 10

# Filter by date range (format: YYYY.MM.DD-YYYY.MM.DD)
python visualize_cli.py --csv_in market.csv --dates 2025.05.01-2025.10.25

# Full example
python visualize_cli.py \
    --csv_in data/market.csv \
    --csv_out optimization_results/ \
    --top 15 \
    --dates 2025.05.01-2025.09.30 \
    --output my_charts/
```

### Installing Dependencies

```bash
pip install pandas numpy matplotlib backtesting
```

The `backtesting` library is only used for its drawdown calculation utility function.

## Architecture

### Two-Module Design

The codebase consists of two main Python modules:

1. **visualize_cli.py** (652 lines) - Main entry point and visualization engine
   - CLI argument parsing
   - CSV data loading (market data + optimization results)
   - Chart generation (matplotlib-based multi-panel layouts)
   - Dynamic discovery and integration with `backtest_engine.py`
   - Built-in fallback simulation if project engine not found

2. **backtest_engine.py** (661 lines) - Optional strategy backtesting engine
   - Comprehensive dataclass models (`StrategyParams`, `TradeRecord`, `StrategyResult`)
   - 11 moving average implementations (EMA, SMA, WMA, HMA, VWMA, ALMA, DEMA, KAMA, TMA, T3, VWAP)
   - Full strategy backtesting with position management, stop-losses, and trailing stops
   - Technical indicators (ATR, various MAs)

### Dynamic Module Discovery

The `BacktestEngineImporter` class (visualize_cli.py:21-63) automatically discovers `backtest_engine.py`:

- Searches up to 5 directory levels from script location
- Checks common project structures: root, `src/`, `engine/`
- Uses Python's `importlib.util` for dynamic importing
- **Gracefully degrades** to built-in simulation if not found

This design allows the tool to function completely standalone while automatically leveraging a full project engine when available.

### Data Flow

```
Input CSVs → Data Loading → Parameter Parsing → Backtesting → Indicator Calculation → Chart Generation → PNG Output
```

**Key processing steps:**
1. Market data CSV loaded and normalized (flexible column naming)
2. Optimization results CSV parsed (fixed parameters + combinations table)
3. For each top-N combination:
   - Extract strategy parameters
   - Run backtest (via project engine or fallback)
   - Calculate all indicators (MAs, ATR)
   - Generate 3-panel chart (price + equity + parameters)
   - Save as PNG (~350KB, 20×12", 150 DPI)

## CSV Format Requirements

### Market Data CSV (--csv_in)

**Required columns:** `open`, `high`, `low`, `close` (case-insensitive)

**Timestamp column:** First column OR named `timestamp`/`date`/`datetime`

**Optional:** `volume` (used for VWMA calculations)

Example:
```csv
timestamp,open,high,low,close,volume
2025-05-01 00:00:00,10.5,10.8,10.4,10.6,15000
2025-05-01 00:15:00,10.6,10.9,10.5,10.7,12000
```

### Optimization Results CSV (--csv_out)

**Special multi-section format:**
1. Optimization Metadata (ignored)
2. Fixed Parameters section (header: "Fixed Parameters")
3. Combinations table with columns:
   - MA Type, MA Length
   - Close Count Long, Close Count Short
   - Stop Long X, Stop Short X, Stop Long LP, Stop Short LP
   - Trailing Type, Trailing Len, Trailing Off
   - Performance metrics (Net Profit %, Max DD %, etc.)

The parser in `TradeVisualizer._load_optimization_results()` (visualize_cli.py:144-224) handles this format automatically.

## Working with Indicators

### Supported Moving Average Types

The code supports 9+ MA types (visualize_cli.py:361-408):
- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **WMA** - Weighted Moving Average
- **DEMA** - Double Exponential Moving Average
- **T3** - Triple Exponential Moving Average
- **VWMA** - Volume-Weighted Moving Average
- **ALMA** - Arnaud Legoux Moving Average
- **KAMA** - Kaufman Adaptive Moving Average
- **TMA** - Triangular Moving Average

### Adding New Indicators

To add a new indicator type:

1. Add calculation in `TradeVisualizer._calculate_ma()` (visualize_cli.py:361):
   ```python
   elif ma_type == 'YOUR_MA':
       return your_calculation(data, length)
   ```

2. If needed, also add to `backtest_engine.py` `get_ma()` function for consistency

## Chart Customization

### Chart Specifications

**Default size:** 20×12 inches (figsize at visualize_cli.py:412)
**DPI:** 150 (savefig call at visualize_cli.py:525)
**Layout:** 3 panels (1×3 grid with custom width ratios)

### Key Customization Points

- **Panel layout:** visualize_cli.py:412-420 (GridSpec configuration)
- **Chart size/DPI:** visualize_cli.py:412, 525
- **Font sizes:** visualize_cli.py:518 (parameters panel)
- **Colors:** visualize_cli.py:430-504 (price chart, equity, markers)

## Important Implementation Details

### Backtesting Integration Pattern

The tool uses a **graceful degradation pattern**:

```python
# visualize_cli.py:226-280
def _run_backtest(self, ...):
    if self.be:
        # Use full project engine
        return self._run_project_backtest(...)
    else:
        # Use simplified built-in simulation
        return self._run_simple_backtest(...)
```

This allows the tool to work in any environment while providing full accuracy when integrated with a project.

### Date Filtering

Date filtering (--dates flag) is applied AFTER data loading in `visualize_combination()` (visualize_cli.py:578-584). The format is strict: `YYYY.MM.DD-YYYY.MM.DD`

### File Output Naming

Charts are saved as: `combo_{index}_{MA_Type}_{MA_Length}.png`

The naming logic is in visualize_cli.py:651 and includes combo number, MA type, and MA length for easy identification.

## Project-Specific Conventions

### Russian Documentation

The CLI_README.md and ФИНАЛЬНЫЙ_README.md are in Russian. This is intentional as the project appears to be for a Russian-speaking audience.

### Trading Strategy Context

This tool is specifically designed for the "S_01_v26-TrailingMA-Ultralight" trading strategy (mentioned in CLI_README.md:3), which uses:
- Moving average crossovers for entries
- Trailing stop-losses (long and short)
- Close count thresholds (number of candles before exit signals)
- Stop-loss risk-reward ratios

### OKX Exchange Data

The example CSV files are from OKX exchange (LINKUSDT perpetual contracts, 15-minute timeframe). The tool works with any OHLCV data but was designed and tested with crypto perpetual futures data.

## Performance Characteristics

**Chart generation speed:** ~3-5 seconds per combination
**Output size:** ~350KB per PNG
**Memory usage:** Depends on CSV size (uses pandas vectorized operations)

For batch processing, the tool can handle 50+ charts efficiently. The bottleneck is matplotlib rendering, not backtesting calculations.

## No Testing Framework

Currently, there is no formal test suite. Testing is done manually using the included example CSV files:
- `OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv` (1MB market data)
- `OKX_LINKUSDT.P, 15 2025.05.01-2025.10.25_EMA+SMA+...csv` (26KB optimization results)

When making changes, verify functionality by running the CLI tool with these files.
