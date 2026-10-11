# CoinSeek Model

Public Python training code for CoinSeek. The training workflow downloads public Binance market-data archives, trains a LightGBM return model, validates it on a later time window, exports ONNX, checks ONNX Runtime parity, and publishes the model files as a GitHub Release in this repository.

## Local training

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m coinseek_model.train \
  --model-id local-example \
  --market usdm \
  --interval 1h \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start 2023-01-01T00:00:00Z \
  --end 2025-01-01T00:00:00Z
```

The model predicts the next holding-period return from three versioned features: momentum, negative realized volatility, and quote-volume trend. Fees and slippage are deducted from training labels; USD-M futures labels also account for funding rates during the holding period. Validation uses a strictly later time window and reports long-short returns, annualized Sharpe, drawdown, and ONNX parity error.

## GitHub Workflow

Run **Train and release ONNX model** with `workflow_dispatch`. Choose the market, interval, symbols, training dates, lookback, and prediction horizon. The workflow downloads public monthly and daily archives from Binance Vision and never needs Binance credentials or direct access to the Binance REST API. USD-M funding-rate archives are monthly, so set `end` to midnight UTC on the first day of a month (for example, `2026-10-01T00:00:00Z`) to train through the prior month. After validation succeeds, it creates a versioned GitHub Release containing `model.onnx` and `manifest.json`; the workflow uses its repository-scoped `GITHUB_TOKEN` with `contents: write` permission.

Release assets are public. The generated model ID includes the workflow run and attempt IDs so each run has a unique release tag.

## Model contract

The ONNX input is a `float32` tensor named `features`, shaped `[asset_count, 3]`, in the feature order declared in `manifest.json`. The output is one predicted return score per asset. The model artifact is immutable and carries its feature schema version, training window, code commit, runtime versions, validation metrics, and SHA-256 checksum.

CoinSeek validates the manifest and checksum during manual import, then loads and smoke-tests the ONNX model with its Go runtime. Strategy execution must pin an approved model version; training only publishes files and never changes a live strategy's model automatically.

## CoinSeek server configuration

Download `model.onnx` and `manifest.json` from the desired Release, then import both files in CoinSeek's Factor Research page. The authenticated user owns the imported candidate. CoinSeek stores the artifact in its configured file storage, so no GitHub token or S3 publishing secrets are needed by the training workflow. Install a compatible ONNX Runtime shared library on the Go server and set its absolute path with `ONNX_RUNTIME_LIBRARY`.

## License

MIT. Binance market data remains subject to Binance's terms.
