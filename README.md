# CoinSeek Model

Public Python training code for CoinSeek. The training workflow reads public Binance candles, trains a LightGBM return model, validates it on a later time window, exports ONNX, checks ONNX Runtime parity, and publishes candidate files to a private S3-compatible store.

## Local training

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m coinseek_model.train \
  --job-id local-example \
  --market usdm \
  --interval 1h \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start 2023-01-01T00:00:00Z \
  --end 2025-01-01T00:00:00Z
```

The model predicts the next holding-period return from three versioned features: momentum, negative realized volatility, and quote-volume trend. Fees and slippage are deducted from training labels; USD-M futures labels also account for funding rates during the holding period. Validation uses a strictly later time window and reports long-short returns, annualized Sharpe, drawdown, and ONNX parity error.

## GitHub Workflow

Run **Train and publish ONNX model** with `workflow_dispatch`. The training job uses public market endpoints and never needs Binance credentials. Add a protected GitHub Environment named `model-publish` and configure these secrets:

- `MODEL_S3_ENDPOINT`
- `MODEL_S3_ACCESS_KEY`
- `MODEL_S3_SECRET_KEY`
- `MODEL_S3_BUCKET`

Optionally set `MODEL_S3_PREFIX` as an environment variable. Restrict environment deployment to the protected default branch and require review for publishing changes. Candidate files are written to `<prefix>/<job_id>/model.onnx` and `manifest.json`; they are not uploaded as public GitHub artifacts.

## Model contract

The ONNX input is a `float32` tensor named `features`, shaped `[asset_count, 3]`, in the feature order declared in `manifest.json`. The output is one predicted return score per asset. The model artifact is immutable and carries its feature schema version, training window, code commit, runtime versions, validation metrics, and SHA-256 checksum.

CoinSeek should validate the manifest and checksum before loading a model. Strategy execution must pin an approved model version; training only creates candidates and never changes a live strategy's model automatically.

## CoinSeek server configuration

Configure the backend with a GitHub token that can dispatch and read workflow runs in this repository (`MODEL_WORKFLOW_TOKEN`). Set `MODEL_ARTIFACT_BUCKET` and `MODEL_ARTIFACT_PREFIX` to the same bucket and prefix configured in the `model-publish` environment. The server's configured S3/MinIO credentials must be able to read those objects. Install a compatible ONNX Runtime shared library on the Go server and set its absolute path with `ONNX_RUNTIME_LIBRARY`; a candidate is not approvable until the Go runtime loads it and successfully runs a validation input.

## License

MIT. Binance market data remains subject to Binance's terms.
