# AI Camera Demo (MediaPipe Face Mesh + Hands)

## Dockerで環境構築

### 1. ビルド

```bash
docker compose build
```

### 2. 起動

```bash
docker compose up
```

### 3. アクセス

ブラウザで以下にアクセスしてください。

```
http://localhost:8888/?token=demo
```

> **メモ**
> - コンテナ内のJupyterLabから `colab_demo.md` を開き、セルを順に実行してください。
> - ローカル環境でWebカメラを使う場合は、ブラウザでHTTPSまたはlocalhost経由のアクセスが必要です。

## Colab向けコード

`colab_demo.md` にインストールセル・JavaScriptセル・Pythonセルの全コードをまとめています。
