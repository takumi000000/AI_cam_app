# AI Camera Demo (MediaPipe Face Mesh + Hands)

## Dockerで環境構築（ローカル実行）

### 1. ビルド

```bash
docker compose build
```

### 2. 起動（Webカメラを使用）

```bash
docker compose up
```

### 3. アクセス

ブラウザで以下にアクセスしてください。

```
http://localhost:8000/
```

> **メモ**
> - DockerコンテナからWebカメラを使うために `/dev/video0` をマウントしています。
> - macOS/Windowsの場合はDockerのデバイス共有設定が必要になるため、必要に応じて `docker-compose.yml` を調整してください。
> - 画面にはMJPEGストリームが表示されます。

## Colab向けコード（参考）

`colab_demo.md` にインストールセル・JavaScriptセル・Pythonセルの全コードをまとめています。
