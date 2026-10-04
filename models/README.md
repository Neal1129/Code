# 模型说明

ESM-2 打分使用官方预训练模型 `esm2_t33_650M_UR50D`。

- 权重不随本代码包分发。
- 首次运行 `python predict.py` 时，`fair-esm` 会自动下载并缓存模型权重。
- 需要已安装 `torch` 与 `fair-esm`，并在首次下载时保持网络连接。
