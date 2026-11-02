The trained models live here (written by `training/export.py`):

- `manifest.json` – file, input size and labels of the router and of each crop specialist
- `router.onnx` – which crop (or "other"); input `[N,3,224,224]` float32, ImageNet-normalised RGB
- `<crop>.onnx` – healthy or which disease for that crop; input `[N,3,288,288]`
- every model outputs `logits` and an L2-normalised `embedding`
- `refs.json` – reference photos per condition with int8 embeddings; `metrics.json` – validation scores

Class ids must match the `id`s in `src/data/crops.ts`. Without `manifest.json` the app falls back to a single
`model.onnx` + `labels.json`, and with neither it runs in DEMO mode.
