import * as ort from 'onnxruntime-web';
import { CONDITION_INDEX, CROPS, OTHER_ID } from '../data/crops';

export interface Prediction {
  id: string;
  score: number;
}

export interface Match {
  id: string;
  img: string;
  similarity: number;
}

export interface Diagnosis {
  /** top conditions, best first; score = P(leaf type) x P(condition | leaf type) */
  predictions: Prediction[];
  /** leaf type (crop, or "other") from the router model, best first */
  crops: Prediction[];
  /** closest reference photos of the identified crop */
  matches: Match[];
}

export interface Classifier {
  demo: boolean;
  diagnose(img: HTMLCanvasElement): Promise<Diagnosis>;
}

interface ModelInfo {
  file: string;
  size: number;
  labels: string[];
}

interface Manifest {
  router: ModelInfo;
  specialists: Record<string, ModelInfo>;
}

interface Ref {
  id: string;
  crop: string;
  img: string;
  emb: Float32Array;
}

interface Output {
  probs: number[];
  emb: Float32Array | null;
}

/** Same thresholds as training/evaluate.py, which measured the pipeline's accuracy with them. */
const SECOND_CROP_AT = 0.2; // also ask the runner-up crop's specialist when the router gives it this much
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const base = import.meta.env.BASE_URL;

let cached: Promise<Classifier> | null = null;

export function loadClassifier(): Promise<Classifier> {
  cached ??= load();
  return cached;
}

/** The stage-1 router says which crop the leaf is; a specialist model per crop then says healthy or which disease. */
async function load(): Promise<Classifier> {
  try {
    const res = await fetch(`${base}models/manifest.json`);
    // A missing file can come back as the SPA's index.html with status 200, so check it is JSON too.
    if (!res.ok || !res.headers.get('content-type')?.includes('json')) return await loadSingle();
    const manifest: Manifest = await res.json();
    const sessions = new Map<string, Promise<ort.InferenceSession>>();
    // Specialists load on first use, so the first scan does not wait for all six models.
    const session = (info: ModelInfo) => {
      if (!sessions.has(info.file))
        sessions.set(info.file, ort.InferenceSession.create(`${base}${info.file}`, { executionProviders: ['webgpu', 'wasm'] }));
      return sessions.get(info.file)!;
    };
    const [, refs] = await Promise.all([session(manifest.router), loadRefs()]);

    const run = async (info: ModelInfo, canvas: HTMLCanvasElement): Promise<Output> => {
      const s = await session(info);
      const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', views(canvas, info.size), [VIEWS, 3, info.size, info.size]) });
      return { probs: meanSoftmax(out.logits.data as Float32Array, info.labels.length), emb: meanEmbedding(out.embedding?.data as Float32Array | undefined) };
    };

    return {
      demo: false,
      async diagnose(canvas) {
        const { router, specialists } = manifest;
        const r = (await run(router, canvas)).probs;
        const crops = router.labels
          .map((id, i) => ({ id: id === 'other' ? OTHER_ID : id, score: r[i] }))
          .sort((a, b) => b.score - a.score);
        // Even when the router says "other", ask the specialists, so the result can still show the closest crop.
        const candidates = crops.filter((c) => c.id !== OTHER_ID).slice(0, 2);
        const predictions: Prediction[] = [];
        let best: { crop: string; score: number; emb: Float32Array | null } | null = null;
        for (const [rank, c] of candidates.entries()) {
          if (rank === 1 && c.score < SECOND_CROP_AT) break;
          const info = specialists[c.id];
          const { probs, emb } = await run(info, canvas);
          info.labels.forEach((id, i) => predictions.push({ id, score: c.score * probs[i] }));
          const top = c.score * Math.max(...probs);
          if (!best || top > best.score) best = { crop: c.id, score: top, emb };
        }
        predictions.sort((a, b) => b.score - a.score);
        const matches = best?.emb ? nearest(best.emb, refs.filter((ref) => ref.crop === best.crop), 4) : [];
        return { predictions: predictions.slice(0, 3), crops, matches };
      },
    };
  } catch (err) {
    console.warn('Model not found, running in demo mode:', err);
    return demoClassifier();
  }
}

/** Older export: one model.onnx over every condition (labels.json), used until the router and specialists are trained. */
async function loadSingle(): Promise<Classifier> {
  const res = await fetch(`${base}models/labels.json`);
  if (!res.ok) throw new Error('no model manifest or labels');
  const labels: string[] = await res.json();
  const size = 224;
  const [s, refs] = await Promise.all([
    ort.InferenceSession.create(`${base}models/model.onnx`, { executionProviders: ['webgpu', 'wasm'] }),
    loadRefs(),
  ]);

  return {
    demo: false,
    async diagnose(canvas) {
      // This export takes one image at a time, so run the views one by one and stack the outputs.
      const input = views(canvas, size);
      const step = 3 * size * size;
      const logits: number[] = [];
      const embs: number[] = [];
      for (let v = 0; v < VIEWS; v++) {
        const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input.slice(v * step, (v + 1) * step), [1, 3, size, size]) });
        logits.push(...(out.logits.data as Float32Array));
        if (out.embedding) embs.push(...(out.embedding.data as Float32Array));
      }
      const probs = meanSoftmax(Float32Array.from(logits), labels.length);
      const predictions = labels.map((id, i) => ({ id, score: probs[i] })).sort((a, b) => b.score - a.score);
      const byCrop = new Map<string, number>();
      for (const p of predictions) {
        const crop = p.id === OTHER_ID ? OTHER_ID : p.id.split('___')[0];
        byCrop.set(crop, (byCrop.get(crop) ?? 0) + p.score);
      }
      const crops = [...byCrop].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);

      // Keep the best crop conditions even when "other" wins, so the result can always name a leaf and disease.
      const conditions = predictions.filter((p) => p.id !== OTHER_ID);
      const emb = meanEmbedding(embs.length ? Float32Array.from(embs) : undefined);
      const crop = conditions[0].id.split('___')[0];
      const matches = emb ? nearest(emb, refs.filter((ref) => ref.crop === crop), 4) : [];
      return { predictions: predictions.slice(0, 4), crops, matches };
    },
  };
}

async function loadRefs(): Promise<Ref[]> {
  try {
    const res = await fetch(`${base}models/refs.json`);
    if (!res.ok) return [];
    const raw: { id: string; crop?: string; img: string; emb: string }[] = await res.json();
    return raw.map((r) => {
      const bytes = Uint8Array.from(atob(r.emb), (c) => c.charCodeAt(0));
      const emb = Float32Array.from(new Int8Array(bytes.buffer));
      return { id: r.id, crop: r.crop ?? r.id.split('___')[0], img: `${base}${r.img}`, emb: normalise(emb) };
    });
  } catch {
    return [];
  }
}

function nearest(query: Float32Array, refs: Ref[], k: number): Match[] {
  const q = normalise(Float32Array.from(query));
  return refs
    .map((r) => {
      let dot = 0;
      for (let i = 0; i < q.length; i++) dot += q[i] * r.emb[i];
      return { id: r.id, img: r.img, similarity: dot };
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);
}

// Random but plausible output so the whole UI works before a model is trained.
function demoClassifier(): Classifier {
  const ids = [...CONDITION_INDEX.keys()];
  return {
    demo: true,
    async diagnose() {
      await new Promise((r) => setTimeout(r, 700));
      const crop = CROPS[Math.floor(Math.random() * CROPS.length)];
      const scores = ids.map((id) => (id.startsWith(crop.id) ? Math.random() ** 2 * 4 : Math.random() * 0.05));
      const sum = scores.reduce((a, b) => a + b, 0);
      const predictions = ids.map((id, i) => ({ id, score: scores[i] / sum })).sort((a, b) => b.score - a.score);
      const byCrop = new Map<string, number>();
      for (const p of predictions) byCrop.set(p.id.split('___')[0], (byCrop.get(p.id.split('___')[0]) ?? 0) + p.score);
      const crops = [...byCrop].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);
      return { predictions: predictions.slice(0, 3), crops, matches: [] };
    },
  };
}

function normalise(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

/**
 * The model looks at the photo three ways and averages: the centre square (as in training), the same square
 * mirrored, and the whole photo squeezed to a square, so a leaf near the edge of an uploaded photo is not cut off.
 */
const VIEWS = 3;

function views(src: HTMLCanvasElement, size: number): Float32Array {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  const side = Math.min(src.width, src.height);
  const sx = (src.width - side) / 2;
  const sy = (src.height - side) / 2;
  const plane = size * size;
  const out = new Float32Array(VIEWS * 3 * plane);
  const draws = [
    () => ctx.drawImage(src, sx, sy, side, side, 0, 0, size, size),
    () => {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(src, sx, sy, side, side, -size, 0, size, size);
      ctx.restore();
    },
    () => ctx.drawImage(src, 0, 0, src.width, src.height, 0, 0, size, size),
  ];
  draws.forEach((draw, v) => {
    draw();
    const { data } = ctx.getImageData(0, 0, size, size);
    for (let i = 0; i < plane; i++) {
      for (let ch = 0; ch < 3; ch++) {
        out[(v * 3 + ch) * plane + i] = (data[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
      }
    }
  });
  return out;
}

/** Softmax of each view's logits, averaged over the views. */
function meanSoftmax(logits: Float32Array, n: number): number[] {
  const mean = new Array(n).fill(0);
  for (let v = 0; v < logits.length / n; v++) {
    const row = Array.from(logits.subarray(v * n, (v + 1) * n));
    const max = Math.max(...row);
    const exps = row.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    exps.forEach((e, i) => (mean[i] += e / sum / VIEWS));
  }
  return mean;
}

function meanEmbedding(emb: Float32Array | undefined): Float32Array | null {
  if (!emb) return null;
  const d = emb.length / VIEWS;
  const mean = new Float32Array(d);
  for (let v = 0; v < VIEWS; v++) for (let i = 0; i < d; i++) mean[i] += emb[v * d + i];
  return normalise(mean);
}
