/**
 * Real, frozen CLIP inference. Region colors are cosine-similarity-derived
 * weights, not internal attention, segmentation, or calibrated probabilities.
 * No robot policy or learned action decoder is implemented by this module.
 */
import {
  env, AutoTokenizer, AutoProcessor, RawImage,
  CLIPTextModelWithProjection, CLIPVisionModelWithProjection,
} from './vendor/transformers.min.js';

export const MODEL = Object.freeze({
  id: 'Xenova/clip-vit-base-patch32',
  base: 'openai/clip-vit-base-patch32',
  revision: 'd15189d7028b43f1d3e65039190477f6af591c2a',
  runtime: 'Transformers.js 3.8.1 / ONNX Runtime Web / WASM',
  dtype: 'q8', embeddingDimension: 512,
  method: 'Independent grid-crop CLIP cosine similarity',
});

env.allowLocalModels = true;
env.localModelPath = new URL('./models/', import.meta.url).href;
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
// One thread works without cross-origin isolation on ordinary static hosting.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

let state = null;
let pendingLoad = null;
let lastImage = null;
const textCache = new Map();
let generation = 0;

const now = () => performance.now();
const report = (callback, info) => { if (typeof callback === 'function') callback(info); };

export function cacheInfo() {
  return { loaded: !!state, imageCached: !!lastImage,
    cachedGridSize: lastImage?.gridSize ?? null, textEntries: textCache.size };
}

export function clearEmbeddingCache() {
  lastImage = null;
  textCache.clear();
  generation++;
}

export async function loadModel(onProgress) {
  if (state) return { ...MODEL, alreadyLoaded: true, loadMs: 0 };
  if (!pendingLoad) {
    pendingLoad = (async () => {
      const started = now();
      const progress_callback = event => report(onProgress, { phase: 'model', ...event });
      const common = { revision: MODEL.revision, progress_callback };
      report(onProgress, { phase: 'model', status: 'loading', message: 'Loading frozen CLIP encoders. First use downloads about 154 MB of model weights.' });
      // Load sequentially to avoid competing browser memory peaks.
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL.id, common);
      const processor = await AutoProcessor.from_pretrained(MODEL.id, common);
      const textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL.id,
        { ...common, dtype: 'q8', device: 'wasm' });
      const visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL.id,
        { ...common, dtype: 'q8', device: 'wasm' });
      state = { tokenizer, processor, textModel, visionModel, loadMs: now() - started };
      report(onProgress, { phase: 'model', status: 'ready', message: 'Real CLIP encoders ready.' });
      return { ...MODEL, alreadyLoaded: false, loadMs: state.loadMs };
    })().catch(error => { pendingLoad = null; throw error; });
  }
  return pendingLoad;
}

function normalized(vector) {
  let ss = 0;
  for (const x of vector) ss += x * x;
  const divisor = Math.sqrt(ss);
  if (!Number.isFinite(divisor) || divisor === 0) throw new Error('Invalid model embedding.');
  return Array.from(vector, x => x / divisor);
}

async function loadImage(source) {
  const image = new Image();
  image.crossOrigin = 'anonymous';
  image.src = source;
  await image.decode();
  if (!image.naturalWidth || !image.naturalHeight) throw new Error('Image has no pixels.');
  return image;
}

export async function encodeImageRegions(imageDataUrl, gridSize = 3, onProgress) {
  if (![2, 3, 4].includes(Number(gridSize))) throw new Error('Grid size must be 2, 3, or 4.');
  gridSize = Number(gridSize);
  if (lastImage?.source === imageDataUrl && lastImage.gridSize === gridSize) {
    return { ...lastImage, cacheHit: true, imageMs: 0 };
  }
  await loadModel(onProgress);
  const started = now();
  const currentGeneration = generation;
  const image = await loadImage(imageDataUrl);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const embeddings = [];
  const boxes = [];
  // Each independent crop is stretched into a square before CLIP processing so
  // that rectangular uploads do not silently discard crop edges at center-crop.
  canvas.width = 224;
  canvas.height = 224;
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const x0 = Math.floor(col * image.naturalWidth / gridSize);
      const y0 = Math.floor(row * image.naturalHeight / gridSize);
      const x1 = Math.floor((col + 1) * image.naturalWidth / gridSize);
      const y1 = Math.floor((row + 1) * image.naturalHeight / gridSize);
      context.clearRect(0, 0, 224, 224);
      context.drawImage(image, x0, y0, x1 - x0, y1 - y0, 0, 0, 224, 224);
      const data = context.getImageData(0, 0, 224, 224);
      const raw = new RawImage(new Uint8ClampedArray(data.data), 224, 224, 4);
      const inputs = await state.processor(raw);
      const output = await state.visionModel(inputs);
      embeddings.push(normalized(output.image_embeds.data));
      boxes.push([x0 / image.naturalWidth, y0 / image.naturalHeight,
        (x1 - x0) / image.naturalWidth, (y1 - y0) / image.naturalHeight]);
      for (const tensor of Object.values(inputs)) tensor?.dispose?.();
      for (const tensor of Object.values(output)) tensor?.dispose?.();
      report(onProgress, { phase: 'vision', completed: embeddings.length,
        total: gridSize * gridSize, message: `Encoded crop ${embeddings.length}/${gridSize * gridSize}` });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  const result = { source: imageDataUrl, gridSize, embeddings, boxes,
    width: image.naturalWidth, height: image.naturalHeight,
    cacheHit: false, imageMs: now() - started };
  if (generation === currentGeneration) lastImage = result;
  return result;
}

export async function encodeText(command, onProgress) {
  command = String(command).trim();
  if (!command) throw new Error('Enter a target description first.');
  if (command.length > 400) throw new Error('Use a short target description (400 characters maximum).');
  if (textCache.has(command)) return { ...textCache.get(command), cacheHit: true, textMs: 0 };
  await loadModel(onProgress);
  const started = now();
  const inputs = state.tokenizer(command, { padding: true, truncation: true, max_length: 77 });
  const tokenIds = Array.from(inputs.input_ids.data, Number);
  const tokens = tokenIds.map(id => state.tokenizer.decode([id], { skip_special_tokens: false }));
  const output = await state.textModel(inputs);
  const result = { vector: normalized(output.text_embeds.data), tokenIds, tokens,
    tokenCount: tokenIds.length, cacheHit: false, textMs: now() - started };
  for (const tensor of Object.values(inputs)) tensor?.dispose?.();
  for (const tensor of Object.values(output)) tensor?.dispose?.();
  if (textCache.size >= 32) textCache.delete(textCache.keys().next().value);
  textCache.set(command, result);
  return result;
}

export function scoreRegions(embeddings, textVector, temperature = 0.07) {
  temperature = Number(temperature);
  if (!Number.isFinite(temperature) || temperature <= 0) throw new Error('Temperature must be positive.');
  if (!embeddings.length || !textVector.length) throw new Error('Missing embeddings.');
  const scores = embeddings.map(vector => {
    if (vector.length !== textVector.length) throw new Error('Image and text embedding widths differ.');
    return vector.reduce((total, x, j) => total + x * textVector[j], 0);
  });
  const max = Math.max(...scores);
  const exp = scores.map(score => Math.exp((score - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return { scores, weights: exp.map(x => x / sum) };
}

export async function analyze({ imageDataUrl, prompt, gridSize = 3, temperature = 0.07, onProgress }) {
  const started = now();
  const modelLoad = await loadModel(onProgress);
  const images = await encodeImageRegions(imageDataUrl, gridSize, onProgress);
  const text = await encodeText(prompt, onProgress);
  const scoreStarted = now();
  const { scores, weights } = scoreRegions(images.embeddings, text.vector, temperature);
  const scoreMs = now() - scoreStarted;
  const crops = images.boxes.map((box, index) => ({ index,
    row: Math.floor(index / gridSize), col: index % gridSize,
    box, score: scores[index], weight: weights[index] }));
  return {
    crops, textTokens: text.tokens, tokenIds: text.tokenIds,
    embeddings: images.embeddings, textEmbedding: text.vector,
    imageSize: [images.width, images.height], prompt: String(prompt).trim(),
    gridSize: Number(gridSize), temperature: Number(temperature),
    shapes: { imageInput: [1, 3, 224, 224], cropEmbeddings: [crops.length, 512],
      textInput: [1, text.tokenCount], textEmbedding: [1, 512],
      cosineScores: [crops.length], weights: [crops.length] },
    timings: { loadMs: modelLoad.loadMs, imageMs: images.imageMs, textMs: text.textMs,
      scoreMs, totalMs: now() - started },
    cache: { imageHit: images.cacheHit, textHit: text.cacheHit },
    model: MODEL,
  };
}
