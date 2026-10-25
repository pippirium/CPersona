"""Isolated retrieval PoC. Never opens a CPersona memory database or returns images."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODELS = {
    'gemma': 'google/embeddinggemma-2',
    'jina': 'jinaai/jina-embeddings-v5-text-nano',
}
REVISIONS = {
    'gemma': '914f7f89142e33e77833254d9c9b90c3cef7303b',
    'jina': '8a7f00aac812071b69403df470f1038ec85f8925',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['gemma', 'jina', 'both'], default='both')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'sample.json')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--image', type=Path, help='Explicitly opt in to embedding one image')
    parser.add_argument('--image-query', default='赤い警告が表示された設定画面')
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text())
    ids = [r['id'] for r in data['records']]
    if len(ids) != len(set(ids)) or not data['queries']:
        raise ValueError('Record ids must be unique and queries nonempty')
    for q in data['queries']:
        if not q['relevant'] or not set(q['relevant']) <= set(ids):
            raise ValueError('Every query needs existing relevant record ids')
    output = {'dataset': str(args.dataset.resolve()), 'device': args.device,
              'scope': 'dense retrieval only; synthetic sample is a smoke test, not a quality verdict',
              'models': {}}
    for name in (MODELS if args.model == 'both' else [args.model]):
        start = time.perf_counter()
        kwargs = {'device': args.device, 'model_kwargs': {'torch_dtype': torch.float32}}
        if name == 'gemma':
            kwargs['config_kwargs'] = {'audio_config': None, **({'vision_config': None} if not args.image else {})}
        else:
            # Required by this publisher's SentenceTransformers implementation.
            kwargs['trust_remote_code'] = True
        print(f'Loading {MODELS[name]}', flush=True)
        model = SentenceTransformer(MODELS[name], revision=REVISIONS[name], **kwargs)
        load_s = time.perf_counter() - start
        common = {'normalize_embeddings': True, 'convert_to_numpy': True,
                  'show_progress_bar': False, 'batch_size': 4}
        start = time.perf_counter()
        docs = model.encode([r['text'] for r in data['records']],
                            **({'prompt_name': 'Document'} if name == 'gemma' else {'task': 'retrieval', 'prompt_name': 'document'}), **common)
        index_s = time.perf_counter() - start
        rows = []
        for q in data['queries']:
            start = time.perf_counter()
            vec = model.encode([q['text']],
                               **({'prompt_name': 'SearchQuery'} if name == 'gemma' else {'task': 'retrieval', 'prompt_name': 'query'}), **common)[0]
            scores = docs @ vec
            order = np.argsort(-scores, kind='stable')
            relevant = set(q['relevant'])
            hits = [i + 1 for i, pos in enumerate(order) if ids[pos] in relevant]
            rows.append({'query': q['text'], 'recall_at_1': sum(h <= 1 for h in hits) / len(relevant),
                         'recall_at_5': sum(h <= 5 for h in hits) / len(relevant), 'rr': 1 / hits[0],
                         'query_ms': (time.perf_counter() - start) * 1000,
                         'top5': [{'id': ids[p], 'score': float(scores[p])} for p in order[:5]]})
        if not np.isfinite(docs).all():
            raise ValueError('Nonfinite embeddings')
        result = {'model': MODELS[name], 'revision': REVISIONS[name], 'dimension': docs.shape[1], 'load_s': load_s, 'index_s': index_s,
                  'recall_at_1': float(np.mean([r['recall_at_1'] for r in rows])),
                  'recall_at_5': float(np.mean([r['recall_at_5'] for r in rows])),
                  'mrr': float(np.mean([r['rr'] for r in rows])),
                  'median_query_ms': float(np.median([r['query_ms'] for r in rows])), 'queries': rows}
        if name == 'gemma' and args.image:
            image_vec = model.encode({'image': str(args.image.resolve())}, **common)
            query_vec = model.encode(args.image_query, prompt_name='SearchQuery', **common)
            if not np.isfinite(image_vec).all():
                raise ValueError('Nonfinite image embedding')
            result['image_smoke'] = {'shape': list(image_vec.shape),
                                     'similarity': float(np.asarray(image_vec @ query_vec.T).squeeze()),
                                     'note': 'One image tests execution, not retrieval quality. No image output.'}
        output['models'][name] = result
        print(json.dumps({k: v for k, v in result.items() if k not in ['queries']}, ensure_ascii=False), flush=True)
        del model
    target = ROOT / 'results'
    target.mkdir(exist_ok=True)
    path = target / f'{time.time_ns()}.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f'Results: {path}')


if __name__ == '__main__':
    main()
