import argparse
import pandas as pd


def weighted_mrr_similarity(a, b, weights):
    weighted_dot = sum(w * ai * bi for w, ai, bi in zip(weights, a, b))
    norm_a_sq    = sum(w * ai * ai for w, ai in zip(weights, a))
    norm_b_sq    = sum(w * bi * bi for w, bi in zip(weights, b))
    if norm_a_sq == 0.0 and norm_b_sq == 0.0:
        return 0.0
    return weighted_dot / max(norm_a_sq, norm_b_sq)


def compute_similarity(csv_file, reference, output_file, lambda_decay=0.9):
    df = pd.read_csv(csv_file)

    pool_sizes = sorted(df['pool_size'].unique())
    d = len(pool_sizes)

    span_weights = []
    for i in range(d):
        if i == 0:
            span = pool_sizes[1] - pool_sizes[0]
        elif i == d - 1:
            span = pool_sizes[i] - pool_sizes[i - 1]
        else:
            span = (pool_sizes[i + 1] - pool_sizes[i - 1]) / 2.0
        span_weights.append(span)

    importance = [lambda_decay ** i for i in range(d)]
    weights = [s * w for s, w in zip(span_weights, importance)]

    data = {
        dataset: df[df['dataset'] == dataset].sort_values('pool_size')['MRR'].tolist()
        for dataset in df['dataset'].unique()
    }

    if reference not in data:
        raise ValueError(f"Reference dataset '{reference}' not found in CSV.")

    ref_mrr = data[reference]
    results = []
    for dataset, mrr in data.items():
        sim = weighted_mrr_similarity(ref_mrr, mrr, weights)
        offset_pct = -(abs(sim - 1.0)) * 100
        results.append({'dataset': dataset, 'similarity': sim, 'offset_pct': offset_pct})

    results.sort(key=lambda x: x['similarity'], reverse=True)

    out = pd.DataFrame(results)
    out.to_csv(output_file, index=False)

    print(f"\n{'Dataset':<35} {'Sim.':<8} {'Offset'}")
    print("=" * 55)
    for row in results:
        marker = " *" if row['dataset'] == reference else ""
        print(f"{row['dataset']:<35} {row['similarity']:.3f}    {row['offset_pct']:+.2f}%{marker}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--reference', required=True, help='Dataset name to use as reference (e.g. "Ground Truth")')
    parser.add_argument('--output', required=True)
    parser.add_argument('--lambda_decay', type=float, default=0.9)
    args = parser.parse_args()

    compute_similarity(args.csv, args.reference, args.output, args.lambda_decay)
