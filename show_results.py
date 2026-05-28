"""Display compiled experiment results."""
import pandas as pd
from pathlib import Path

res_dir = Path(__file__).parent / "results"

for d in sorted([d for d in res_dir.iterdir() if d.is_dir()]):
    csv = d / 'benchmark_results.csv'
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    name = d.name
    best = df.loc[df['R2'].idxmax()]
    print(f'\n{"="*70}')
    print(f'{name} (N={int(df.iloc[0]["N_Edges"]**0.5+1) if "N_Edges" in df.columns else "?"})')
    print(f'Best: {best["Model"]}, R2={best["R2"]:.4f}, RMSE={best["RMSE"]:.4f}, MAE={best["MAE"]:.4f}')
    print(f'{"Model":<25s} {"R2":>8s} {"RMSE":>8s} {"MAE":>8s}')
    print('-' * 52)
    for _, row in df.sort_values('R2', ascending=False).iterrows():
        marker = '  <<< BEST' if row['R2'] == best['R2'] else ''
        print(f'{row["Model"]:<25s} {row["R2"]:>+8.4f} {row["RMSE"]:>8.4f} {row["MAE"]:>8.4f}{marker}')
