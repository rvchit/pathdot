# PathVQA comparison

Same 1000 seeded test questions for every run; paired exact McNemar test on overall exact match.

| run | yes/no acc | open exact | open recall | overall exact | p50 ms | $/1k questions |
|---|---|---|---|---|---|---|
| qwen3-vl-8b_zeroshot | 67.1% (63.1-71.3) | 1.2% (0.4-2.2) | 5.8% (4.2-7.6) | 34.1% (31.2-37.1) | 305 | $0.40 |
| qwen3-vl-8b_8shot | 62.1% (57.7-66.3) | 1.8% (0.8-3.0) | 5.8% (4.0-7.7) | 31.9% (29.0-34.8) | 772 | $0.47 |
| qwen3p5-9b_zeroshot | 75.2% (71.1-79.0) | 5.2% (3.4-7.2) | 7.6% (5.6-9.8) | 40.1% (37.2-43.1) | 312 | $0.65 |
| qwen3p5-9b_ft_e1 | 90.6% (88.0-93.0) | 27.5% (23.5-31.3) | 32.5% (28.5-36.4) | 59.0% (56.1-62.2) | 473 | $0.76 |

| A | B | B - A (pts) | n01 (A right, B wrong) | n10 | p |
|---|---|---|---|---|---|
| qwen3-vl-8b_zeroshot | qwen3-vl-8b_8shot | -2.2 | 57 | 35 | 0.02802 |
| qwen3-vl-8b_zeroshot | qwen3p5-9b_zeroshot | +6.0 | 57 | 117 | 6.323e-06 |
| qwen3-vl-8b_zeroshot | qwen3p5-9b_ft_e1 | +24.9 | 20 | 269 | 7.521e-57 |
| qwen3-vl-8b_8shot | qwen3p5-9b_zeroshot | +8.2 | 61 | 143 | 9.041e-09 |
| qwen3-vl-8b_8shot | qwen3p5-9b_ft_e1 | +27.1 | 18 | 289 | 4.508e-64 |
| qwen3p5-9b_zeroshot | qwen3p5-9b_ft_e1 | +18.9 | 24 | 213 | 4.855e-39 |
