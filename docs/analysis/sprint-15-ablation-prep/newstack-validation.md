# New-stack validation — 2026-08-25

## A. Binary A/B (old weights, pp2048/tg128, fa on)

### homebrew
```
| gemma4 E2B Q8_0                |   4.61 GiB |     4.65 B | BLAS,Vulkan |       6 |   1 |          pp2048 |      3926.96 ± 39.02 |
| gemma4 E2B Q8_0                |   4.61 GiB |     4.65 B | BLAS,Vulkan |       6 |   1 |           tg128 |         78.15 ± 4.43 |
```
### coopmat2
```
| gemma4 E2B Q8_0                |   4.61 GiB |     4.65 B | Vulkan     |  -1 |   1 |          pp2048 |      4094.97 ± 25.29 |
| gemma4 E2B Q8_0                |   4.61 GiB |     4.65 B | Vulkan     |  -1 |   1 |           tg128 |         78.19 ± 3.35 |
```

## B. Weights
old(Apr-21) sha256[:16] = e049411c01fb7a81
refreshed   sha256[:16] = 996d08777aadc6bf
differ: YES

## C vs old reasoning_on_b256 (quick-wins effect)

control: 15 paired runs from reasoning_on_b256

=== newstack_e2b_r0715  (15 runs) ===
  outcome                    control   variant     delta       t        p
  distinct args / agent        1.853     1.889    +0.036    0.41    0.687
  top-cid share                0.416     0.420    +0.004    0.18    0.857
  distinct args / room         5.267     5.600    +0.333    1.43    0.173
  saturation @r1               0.807     0.799    -0.008   -0.41    0.691
  DRI (final)                  0.316     0.194    -0.122   -1.53    0.149
  meta-consensus (final)       0.698     0.698    +0.000    0.01    0.992
  opinion var (final)          0.254     0.248    -0.006   -0.26    0.799
  LLM calls / run            609.133   605.133    -4.000   -0.45    0.662

wrote agora/analysis/sprint-15-ablation-prep/newstack-e2b-scores.json

## D vs C (model-tier effect, same stack)

control: 15 paired runs from newstack_e2b_r0715

=== newstack_e4b_qat  (15 runs) ===
  outcome                    control   variant     delta       t        p
  distinct args / agent        1.889     1.789    -0.100   -1.35    0.199
  top-cid share                0.420     0.417    -0.002   -0.09    0.932
  distinct args / room         5.600     5.200    -0.400   -1.87    0.082
  saturation @r1               0.799     0.796    -0.003   -0.26    0.797
  DRI (final)                  0.194     0.231    +0.037    0.47    0.645
  meta-consensus (final)       0.698     0.683    -0.015   -1.76    0.100
  opinion var (final)          0.248     0.249    +0.001    0.10    0.921
  LLM calls / run            605.133   647.267   +42.133    6.59    <.001

wrote agora/analysis/sprint-15-ablation-prep/newstack-e4b-scores.json
