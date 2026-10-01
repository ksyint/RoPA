# Temporal gap measurements

Gap evaluation uses one context frame at each requested distance. It evaluates every valid source/target pair and retains the source frame foreground instances. The same nearest-neighbor affinities, temperature and locality apply at each gap. This separates gap sensitivity from the multi-frame propagation history.

```bash
python ropa.py temporal --sequences data/evaluation/sequences.jsonl --gaps 2 32 64 128 256 512 --output outputs/gaps --device cuda
```

The summary reports the first measured gap whose mean J-and-F falls below the chosen fraction of the shortest requested gap. The default fraction is 0.9. Use sequences long enough to contain the requested gaps. `pairs.jsonl` preserves every contributing pair for later grouping.
