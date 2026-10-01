# Checkpoint inspection

Inspection reports parameter counts, storage sizes, tensor dtypes and the actual temporal frequencies recorded in every attention block. The reported half-cycle and local scale come from the saved frequency tensors. The task configuration records the selected temporal recipe.

```bash
python ropa.py checkpoint inspect --checkpoint outputs/ropa/last.pt
python ropa.py checkpoint compare --first outputs/earlier/last.pt --second outputs/later/last.pt
```

Comparison reports tensor shape changes and RMS or maximum parameter differences in bounded chunks. Use it to check which parts changed between adaptation runs. Training with an initialization checkpoint retains the temporal frequencies selected by the new recipe, while inference restores the learned checkpoint state.
