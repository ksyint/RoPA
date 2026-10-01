# Point correspondence

Point archives are NPZ files containing `features`, `queries`, `targets`, `visible`, `grid_height` and `grid_width`. Features have shape T,H*W,D. Queries are Q,3 in frame,y,x order. Targets are Q,T,2 in x,y order and visibility is Q,T. All coordinates use the feature patch grid.

```bash
python ropa.py temporal --points data/evaluation/points.npz --thresholds 1 2 4 8 16 --output outputs/points --device cuda
```

The tracker samples the query embedding bilinearly and finds its most similar patch independently in each frame. Evaluation excludes query-frame observations and invisible targets. It exports tracks, mean visible-point error and thresholded point accuracy in patch-grid pixels.
