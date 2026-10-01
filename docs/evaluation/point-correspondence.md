# Point correspondence

Point archives are NPZ files containing `features`, `queries`, `targets`, `visible`, `grid_height` and `grid_width`. Features have shape T,H*W,D. Queries are Q,3 in frame,y,x order. Targets are Q,T,2 in x,y order and visibility is Q,T. All coordinates use the feature patch grid.

```bash
python ropa.py temporal --points data/evaluation/points.npz --thresholds 1 2 4 8 16 --output outputs/points --device cuda
```

The tracker samples the query embedding bilinearly and finds its most similar patch independently in each frame. Evaluation excludes query-frame observations and invisible targets. It exports tracks, mean visible-point error and thresholded point accuracy in patch-grid pixels.

TAP-Vid and JHMDB share the `tracking` command:

```bash
python ropa.py prepare-tracks --index features/index.jsonl --annotations annotations/tracks.jsonl --output prepared/tracks
python ropa.py tracking tapvid --sequences prepared/tracks/sequences.jsonl --query-mode strided --output results/tapvid
python ropa.py tracking jhmdb --sequences prepared/tracks/sequences.jsonl --normalization max-side --output results/jhmdb
```

The annotation JSONL contains `clip_id`, `points`, `image_height`, `image_width` and an optional sequence `name`. A points NPZ contains `queries` shaped Q×3 as `[frame,y,x]`, `targets` shaped Q×T×2 as `[x,y]`, and Boolean `visible` shaped Q×T. Coordinates refer to the original image. Set `coordinates` to `normalized` for fractions of width and height.

For TAP-style NPZ arrays, set `format: "tapvid"` and provide `query_points`, `target_points` and `occluded`. JHMDB also requires positive `normalizers` shaped T or Q×T, or body `boxes` shaped T×4 or Q×T×4 as `[x1,y1,x2,y2]`. A `body_masks` NPY path can supply boxes from nonzero foreground.

When annotation frames are denser than feature tubelets, supply `frame_indices` to select the matching annotation times. The default query policy requires an exact retained query frame. `query_policy: "nearest"` additionally needs `maximum_query_distance`, measured in original frames, and moves each query to the corresponding visible reference point.

Direct benchmark manifests can instead pair `source_timestamps` in the points NPZ with `tubelet_timestamps` in each sequence record. Coordinates are interpolated only where both adjacent reference observations are visible. The processor resize and crop are inverted before distance scoring.

TAP scores use the 256-pixel coordinate scale. `--query-mode first` evaluates frames after the query, while `strided` excludes only the query frame. `--visibility-threshold` thresholds cosine similarity and `--cycle-threshold` thresholds round-trip patch-grid error. Without either switch, the correspondence predictor marks all frames visible.

TAP conversion treats both `query_points` and `target_points` as pixel coordinates by default, matching sampled TAP evaluation arrays. Set `query_coordinates` and `target_coordinates` individually to `pixels` or `normalized` when an archive uses another convention. `target_layout: "tqx"` accepts T×Q×2 targets, otherwise the expected layout is Q×T×2. An optional singleton batch axis is removed before layout conversion.

`tracking --tracking-stride 8` bilinearly resamples the frozen feature maps to an eight-pixel processor-crop stride before matching. Query mapping and inverse mapping use the resulting grid. The original annotations stay in image pixels. Timestamp alignment also interpolates JHMDB boxes or normalizers along their time axis.

TAP reports include temporal-gap bins controlled by `--temporal-bins 1 4 16 64 256`. `--visibility-sweep 0.1 0.2 0.3` evaluates cosine thresholds using the same saved correspondence predictions and optional cycle filter. These sweeps report visible-point precision and recall alongside point, Jaccard and occlusion scores.

Within each TAP video, counts are summed over query and time before computing point accuracy, Jaccard and occlusion accuracy. The final benchmark score averages the resulting video scores. Per-track counts and ratios remain available in the threshold details.
