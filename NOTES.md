# Project Notes

## Status
- `python main.py` runs successfully (tested).
- Models: regression, classification, clustering all wired in `main.py`.
- Feature importance exported with RandomForestRegressor.

## Outputs
Generated on run:
- `results/metrics/regression_metrics.csv`
- `results/metrics/classification_metrics.csv`
- `results/metrics/clustering_metrics.csv`
- `results/metrics/clustering_profile.csv`
- `results/metrics/feature_importance.csv`
- `results/figures/regression_pred_vs_true.png`
- `results/figures/classification_confusion_matrix.png`
- `results/figures/classification_roc_curve.png`
- `results/figures/clustering_k_selection.png`
- `results/figures/clustering_profile.png`
- `results/figures/clustering_pca.png`
- `results/figures/feature_importance.png`

## Key Files
- `main.py`: orchestration for regression, classification, clustering.
- `src/models.py`: pipelines + split utilities + clustering/SVD.
- `src/evaluation.py`: metrics + plots, including feature importance.
- `requirements.txt`: dependencies.
- `AI_USAGE.md`: AI usage disclosure.
- `.gitignore`: ignores `results/` outputs.

## Recent Commits
- `Add core dependencies for reproducible runs`
- `Refactor model pipelines and reduce heavy imports`
- `Add evaluation metrics and plotting utilities`
- `Wire end-to-end training flow in main`
- `Ignore generated results outputs`

## Feature Importance (Top)
Observed top features from RandomForestRegressor:
- `num__User_Count`
- `num__Critic_Count`
- `cat__Platform`
- `cat__Publisher`
- `num__Year_of_Release`
- `num__Critic_Score`
- `cat__Developer`
- `cat__Genre`
- `num__User_Score_100`
- `cat__Rating`

Interpretation ready for report:
- Critic/user engagement features explain most variance in sales.
- Platform/publisher/developer + release timing are strong market factors.
- Genre/rating are important but secondary.

## Next Steps (Optional)
- Add `environment.yml`.
- Expand README with more methodology + results summary.
- Start report skeleton with required sections.
