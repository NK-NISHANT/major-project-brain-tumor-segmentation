# Experiment Protocol

## Baseline principle
Keep the dataset split, preprocessing, augmentation, optimizer, learning rate, batch size, epochs, loss, and seed controlled when comparing architectures.

## Metrics
Primary: Dice, IoU/Jaccard.
Secondary: precision, recall/sensitivity, specificity where appropriate, Hausdorff distance if feasible, parameter count, training time, inference time.

## Traceability
Every result should identify dataset split, model/version, configuration, checkpoint, code revision, and experiment ID.

## Clean vs degraded
Do not compare conditions without a documented evaluation protocol. The exact degradation/noise types and severity levels should follow mentor guidance before final experiments.
