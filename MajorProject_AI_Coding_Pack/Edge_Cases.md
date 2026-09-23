# Edge Cases

## Data
- missing modality
- missing mask
- corrupted NIfTI
- inconsistent dimensions
- unexpected orientation/spacing
- empty tumor mask
- duplicate case
- train/validation leakage
- severe class imbalance

## Preprocessing
- NaN/Inf
- extreme intensity outliers
- all-zero slice
- no-tumor slice
- crop removes anatomy
- incorrect mask interpolation
- image/mask misalignment

## Training
- GPU out-of-memory
- NaN loss
- exploding gradients
- validation metric stagnation
- checkpoint corruption
- slow data loading

## Evaluation
- ambiguous Dice when both masks are empty
- division by zero
- wrong label
- accidental validation-data training
- unfair model comparison

## Agent
- no valid candidate
- all candidates highly uncertain
- requested modality missing
- threshold boundary
- compute budget exceeded
- invalid action
- contradictory quality/uncertainty signals

## Demo
- unsupported file
- oversized input
- slow inference
- missing checkpoint
- wrong device
- CPU fallback
- stale result displayed as current
