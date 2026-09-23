# Codex Workflow

## Principle
Do not ask Codex to build the whole project in one prompt.

## Every task should specify
- context
- exact files to create/change
- inputs/outputs
- constraints
- acceptance tests
- what must not change

## Example task
Create dataset_inspector.py that uses NiBabel to inspect a specified BraTS-GLI directory, identify expected modality/mask files, report missing files and dimensions, never modify data, and run on one case as a test.

## Milestones
v0.1 project skeleton
v0.2 dataset inspector
v0.3 preprocessing
v0.4 U-Net
v0.5 metrics
v0.6 visualization
v0.7 experiment logger
v0.8 U-Net++
v0.9 additional models
v1.0 clean benchmark
v1.1 controlled degradation
v1.2 uncertainty
v1.3 agentic selection
v1.4 final dashboard/report

## After every task
Run tests, inspect the diff, understand the module, update Memory.md, and create a meaningful Git checkpoint.
