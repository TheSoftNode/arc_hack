# Models Directory

This directory contains saved model files and checkpoints.

## Structure

- `checkpoints/` - Training checkpoints and intermediate models
- `trained/` - Final trained models ready for inference
- `pretrained/` - Pre-trained models downloaded from external sources

## File Types

- `.pkl` / `.joblib` - Scikit-learn models
- `.pt` / `.pth` - PyTorch models
- `.h5` / `.hdf5` - Keras/TensorFlow models
- `.onnx` - ONNX format models for cross-platform inference
- `.json` - Model configuration files

## Usage

Models are automatically saved here by the training pipeline and loaded by the inference system.

## Git Ignore

Large model files are ignored by git. Use model versioning systems like MLflow or DVC for tracking model artifacts.
