# AltaStata ChRIS Demo

`altastata-chris-demo` is a [_ChRIS_](https://chrisproject.org/) Python plugin that
trains a PyTorch convolutional neural network using data from AltaStata Fortified Data Lake.

The plugin demonstrates:
- Loading training data directly from AltaStata
- Training a CNN model using PyTorch
- Saving the trained model back to AltaStata
- Generating a training summary with metrics

## AltaStata Integration

The plugin loads credentials from `altastata_config.py` in the project root directory.
The config file should contain:
- `user_properties` - AltaStata user properties
- `private_key` - RSA private key
- `account_id` - AltaStata account ID (e.g., "bob123_rsa")

The `ALTASTATA_ACCOUNT_PASSWORD` environment variable is required for authentication.

## Command-Line Usage

```bash
ALTASTATA_ACCOUNT_PASSWORD=your-password \
python app.py \
  --data-root pytorch_test/data/images \
  --model-output pytorch_test/model/best_model.pth \
  input/ output/
```

### Arguments

- `--data-root` (required) - AltaStata path containing training images
- `--model-output` (optional) - AltaStata path where the trained model will be uploaded
- `--summary-filename` (optional) - Filename for training summary JSON (default: "training_summary.json")
- `input/` - ChRIS input directory
- `output/` - ChRIS output directory

## Output Files

- `training_summary.json` - Saved to the project root directory. Contains per-epoch metrics (loss, accuracy, elapsed time) and dataset information.

When `--model-output` is provided, the trained model and provenance file are uploaded to AltaStata.

## Code Structure

- `app.py` - Main plugin entry point (AltaStata integration and orchestration)
- `model_trainer.py` - Model training class (model definition and training logic)
- `altastata_config.py` - AltaStata credentials configuration

## Development Notes

- Dependencies are declared in `requirements.txt` and `setup.py`
- Resource requests for the plugin are set in the `@chris_plugin` decorator
- The model training logic is encapsulated in the `ModelTrainer` class
- Training parameters can be adjusted in `model_trainer.py`
