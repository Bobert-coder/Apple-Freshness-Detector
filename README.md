# Apple Freshness detector

## Project description
you can use it at a processing plant to distinguish
rotten from not rotten apples

## Tech Stack
- Python
- TensorFlow
- Keras
- cv2

## Installation
1.`python -m venv venv`
2. Activate virtual environment `venv/scripts/activate`
3. install the dependencies `pip install -r requirements.txt`


## Usage
### Training the model
1. Create dataset directory
2. In dataset directory create subdirectories for Fresh and Rotten classes
3. Add images to subdirectories
4. Adjust `DATASET_PATH` in `freshness_train.py`
5. Run `python freshness_train.py`

### Testing model
1. Adjust `IMG_PATH` and `DATASET_PATH`
2. Run `python freshness_test.py`