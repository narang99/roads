Roads garbage analysis for indian cities.

Workflow:
- `mapillary_downloader` is used to download datasets (done in it's own directory)
- `label-studio-backends` is used to create ML backends to aide dataset creation
- `training` includes scripts/etc. to load the dataset downloaded from mapillary and train a yolo model out of it. We also want support for colab.
  - preparing dataset from exported label-studio labels
  - syncing data to drive folder (which is mounted on my mac)
  - notebook for working in colab for training a yolo model from the dataset
  - the model params itself need to be kept somewhere
