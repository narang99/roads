#!/bin/bash
set -euxo pipefail

# generate data
# split
# tar
# move to colab
# provide instructions to run in colab
# should always be reproducible

TMP_DIR="./tmp"

SAMPLE_DIR="/Users/hariomnarang/Desktop/personal/roads/mapillary_downloader/data/samples" 

### parse
FRAG_DIR="$1"
PROJ_NAME="$2"
NUM_IMGS="$3"

if [[ -z "$FRAG_DIR" || -z "$PROJ_NAME" ]]; then
  echo "Usage: $0 <FRAG_DIR> <project-name>" >&2
  exit 1
fi

ROOT="./tmp/$PROJ_NAME"
mkdir $ROOT || true


echo "Fragments: $FRAG_DIR"
echo "Project Name: $PROJ_NAME"
echo "working space: $ROOT"

# generate data
GEN_DATA_DIR="$ROOT/gen_data"
rm -rf $GEN_DATA_DIR || true
mkdir $GEN_DATA_DIR || true
uv run scripts/gen_data.py -i "${SAMPLE_DIR}" -d "${GEN_DATA_DIR}" -n "$NUM_IMGS" -f "${FRAG_DIR}"

# split
SPLIT_DIR="$ROOT/split"
rm -rf $SPLIT_DIR || true
mkdir $SPLIT_DIR || true
uv run scripts/yolo_split_ds.py \
    --images "./$GEN_DATA_DIR/images" \
    --labels "./$GEN_DATA_DIR/labels" \
    --output "$SPLIT_DIR" \
    --classes trash

TAR_NAME="synth_train_${PROJ_NAME}.tar"
TAR_LOC="$ROOT/$TAR_NAME"
tar -cvf "$TAR_LOC" "$SPLIT_DIR"

say "compression done, starting upload"

S3_BUCK="narang-public-s3"
aws s3 cp "$TAR_LOC" "s3://$S3_BUCK/"

say "upload done"

cat << EOF
Execution on this side is finished, copy the cells below in colab and run them to download and initialise data 





########################### function definitions ##########################################################
! pip install boto3

from pathlib import Path
import boto3
import zstandard as zstd

EXTRACTED_PATH = Path("/content/synth")

def download_synth_tgz():
  s3 = boto3.client(
      's3',
      aws_access_key_id='<>',
      aws_secret_access_key='<>',
      region_name='us-east-1',
  )
  s3.download_file('narang-public-s3', 'synth_train_$PROJ_NAME.tar', 'synth_train_$PROJ_NAME.tar')

def rm_mac_fake_files(root=EXTRACTED_PATH):
  count = 0
  for p in list(root.rglob("._*.jpg")):
    p.unlink()
    count += 1

  for p in list(root.rglob("._*.txt")):
    p.unlink()
    count += 1
  print("deleted", count, "files")

################ download and extract and clean ###########################
download_synth_tgz()
! rm -rf /content/synth || true
! mkdir /content/synth || true
! tar -xvf "/content/synth_train_$PROJ_NAME.tar" -C /content/synth
rm_mac_fake_files()

################## dataset directory definition ##############################
from pathlib import Path

DATASET_DIR = Path("/content/synth/tmp/$PROJ_NAME/split")
PROJECT_NAME = "$PROJ_NAME"
GDRIVE_ROOT = Path("/content/gdrive/Othercomputers/My MacBook Pro/gdrive-sync/")
YOLO_PROJECT = str(GDRIVE_ROOT / PROJECT_NAME)
DATASET_DIR.exists()
EOF