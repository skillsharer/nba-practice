uv venv .venv
source .venv/bin/activate
uv sync
cd ./data
./download.sh
cd ../src
python train.py
deactivate