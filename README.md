# Code translation + documentation
Jack Gindi and Lorenzo Meninato

Project proposal: https://www.overleaf.com/project/64179a5010a2d6ff7f485fe4

To activate the project environment:

```
conda create -n nlp-final-project python=3.8
conda activate nlp-final-project
pip install -r requirements.txt

# Copy your kaggle key to ~/.kaggle/kaggle.json
# should look like: {"username":"lorenzom","key":"<KEY>"}
python3 data.py # downloads and unzips data to ./data
```

To run the code with optional arguments:

```
python3 main.py --max_function_length <MAX_FUNCTION_LENGTH> \
                --d_model <D_MODEL> \
                --d_hid <D_HID> \
                --num_layers <NUM_LAYERS> \
                --nhead <NHEAD> \
                --dropout <DROPOUT> \
                --batch_size <BATCH_SIZE> \
                --num_workers <NUM_WORKERS>
```

Replace <MAX_FUNCTION_LENGTH>, <D_MODEL>, <D_HID>, <NUM_LAYERS>, <NHEAD>, <DROPOUT>, <BATCH_SIZE>, and <NUM_WORKERS> with your desired values or simply use the default values by omitting the optional arguments.

To run with default args: `python3 main.py`

Example running with custom args:

```
python3 main.py --max_function_length 256 \
                --d_model 512 \
                --d_hid 1024 \
                --num_layers 4 \
                --nhead 8 \
                --dropout 0.2 \
                --batch_size 32 \
                --num_workers 8
```