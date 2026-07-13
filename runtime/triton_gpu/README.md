## Triton Inference Serving Best Practice for SenseVoice

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up --build
```

### Build Image
Build the docker image from scratch. 
```sh
# build from scratch, cd to the parent dir of Dockerfile.server
docker build . -f Dockerfile/Dockerfile.sensevoice -t soar97/triton-sensevoice:24.05
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "sensevoice-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-sensevoice:24.05
```

### Export SenseVoice Model to Onnx
Please follow the official guide of FunASR to export the sensevoice onnx file. Also, you need to download the tokenizer file by yourself. 
```
from funasr import AutoModel
import os

# 1. 改为你想保存 ONNX 文件的本地目标目录
output_dir = os.path.abspath("/root/FunASR/runtime/triton_gpu/onnx") 

# 2. 【核心】model 参数直接填写你本地下载好的模型文件夹绝对路径
# 例如："/home/user/models/SenseVoiceSmall"
local_model_path = "/root/FunASR/runtime/triton_gpu/iic" 

print("正在从本地路径加载并转换模型...")
model = AutoModel(model=local_model_path)

# 3. 执行导出
# opset_version 建议设为 14 或以上，以完美兼容 Triton 的 ONNX Runtime Backend
model.export(type="onnx", quantize=False, opset_version=18, save_dir=output_dir)

print(f"导出成功！ONNX 相关文件已保存在: {output_dir}")
```
### Launch Server
Log of directory tree:
```sh
model_repo_sense_voice_small
|-- encoder
|   |-- 1
|   |   `-- model.onnx -> /your/path/model.onnx
|   |   `-- model.data -> /your/path/model.data
|   `-- config.pbtxt
|-- feature_extractor
|   |-- 1
|   |   `-- model.py
|   |-- am.mvn
|   |-- config.pbtxt
|   `-- config.yaml
|-- scoring
|   |-- 1
|   |   `-- model.py
|   |-- chn_jpn_yue_eng_ko_spectok.bpe.model -> /your/path/chn_jpn_yue_eng_ko_spectok.bpe.model
|   `-- config.pbtxt
`-- sensevoice
    |-- 1
    `-- config.pbtxt

8 directories, 10 files


# launch the service 
tritonserver --model-repository /workspace/model_repo_sensevoice_small \
             --pinned-memory-pool-byte-size=512000000 \     #根据显存实际情况
             --cuda-memory-pool-byte-size=0:1024000000
```


### Benchmark using Dataset
```sh
git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
cd Triton-ASR-Client
num_task=32
python3 client.py \
    --server-addr localhost \
    --server-port 10086 \
    --model-name sensevoice \
    --compute-cer \
    --num-tasks $num_task \
    --batch-size 16 \
    --manifest-dir ./datasets/aishell1_test
```

Benchmark results below were based on Aishell1 test set with a single V100, the total audio duration is 36108.919 seconds.
|concurrent-tasks | batch-size-per-task | processing time(s) | RTF |
|----------|--------------------|------------|---------------------|
| 32 (onnx fp32)                | 16 | 67.09 | 0.0019|
| 32 (onnx fp32)                | 1 | 82.04  | 0.0023|

(Note: for batch-size-per-task=1 cases, tritonserver could use dynamic batching to improve throughput.)

## Acknowledge
This part originates from NVIDIA CISI project. We also have TTS and NLP solutions deployed on triton inference server. If you are interested, please contact us.
