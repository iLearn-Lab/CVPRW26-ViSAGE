
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    

RUN ln -s /usr/bin/python3.12 /usr/bin/python


WORKDIR /workspace/ilearn

RUN pip install --no-cache-dir torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 7. 拷贝并安装其他 Python 依赖
COPY requirements.txt .
# 注意：如果你的 requirements.txt 里还有 torch，pip 会自动识别已安装而跳过
RUN pip install --no-cache-dir -r requirements.txt

# 8. 将项目所有文件（Expert1, checkpoints, 等）拷贝进容器
COPY . .

# 9. 容器启动时默认进入 bash
CMD ["/bin/bash"]