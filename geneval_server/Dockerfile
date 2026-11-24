# GenEval scoring, as used for the paper:
#  - Flow-GRPO's reward server (yifan123/reward-server at 8245af2) for the dense
#    training reward, with one fix: it passed RGB arrays to MMDetection, which
#    expects BGR (official GenEval reads images from files). See reward-server.patch.
#  - Official GenEval (djghosh13/geneval at af4902f) for the benchmark.
# Both use MMDetection 2.28.2 with the GenEval Mask2Former Swin-S detector.
#
#   docker build -t ponder-geneval geneval_server
#   docker run --gpus all -p 127.0.0.1:18085:18085 ponder-geneval   # reward server on GPU 0, see below for --workers
#   docker run --gpus all -v $PWD/runs:/runs ponder-geneval \
#       python3.10 /opt/geneval/geneval/evaluation/evaluate_images.py /runs/geneval/r004 \
#       --outfile /runs/geneval/r004.jsonl --model-path /opt/geneval/object_detector \
#       --model-config /opt/geneval/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
#   docker run -v $PWD/runs:/runs ponder-geneval \
#       python3.10 /opt/geneval/geneval/evaluation/summary_scores.py /runs/geneval/r004.jsonl
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS=8 PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" HF_HOME=/opt/geneval/huggingface

RUN apt-get update && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates curl git libgl1 libglib2.0-0 ninja-build \
        python3.10 python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --no-cache-dir pip==25.1.1 setuptools==75.8.2 wheel==0.45.1
RUN python3.10 -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128
RUN python3.10 -m pip install --no-cache-dir addict==2.4.0 clip-benchmark==1.6.1 cython==3.0.12 \
        flask==3.1.0 gunicorn==23.0.0 matplotlib==3.10.0 ninja==1.11.1.3 numpy==1.26.0 \
        open-clip-torch==2.31.0 opencv-python-headless==4.11.0.86 packaging==24.2 pandas==2.2.3 \
        pillow==11.1.0 psutil==6.1.1 pycocotools==2.0.8 pyyaml==6.0.2 requests==2.32.3 scipy==1.15.1 \
        six==1.17.0 terminaltables==3.1.10 yapf==0.40.2

WORKDIR /opt/geneval
# MMCV 1.7.2 (no CUDA 12.8 wheel exists, so its ops are compiled here) and MMDetection 2.28.2.
RUN git clone https://github.com/open-mmlab/mmcv.git && git -C mmcv checkout 4c01b026f0afa5a91a5f54aea313788da1e40f95 \
    && python3.10 -m pip install --no-cache-dir --no-build-isolation ./mmcv \
    && git clone https://github.com/open-mmlab/mmdetection.git \
    && git -C mmdetection checkout e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1 \
    && python3.10 -m pip install --no-cache-dir --no-deps --editable ./mmdetection
RUN mkdir object_detector && curl --fail --location --retry 5 --output \
        object_detector/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
        https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
    && echo "743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54  object_detector/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" | sha256sum --check
COPY reward-server.patch .
RUN git clone https://github.com/yifan123/reward-server.git && git -C reward-server checkout 8245af22b75fe640950a880ad31fe702a5d9a014 \
    && git -C reward-server apply /opt/geneval/reward-server.patch \
    && git clone https://github.com/djghosh13/geneval.git && git -C geneval checkout af4902f24d3ca90ebbb446dd9891a59e0f82725f
# Cache the CLIP ViT-L/14 used for color classification.
RUN python3.10 -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')"

WORKDIR /opt/geneval/reward-server
EXPOSE 18085
# reward-server's gunicorn.conf.py pins worker k to GPU k, so --workers sets the GPU count.
CMD ["python3.10", "-m", "gunicorn", "--bind", "0.0.0.0:18085", "--workers", "1", "app_geneval:create_app()"]
