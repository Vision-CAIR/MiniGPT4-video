FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
# FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu20.04
# FROM nvcr.io/nvidia/pytorch:24.01-py3
# Install necessary tools
RUN apt-get update && apt-get install -y curl gnupg wget

# Add the NVIDIA GPG key and repository
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && apt-get update

# Install the NVIDIA container toolkit
RUN apt-get install -y nvidia-container-toolkit
# Set the default runtime to nvidia
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# RUN apt install python3-pip -y
COPY ./ /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install gcc -y

RUN pip install -r requirements.txt

ENV CUDA_VISIBLE_DEVICES=0   
ENV HF_TKN="put your huggingface token here"

EXPOSE 7860
CMD ["python", "minigpt4_video_demo.py"]