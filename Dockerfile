FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

COPY ./ /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install gcc -y

RUN pip install -r req.txt

EXPOSE 7860

ENV CUDA_VISIBLE_DEVICES=0   

CMD ["python", "minigpt4_video_demo.py"]