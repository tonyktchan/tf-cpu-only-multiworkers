# original with GPU : FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-12.py310
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
