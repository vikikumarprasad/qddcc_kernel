FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create --solver libmamba -f environment.yml && conda clean -afy

COPY . .

SHELL ["/bin/bash", "--login", "-c"]

CMD ["conda", "run", "--no-capture-output", "-n", "qddcc_kernel", "python", "Graphs/regenerate_available_figures.py"]
