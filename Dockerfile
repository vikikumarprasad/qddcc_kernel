FROM anaconda/miniconda:26.9.4

# TeX for the LaTeX-typeset labels of Figure 1(b).
RUN apt-get update \
    && apt-get install -y --no-install-recommends texlive-latex-extra cm-super-minimal dvipng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# environment.yml lists conda-forge as its only channel.
COPY environment.yml .
RUN conda config --system --remove channels defaults \
    && conda env create --solver libmamba -f environment.yml \
    && conda clean -afy

# Scripts and saved result tables. data/, images/ and paper_results/ are mounted at run time.
COPY Circuit_Classes.py EncodingCircuit.py FunctionEncoder.py QKernel_Methods_Test.py \
     Test_Blind.py Training_Functions.py Transferability_Test.py README.md LICENCE.txt ./
COPY ["Data Generation", "Data Generation/"]
COPY Graphs Graphs/
COPY Experiments Experiments/

ENV PATH=/opt/miniconda3/envs/qddcc_kernel/bin:$PATH \
    HOME=/tmp \
    MPLBACKEND=Agg

CMD ["python", "Graphs/regenerate_available_figures.py"]
