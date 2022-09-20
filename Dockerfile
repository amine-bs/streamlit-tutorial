FROM inseefrlab/onyxia-python-minimal

COPY . /app
WORKDIR /app

RUN pip --no-cache-dir --upgrade install -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "main.py"]
