# Imagem base enxuta com a mesma versão de Python usada no desenvolvimento
FROM python:3.14-slim

# Impede o Python de gravar os arquivos .pyc (Imagem mais limpa, sem cache inútil)
# Desativa o buffering da saída padrão do Python (Docker logs em tempo real)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala as dependências primeiro
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto (notebooks, dados, relatórios)
COPY . .

# Porta padrão do Jupyter Lab
EXPOSE 8888

# Sobe o Jupyter Lab
# Para reproduzir os resultados: abra notebooks/ e execute as células
CMD ["jupyter", "lab", \
    "--ip=0.0.0.0", \
    "--port=8888", \
    "--no-browser", \
    "--allow-root", \
    "--NotebookApp.token=", \
    "--NotebookApp.password="]
