FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# set a directory for the app
WORKDIR /app

# copy these files to the container
COPY [ "Pipfile", "Pipfile.lock", "starter.py", "./" ]

# install dependencies
RUN pip install -U pip
RUN pip install pipenv 
RUN pipenv install --system --deploy

# run the script
ENTRYPOINT ["python", "starter.py"]
