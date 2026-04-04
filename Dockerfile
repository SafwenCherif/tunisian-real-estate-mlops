FROM apache/airflow:2.8.1-python3.11

# Install project runtime dependencies once at build time.
# This avoids slow container startup caused by `_PIP_ADDITIONAL_REQUIREMENTS`
# and prevents Airflow tasks from staying in `queued` while workers aren't ready.

USER airflow

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    dvc dvc-gdrive pandas numpy scikit-learn requests beautifulsoup4 lxml \
    mlflow geopy nbconvert ipykernel \
    matplotlib seaborn

