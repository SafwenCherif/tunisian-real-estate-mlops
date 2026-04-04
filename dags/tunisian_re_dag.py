from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Configuration par défaut (gestion des erreurs, retries)
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'retries': 2, # Si le site mubawab plante, Airflow réessaiera 2 fois
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'tunisian_real_estate_pipeline',
    default_args=default_args,
    description='Automated Scraping, DVC, and MLflow Pipeline',
    schedule_interval='0 3 * * *', # Tourne tous les jours à 3h00 du matin
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'scraping', 'modeling'],
) as dag:

    # Étape 1 : Scraping incrémental
    scrape_task = BashOperator(
        task_id='incremental_scrape',
        bash_command='cd /opt/airflow/project && python pipeline/incremental_scrape.py',
        queue='default',
    )

    # Étape 2 : Géocodage
    geo_task = BashOperator(
        task_id='geo_enrichment',
        bash_command='cd /opt/airflow/project && python pipeline/incremental_geo.py',
        queue='default',
    )

    # Étape 3 : Entraînement et MLflow (La magie de DVC !)
    # Au lieu d'appeler 3 scripts, on laisse DVC s'occuper de tout le DAG ML
    dvc_repro_task = BashOperator(
        task_id='run_ml_pipeline',
        bash_command='cd /opt/airflow/project && dvc repro',
        queue='default',
    )

    # Étape 4 : Sauvegarde du nouveau modèle et CSV sur Google Drive
    dvc_push_task = BashOperator(
        task_id='backup_to_gdrive',
        bash_command='cd /opt/airflow/project && dvc push',
        queue='default',
    )

    # L'ordre d'exécution (Les flèches du graphe)
    scrape_task >> geo_task >> dvc_repro_task >> dvc_push_task