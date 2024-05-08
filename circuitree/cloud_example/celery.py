from celery import Celery
import os

# Use a Redis server hosted on the cloud
database_url = os.environ["CIRCUITREE_CLOUD_REDIS_URL"]
if not database_url:
    raise ValueError(
        "Please set the CIRCUITREE_CLOUD_REDIS_URL environment variable "
        "to the URL of a Redis server."
    )
app = Celery("celery", broker=database_url, backend=database_url)
app.autodiscover_tasks(["circuitree.cloud_example.tasks"])