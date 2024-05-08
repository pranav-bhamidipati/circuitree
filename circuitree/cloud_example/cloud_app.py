from celery import Celery
import os

# Load Celery tasks from modules named `tasks.py` in these folders.
# Celery Tasks have the @shared_task decorator.
import worker_tasks
import main_tasks

# Use a Redis server hosted on the cloud
database_url = os.environ["CIRCUITREE_CLOUD_REDIS_URL"]
if not database_url:
    raise ValueError(
        "Please set the CIRCUITREE_CLOUD_REDIS_URL environment variable "
        "to the URL of a Redis server."
    )

# Create a Celery app that uses the Redis server
app = Celery("celery", broker=database_url, backend=database_url)

# Get extra configuration settings
#  - The `CELERY_TASK_ROUTES` setting specifies the queue for each task
app.config_from_object("celeryconfig")