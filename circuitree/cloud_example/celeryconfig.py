# Specify different queues for main and worker nodes
CELERY_TASK_ROUTES = {
    'tasks.get_reward_celery': {'queue': 'worker_node'},
    'tasks.run_mcts_parallel': {'queue': 'main_node'},
}
