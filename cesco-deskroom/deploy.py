from prefect import flow
from prefect.client.schemas.schedules import CronSchedule
from prefect.runner.storage import GitRepository
from prefect_github import GitHubCredentials


def deploy():
    voc_event_flow = flow.from_source(
        source=GitRepository(
            url="https://github.com/CloserLabs/cesco-deskroom",
            branch="main",
            credentials=GitHubCredentials.load("github-credentials"),
        ),
        entrypoint="flows/analyze_voc_message/main.py:analyze_voc",
    )

    voc_event_flow.deploy(
        name="Analyze VoC Messages",
        work_pool_name="docker-pool",
        job_variables={
            "image_pull_policy": "Never",
            "network_mode": "cesco-deskroom_cesco_net",
            "env": {
                "DB_HOST": "cesco_postgres",
                "DB_USER": "cesco_admin",
                "DB_PASS": "Cesco_1588",
                "DB_NAME": "deskroom_core",
            },
            "stream_output": True,
        },
        # Run every hour at minute 0
        schedules=[
            CronSchedule(
                cron="0 * * * *",
                timezone="Asia/Seoul",
            )
        ],
        tags=["service: voc-analysis", "type: voc-event-generation"],
    )

    current_user_flow = flow.from_source(
        source=GitRepository(
            url="https://github.com/CloserLabs/cesco-deskroom",
            branch="main",
            credentials=GitHubCredentials.load("github-credentials"),
        ),
        entrypoint="flows/analyze_current_user/main.py:analyze_current_user",
    )

    current_user_flow.deploy(
        name="Analyze Current Users",
        work_pool_name="docker-pool",
        job_variables={
            "image_pull_policy": "Never",
            "network_mode": "cesco-deskroom_cesco_net",
            "env": {
                "DB_HOST": "cesco_postgres",
                "DB_USER": "cesco_admin",
                "DB_PASS": "Cesco_1588",
                "DB_NAME": "deskroom_core",
            },
            "stream_output": True,
        },
        # Run every first day of month at 02:00 AM
        schedules=[
            CronSchedule(
                cron="0 2 1 * *",
                timezone="Asia/Seoul",
            )
        ],
        tags=["service: current-user-analysis", "type: current-user-analysis"],
    )

    potential_user_flow = flow.from_source(
        source=GitRepository(
            url="https://github.com/CloserLabs/cesco-deskroom",
            branch="main",
            credentials=GitHubCredentials.load("github-credentials"),
        ),
        entrypoint="flows/analyze_potential_user/main.py:analyze_potential_user",
    )

    potential_user_flow.deploy(
        name="Analyze Potential Users",
        work_pool_name="docker-pool",
        job_variables={
            "image_pull_policy": "Never",
            "network_mode": "cesco-deskroom_cesco_net",
            "env": {
                "DB_HOST": "cesco_postgres",
                "DB_USER": "cesco_admin",
                "DB_PASS": "Cesco_1588",
                "DB_NAME": "deskroom_core",
            },
            "stream_output": True,
        },
        # Run every first day of month at 02:00 AM
        schedules=[
            CronSchedule(
                cron="0 2 1 * *",
                timezone="Asia/Seoul",
            )
        ],
        tags=["service: potential-user-analysis", "type: potential-user-analysis"],
    )

    monitoring_flow = flow.from_source(
        source=GitRepository(
            url="https://github.com/CloserLabs/cesco-deskroom",
            branch="main",
            credentials=GitHubCredentials.load("github-credentials"),
        ),
        entrypoint="flows/monitor_model_health/main.py:monitor_model_health",
    )

    monitoring_flow.deploy(
        name="Monitor Model Health",
        work_pool_name="docker-pool",
        job_variables={
            "image_pull_policy": "Never",
            "network_mode": "cesco-deskroom_cesco_net",
            "env": {
                "DB_HOST": "cesco_postgres",
                "DB_USER": "cesco_admin",
                "DB_PASS": "Cesco_1588",
                "DB_NAME": "deskroom_core",
            },
            "stream_output": True,
        },
        # Run every first day of month at 02:00 AM
        schedules=[
            CronSchedule(
                cron="0 2 1 * *",
                timezone="Asia/Seoul",
            )
        ],
        tags=["service: model-health-monitoring", "type: model-health-monitoring"],
    )


if __name__ == "__main__":
    deploy()
