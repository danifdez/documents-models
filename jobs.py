import time
from database.job import get_job_database
from utils.process_job import process_job

POLL_INTERVAL_MS = 1000  # milliseconds


def main():
    db = get_job_database()
    print("Job service started. Polling for pending jobs...")
    while True:
        pending_job = db.get_pending_job()
        if pending_job:
            process_job(pending_job)

        time.sleep(POLL_INTERVAL_MS / 1000.0)


if __name__ == "__main__":
    main()
