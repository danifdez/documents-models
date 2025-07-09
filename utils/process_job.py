from utils.job_registry import TASK_HANDLERS
from database.job import get_job_database
import tasks.summarize.summarize
import tasks.detect_language.detect_language
import tasks.entities.entities
import tasks.extraction.extractor
import tasks.ingest.ingest
import tasks.translate.translate
import tasks.ask.ask
import tasks.search.search

def process_job(job):
    job_type = job.get("type")
    handler = TASK_HANDLERS.get(job_type)
    print(f"Job type: {job_type}, Handler: {handler}")
    if handler:
        print(f"Processing job: {job['_id']} of type: {job_type}")
        db = get_job_database()
        db.update_job_status(job['_id'], "processing")
        result = handler(job["payload"])
        db.update_job_result(job['_id'], result)
        db.update_job_status(job['_id'], "processed")
    else:
        print(f"No handler for job type: {job_type}")