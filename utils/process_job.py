import logging
from utils.job_registry import TASK_HANDLERS
from database.job import get_job_database
import tasks.summarize.summarize

logger = logging.getLogger(__name__)
import tasks.detect_language.detect_language
import tasks.entities.entities
import tasks.extraction.extractor
import tasks.ingest.ingest
import tasks.translate.translate
import tasks.ask.ask
import tasks.search.search
import tasks.embedding.embedding
import tasks.key_points.key_points
import tasks.keywords.keywords
import tasks.distribution.distribution
import tasks.correlation.correlation
import tasks.correlation_matrix.correlation_matrix
import tasks.group_by.group_by
import tasks.time_series.time_series
import tasks.outliers.outliers
import tasks.pivot_table.pivot_table
import tasks.summary.summary
import tasks.query.query
import tasks.chart.chart


def process_job(job):
    job_type = job.get("type")
    handler = TASK_HANDLERS.get(job_type)
    db = get_job_database()

    if not handler:
        logger.warning("No handler for job type: %s", job_type)
        db.update_job_status(job['id'], "failed")
        return

    logger.info("Processing job: %s of type: %s", job['id'], job_type)
    try:
        result = handler(job["payload"])
        db.update_job_result(job['id'], result)
        db.update_job_status(job['id'], "processed")
    except Exception as e:
        logger.error("Job %s failed: %s", job['id'], e)
        db.update_job_status(job['id'], "failed")
