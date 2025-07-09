from pymongo import MongoClient
import os

class Job:
    
    def __init__(self):
        self.uri = os.getenv("MONGO_URI", "mongodb://root:example@database:27017/")
        self.db_name = os.getenv("MONGO_DB_NAME", "documents")
        self.collection_name = "jobs"
        
        self.client = MongoClient(self.uri)
        db = self.client[self.db_name]
        self.collection = db[self.collection_name]
    
    def get_pending_job(self) -> dict:
        """
        Fetch the oldest pending job by priority order: high > normal > low.
        Returns the oldest job with status 'pending' and highest available priority.
        """
        try:
            priorities = ["high", "normal", "low"]
            for priority in priorities:
                job = self.collection.find_one(
                    {"status": "pending", "priority": priority},
                    sort=[("createdAt", 1)]
                )
                if job:
                    return job
            return None
        except Exception as e:
            print(f"Error fetching pending job: {e}")
            return None

    def update_job_status(self, job_id: str, status: str) -> bool:
        """Update job status"""
        try:
            result = self.collection.update_one(
                {"_id": job_id},
                {"$set": {"status": status}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating job status: {e}")
            return False
    
    def update_job_result(self, job_id: str, result: dict) -> bool:
        """Update job result"""
        try:
            result = self.collection.update_one(
                {"_id": job_id},
                {"$set": {"result": result}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating job result: {e}")
            return False
    

# Singleton instance
_job_database = None

def get_job_database() -> Job:
    """Get the singleton Job service instance"""
    global _job_database
    if _job_database is None:
        _job_database = Job()
    return _job_database