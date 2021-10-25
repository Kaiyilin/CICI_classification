# Bonaus: Real example for database connection, the Mongo connection
from pymongo import MongoClient

class MongoDBConnection(object):
    """MongoDB Connection"""
    def __init__(self, host='localhost', port='27017'):
        self.host = host
        self.port = port
        self.connection = None
    def __enter__(self):
        self.connection = MongoClient(self.host, self.port)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()