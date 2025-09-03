import os
from dotenv import load_dotenv

load_dotenv()

def fix_database_url(url):
    if url and url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    
    if url and 'sslmode=' not in url:
        url += '?sslmode=require&connect_timeout=10'
    
    return url

class Configdb:
    DATABASE_URL = fix_database_url(os.getenv('DATABASE_URL'))
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'postgresql://avnadmin:AVNS_8yHNy8xy8LVjYfCNoh1@pg-159e6229-jalal-38c7.d.aivencloud.com:22157/defaultdb?sslmode=require'
    SQLALCHEMY_TRACK_MODIFICATIONS = False