from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
# Create a Base class for our classes definitions
Base = declarative_base()


class Request(Base):
    __tablename__ = 'requests'

    id = Column(Integer, primary_key=True, index=True)
    request_method = Column(String, index=True)
    request_url = Column(String, index=True)
    request_headers = Column(Text)
    request_body = Column(Text)
    response_status = Column(Integer)
    response_headers = Column(Text)
    response_body = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)

