import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing_extensions import Generator

from database.tables import Request

FILE_PATH = Path(__file__).resolve()
ROOT = Path(*FILE_PATH.parts[:-2])


def get_local_session(database_path: Path, **kwargs) -> sessionmaker:
    """
    Instantiate database session to given database path.
    :param database_path: key worded arguments for
    :param kwargs:
    :return:
    """
    # Create a SQLAlchemy engine
    echo = kwargs.get('echo', True)
    # engine = create_engine(f'sqlite:///{DATABASE_PATH}/{database_name}', echo=echo)
    engine = create_engine(f'sqlite:///{database_path}', echo=echo)
    # Create a configured "Session" class
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local


def get_db(local_session: sessionmaker) -> Generator:
    """
    Get database session.
    :param local_session:
    :return:
    """
    db = local_session()
    try:
        yield db
    finally:
        db.close()


def save_request_response(db: Session, method: str, url: str, headers: dict, body: str, status: int,
                          response_headers: dict, response_body: str):
    """
    Save request response.
    :param db: db to which save request response
    :param method: request method
    :param url: request url
    :param headers: request headers
    :param body: request body
    :param status: response status
    :param response_headers: response headers
    :param response_body: response body
    :return:
    """
    headers_json = json.dumps(headers)
    body_json = json.dumps(body)
    response_headers_json = json.dumps(response_headers)
    response_body_json = json.dumps(response_body)

    db_request = Request(
        request_method=method,
        request_url=url,
        request_headers=headers_json,
        request_body=body_json,
        response_status=status,
        response_headers=response_headers_json,
        response_body=response_body_json
    )
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    return db_request
