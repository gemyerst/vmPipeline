import logging

from google.auth.credentials import Credentials
from google.cloud.logging import Client as GoogleLoggingClient
from google.cloud.logging_v2.handlers import CloudLoggingHandler

from typealiases import Runnable


def init_logging(logger: logging.Logger, credentials: Credentials) -> Runnable:
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    # Cloud logging
    client = GoogleLoggingClient(credentials=credentials)

    logger.addHandler(stdout_handler)
    logger.addHandler(CloudLoggingHandler(client))

    def close_callback() -> None:
        client.close()

    return close_callback
