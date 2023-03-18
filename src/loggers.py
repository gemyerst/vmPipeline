import logging
from typing import Callable

from google.auth.credentials import Credentials
from google.cloud.logging import Client as GoogleLoggingClient
from google.cloud.logging_v2.handlers import CloudLoggingHandler


def init_logging(logger: logging.Logger, credentials: Credentials) -> Callable[[], None]:
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler('out.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    # Cloud logging
    client = GoogleLoggingClient(credentials=credentials)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(CloudLoggingHandler(client))

    def close_callback() -> None:
        file_handler.close()
        client.close()

    return close_callback
