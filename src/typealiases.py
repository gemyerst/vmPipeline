

from typing import Callable, TypeVar


T = TypeVar("T")
Runnable = Callable[[], None]
Predicate = Callable[[], bool]
Consumer = Callable[[T], None]
