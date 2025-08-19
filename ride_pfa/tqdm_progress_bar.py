import logging
from typing import Iterable, TypeVar
from typing import Protocol, Any

log = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = [
    'IterableKwargsCallable',
    'TqdmIterableKwargsCallable',
    'set_custom_progress_bar',
    'with_progress'
]


class IterableKwargsCallable(Protocol[T]):
    def __call__(self, iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
        ...


class DefaultIterableKwargsCallable(IterableKwargsCallable[T]):

    def __call__(self, iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
        return iterable


class TqdmIterableKwargsCallable(IterableKwargsCallable[T]):
    def __init__(self) -> None:
        try:
            from tqdm.auto import tqdm
            self.tqdm = tqdm
        except ImportError:
            log.warning('tqdm is not installed. Progress bar is not used')
            self.tqdm = None

    def __call__(self, iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
        if self.tqdm:
            return self.tqdm(iterable, **kwargs)
        return iterable


_default_progress_bar: IterableKwargsCallable = TqdmIterableKwargsCallable()


def set_custom_progress_bar(func: IterableKwargsCallable[T]):
    """Установите свой прогресс-бар (например, из rich.progress)."""
    global _default_progress_bar
    _default_progress_bar = func


def with_progress(iterable: Iterable[T], **kwargs) -> Iterable[T]:
    return _default_progress_bar(iterable, **kwargs)
