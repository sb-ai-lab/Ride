__all__ = [
    'RideException',
    'RidePathNotFoundException'
]


class RideException(Exception):
    """Base class for exceptions in Ride."""


class RidePathNotFoundException(RideException):
    """Exception for algorithms that should return a path when running
        on graphs where such a path does not exist."""
