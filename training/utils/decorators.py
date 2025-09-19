# utils/decorators.py
import torch.distributed as dist
from functools import wraps
from typing import Callable, Any


def rank_zero(fn: Callable) -> Callable:
    """
    Decorator to ensure that the decorated function is only executed on rank-0 (main process)
    in a distributed (DDP) setup. If not in a distributed setup, the function is always executed.

    Args:
        fn (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function that only runs on rank-0.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function that checks if the current process is rank-0 before executing the function.

        Args:
            *args: Positional arguments for the decorated function.
            **kwargs: Keyword arguments for the decorated function.

        Returns:
            Any: The return value of the decorated function if on rank-0, otherwise None.
        """
        rank_is_zero = (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

        if not rank_is_zero:
            return None
        return fn(*args, **kwargs)

    return wrapper
