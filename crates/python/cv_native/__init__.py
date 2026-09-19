from .cv_native import *
import functools

def resource_group(name):
    """
    Decorator to run a function within a specific Resource Group's thread pool.
    
    This fulfills the requirement of steering libraries like nalgebra 
    to use our TaskScheduler pools.
    
    Example:
        @resource_group("high_priority")
        def process_pointcloud(pc):
            # nalgebra calls inside here will use the "high_priority" pool
            return pc.num_points()
    """
    def decorator(func):
        _group = None
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _group
            if _group is None:
                try:
                    _group = get_resource_group(name)
                except ValueError:
                    # Fallback to default if not created explicitly? 
                    # Or just fail with better message.
                    # For now, let's assume it MUST exist but we fetch it lazily.
                    raise ValueError(f"Resource group '{name}' not found. Create it with 'create_resource_group' first.")
            
            return _group.install(lambda: func(*args, **kwargs))
        return wrapper
    return decorator
