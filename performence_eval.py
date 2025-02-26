import tracemalloc
import timeit
import numpy as np

def measure_speed(iterations: int, func: callable, *args):
    """
    Measures the average execution time of a given function over a specified number of iterations.

    Args:
        iterations (int): The number of times to execute the function.
        func (callable): The function to be measured.
        *args: Variable length argument list to be passed to the function.

    Returns:
        float: The average execution time for the specified number of iterations.
    """
    execution_time = timeit.timeit(lambda: func(args[0], args[1], args[2]), number=iterations)
    average_execution_time = execution_time / iterations
    print(f"Average execution time: {average_execution_time:.6f} seconds")
    return average_execution_time

def measure_memory(iterations: int, func: callable, *args):
    """
    Measures the peak memory usage of a given function over a specified number of iterations.
    Args:
        iterations (int): The number of times to run the function for measurement.
        func (callable): The function whose memory usage is to be measured.
        *args: Variable length argument list to be passed to the function.
    Returns:
        int: The average peak memory usage in bytes.
    """
    peaks = []
    for _ in range(iterations):
        tracemalloc.start()
        func(args[0], args[1], args[2])
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak)
        tracemalloc.reset_peak()
    
    tracemalloc.stop()
    avg_peak = np.mean(peaks)
    print(f"Average Peak: {avg_peak / 1024**2:.2f} MB")
    return avg_peak 

