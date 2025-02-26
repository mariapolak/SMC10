import tracemalloc
import timeit
import numpy as np
import config
import librosa
import glob
import pandas as pd

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

def run_performance_test(): 
    output_csv = f"{config.OUTPUT_EVAL_OBJ_DIR}/performance_{config.TIMESTAMP}.csv"
    test_audio = next(glob.iglob('**/*.flac', root_dir=config.INPUT_DIR, recursive=True))
    x, sr = librosa.load(f"{config.INPUT_DIR}/{test_audio}", sr=None)
    audio_length = len(x) / sr
    tsm_factor = 2
    ps_factor = 2

    data = []

    for tsm_algorithm in config.TSM_ALGORITHMS:
        speed = measure_speed(config.SPEED_ITERATIONS, tsm_algorithm.time_stretch, x, sr, tsm_factor)
        memory = measure_memory(config.MEMORY_ITERATIONS, tsm_algorithm.time_stretch, x, sr, tsm_factor)
        data.append({
            "algorithm": tsm_algorithm.name, "type": "TSM", "speed": speed, 
            "speed_iterations": config.SPEED_ITERATIONS, "memory": memory, "memory_iterations": config.MEMORY_ITERATIONS,
            "audio_length": audio_length, "sample_rate": sr, "factor": tsm_factor
        })
        
    for ps_algorithm in config.PS_ALGORITHMS:
        speed = measure_speed(config.SPEED_ITERATIONS, ps_algorithm.pitch_shift, x, sr, ps_factor)
        memory = measure_memory(config.MEMORY_ITERATIONS, ps_algorithm.pitch_shift, x, sr, ps_factor)
        data.append({
            "algorithm": ps_algorithm.name, "type": "PS", "speed": speed, 
            "speed_iterations": config.SPEED_ITERATIONS, "memory": memory, "memory_iterations": config.MEMORY_ITERATIONS,
            "audio_length": audio_length, "sample_rate": sr, "factor": ps_factor
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

