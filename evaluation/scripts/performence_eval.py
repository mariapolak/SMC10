import tracemalloc
import timeit
import numpy as np
import config
import librosa
import glob
import pandas as pd
import logging
from pathlib import Path

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
    logging.info(f"Measuring speed for {func.__name__} with arguments: {args}")
    logging.info(f"Number of iterations: {iterations}")
    execution_time = timeit.timeit(lambda: func(args[0], args[1], args[2]), number=iterations)
    average_execution_time = execution_time / iterations
    logging.info(f"Average execution time: {average_execution_time:.6f} seconds")
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
    logging.info(f"Average Peak: {avg_peak / 1024**2:.2f} MB")
    return avg_peak 

def run_performance_test(): 
    logging.info("Running performance test...")
    output_csv = f"{config.EVAL_OBJ_DIR}/performance/performance_{config.TIMESTAMP}.csv"

    # Initialize the CSV file with headers
    if not Path(output_csv).exists():
        pd.DataFrame(columns=[
            "algorithm", "type", "speed", "speed_iterations", "memory", 
            "memory_iterations", "audio_length", "sample_rate", "factor"
        ]).to_csv(output_csv, index=False)

    files = glob.glob('**/*.wav', root_dir=config.INPUT_DIR, recursive=True)[:5]
    for test_audio in files:
        logging.info(f"Processing file: {test_audio}")
        x, sr = librosa.load(f"{config.INPUT_DIR}/{test_audio}", sr=None)
        audio_length = len(x) / sr
        for tsm_algorithm in config.TSM_ALGORITHMS:
            for tsm_factor in config.ALGORITHM_FACTORS["tsm_factors"]:
                logging.info(f"Testing TSM algorithm: {tsm_algorithm.name} with factor: {tsm_factor}")
                speed = measure_speed(config.SPEED_ITERATIONS, tsm_algorithm.time_stretch, x, sr, tsm_factor)
                memory = measure_memory(config.MEMORY_ITERATIONS, tsm_algorithm.time_stretch, x, sr, tsm_factor)
                data = [{
                    "algorithm": tsm_algorithm.name, "type": "TSM", "speed": speed, 
                    "speed_iterations": config.SPEED_ITERATIONS, "memory": memory, "memory_iterations": config.MEMORY_ITERATIONS,
                    "audio_length": audio_length, "sample_rate": sr, "factor": tsm_factor
                }]
                pd.DataFrame(data).to_csv(output_csv, mode='a', header=False, index=False)
            
        for ps_algorithm in config.PS_ALGORITHMS:
            for ps_factor in config.ALGORITHM_FACTORS["ps_factors"]:
                logging.info(f"Testing PS algorithm: {ps_algorithm.name} with factor: {ps_factor}")
                speed = measure_speed(config.SPEED_ITERATIONS, ps_algorithm.pitch_shift, x, sr, ps_factor)
                memory = measure_memory(config.MEMORY_ITERATIONS, ps_algorithm.pitch_shift, x, sr, ps_factor)
                data = [{
                    "algorithm": ps_algorithm.name, "type": "PS", "speed": speed, 
                    "speed_iterations": config.SPEED_ITERATIONS, "memory": memory, "memory_iterations": config.MEMORY_ITERATIONS,
                    "audio_length": audio_length, "sample_rate": sr, "factor": ps_factor
                }]
                pd.DataFrame(data).to_csv(output_csv, mode='a', header=False, index=False)

if __name__ == "__main__":
    Path(f"{config.EVAL_OBJ_DIR}/performance/").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=f"{config.EVAL_OBJ_DIR}/performance/performance_{config.TIMESTAMP}.log",
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("Starting performance test...")
    run_performance_test()
    logging.info("Performance test completed.")