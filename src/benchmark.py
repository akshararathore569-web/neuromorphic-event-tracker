import time
import torch
import numpy as np
from src.models.snn_tracker import SNNTracker
from src.preprocessing.synthetic_generator import SyntheticTargetGenerator
from src.preprocessing.event_surface import EventSurfaceGenerator

def run_benchmark():
    print("--- Running Latency & Accuracy Benchmark ---")
    
    # 1. Measure Latency
    sample_surface = torch.randn(1, 2, 128, 128)
    model = SNNTracker()
    model.eval()

    warmup_runs = 20
    test_runs = 200
    latencies = []

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_surface)
            
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = model(sample_surface)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(f"Inference Latency: {avg_latency:.3f} ms ± {std_latency:.3f} ms per frame")

    # 2. Measure Tracking Accuracy
    generator = SyntheticTargetGenerator()
    events, true_trajectory = generator.generate_ball_trajectory()
    
    surface_gen = EventSurfaceGenerator(sensor_size=(128, 128))
    time_surfaces = surface_gen.generate_time_surface(events)
    
    predictions = []
    with torch.no_grad():
        for i in range(time_surfaces.shape[0]):
            frame = time_surfaces[i:i+1]
            if frame.dim() == 3:
                frame = frame.unsqueeze(1)
            if frame.shape[1] == 1:
                frame = frame.repeat(1, 2, 1, 1)
                
            pred = model(frame)
            # Convert PyTorch tensor prediction to NumPy array
            predictions.append(pred.squeeze(0).detach().cpu().numpy())
            
    predictions = np.array(predictions)
    true_traj_np = np.array(true_trajectory[:len(predictions)])
    
    # Calculate MSE and RMSE using numpy arrays
    mse = np.mean((predictions - true_traj_np) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"Spatial Tracking MSE:  {mse:.4f}")
    print(f"Spatial Tracking RMSE: {rmse:.4f} pixels")
    print("--------------------------------------------")

if __name__ == "__main__":
    run_benchmark()