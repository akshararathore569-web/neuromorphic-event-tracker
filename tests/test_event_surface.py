import torch
import numpy as np
from src.preprocessing.event_surface import EventSurfaceGenerator

def test_pipeline():
    generator = EventSurfaceGenerator(sensor_size=(128, 128), tau=10000.0)

    np_events = []
    for i in range(1000):
        x = np.random.randint(0, 128)
        y = np.random.randint(0, 128)
        t = float(i * 50)
        p = 1 if np.random.rand() > 0.5 else -1
        np_events.append([x, y, t, p])

    events_tensor = torch.tensor(np_events, dtype=torch.float32)
    surface = generator.generate_time_surface(events_tensor)
    
    print(f"Input events shape:   {events_tensor.shape}")
    print(f"Surface output shape: {surface.shape}")
    print(f"Surface Value Range:  [{surface.min().item():.3f}, {surface.max().item():.3f}]")
    
    assert surface.shape == (2, 128, 128)
    assert surface.max() <= 1.0 and surface.min() >= 0.0
    print("Test passed successfully!")

if __name__ == "__main__":
    test_pipeline()