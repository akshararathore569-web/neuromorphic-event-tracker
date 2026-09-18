import torch
from src.preprocessing.synthetic_generator import SyntheticTargetGenerator
from src.preprocessing.event_surface import EventSurfaceGenerator

def test_synthetic_stream():
    generator = SyntheticTargetGenerator(sensor_size=(128, 128))
    surface_gen = EventSurfaceGenerator(sensor_size=(128, 128))

    events, target_coords = generator.generate_ball_trajectory()
    surface = surface_gen.generate_time_surface(events)

    print(f"Generated {events.shape[0]} synthetic events.")
    print(f"Time Surface Shape: {surface.shape}")
    print(f"Target Final Coords: {target_coords.tolist()}")

    assert events.shape[1] == 4
    assert surface.shape == (2, 128, 128)
    print("Synthetic Stream Test passed successfully!")

if __name__ == "__main__":
    test_synthetic_stream()