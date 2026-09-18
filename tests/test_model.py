import torch
from src.models.snn_tracker import SNNTracker

def test_snn_forward():
    model = SNNTracker(beta=0.5, num_steps=10)
    
    # Synthetic time-surface batch: Batch size 2, 2 Channels, 128x128 resolution
    dummy_surface = torch.rand((2, 2, 128, 128))
    
    output_coords = model(dummy_surface)
    
    print(f"Input surface shape:  {dummy_surface.shape}")
    print(f"Predicted coordinates shape: {output_coords.shape}")
    
    assert output_coords.shape == (2, 2)
    print("SNN Forward pass test passed successfully!")

if __name__ == "__main__":
    test_snn_forward()