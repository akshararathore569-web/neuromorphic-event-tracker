import torch
import torch.nn as nn
import torch.optim as optim
from src.models.snn_tracker import SNNTracker
from src.preprocessing.synthetic_generator import SyntheticTargetGenerator
from src.preprocessing.event_surface import EventSurfaceGenerator


def train_snn_tracker(epochs: int = 20, batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training SNNTracker on device: {device}")

    # Initialize model, loss, and optimizer
    model = SNNTracker(beta=0.5, num_steps=10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Generators
    synth_gen = SyntheticTargetGenerator(sensor_size=(128, 128))
    surface_gen = EventSurfaceGenerator(sensor_size=(128, 128))

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for _ in range(batch_size):
            # Generate synthetic batch on the fly
            events, target_coords = synth_gen.generate_ball_trajectory()
            surface = surface_gen.generate_time_surface(events).unsqueeze(0).to(device)
            target = target_coords.unsqueeze(0).to(device)

            # Forward pass
            optimizer.zero_grad()
            pred_coords = model(surface)
            
            loss = criterion(pred_coords, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / batch_size
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Mean Coordinate MSE Loss: {avg_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), "snn_tracker_checkpoint.pt")
    print("Model checkpoint saved to snn_tracker_checkpoint.pt!")


if __name__ == "__main__":
    train_snn_tracker(epochs=100, batch_size=8)