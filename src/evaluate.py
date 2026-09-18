import torch
import matplotlib.pyplot as plt
from src.models.snn_tracker import SNNTracker
from src.preprocessing.synthetic_generator import SyntheticTargetGenerator
from src.preprocessing.event_surface import EventSurfaceGenerator


def evaluate_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and saved weights
    model = SNNTracker(beta=0.5, num_steps=10).to(device)
    model.load_state_dict(torch.load("snn_tracker_checkpoint.pt", map_location=device))
    model.eval()

    # Generate test trajectory
    synth_gen = SyntheticTargetGenerator(sensor_size=(128, 128))
    surface_gen = EventSurfaceGenerator(sensor_size=(128, 128))

    events, target_coords = synth_gen.generate_ball_trajectory()
    surface = surface_gen.generate_time_surface(events).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_coords = model(surface).squeeze(0).cpu().numpy()

    true_coords = target_coords.numpy()

    print(f"Ground Truth Coordinates: [{true_coords[0]:.2f}, {true_coords[1]:.2f}]")
    print(f"Predicted Coordinates:    [{pred_coords[0]:.2f}, {pred_coords[1]:.2f}]")

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(surface[0, 0].cpu().numpy(), cmap='magma')
    plt.plot(true_coords[0], true_coords[1], 'go', markersize=12, label='Ground Truth')
    plt.plot(pred_coords[0], pred_coords[1], 'rx', markersize=12, markeredgewidth=3, label='SNN Prediction')
    plt.title("Neuromorphic Event Tracker - Coordinate Prediction")
    plt.legend()
    plt.savefig("tracking_result.png")
    plt.show()
    print("Saved visualization plot to tracking_result.png!")


if __name__ == "__main__":
    evaluate_and_plot()