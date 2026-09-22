# Neuromorphic Event Tracker 🧠⚡

> Real-time asynchronous event stream pre-processing and spatial trajectory estimation using Spiking Neural Networks (snnTorch).

---

## 📌 Project Overview
Event-based vision sensors (DVS) record per-pixel intensity changes asynchronously with microsecond temporal resolution. Standard frame-based Convolutional Neural Networks fail to leverage sparse event distributions. This project implements an end-to-end event processing and tracking pipeline that converts asynchronous event data into continuous time-surface representations, passing them to a Leaky Integrate-and-Fire (LIF) Spiking Neural Network for trajectory estimation.

---

## 🏗️ System Architecture

```text
┌────────────────────────┐
│  Event Camera Stream   │  (x, y, timestamp t, polarity p)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Time-Surface Generator │  Vectorized decay tensor calculation (τ)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Conv-SNN Architecture  │  2-Channel Input, Leaky Integrate-and-Fire
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Spatial Tracking Head  │  Surrogate gradient backpropagation
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Predicted Coordinates │  (X, Y) Trajectory output
└────────────────────────┘