# Imports:
import io
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence

# Title
st.title("Bienvenue sur le plot animé !")

# Loading the data
dataHR_dict = torch.load("../serialized_data/dataHR.pt")
dataHR = dataHR_dict["data"].numpy()

# Sélection des paramètres
num_frames = dataHR.shape[1]  # total number of frames
frames_list = []  # list to hold GIF frames

# Initialize the progress bar
progress_bar = st.progress(0)  # Create the progress bar
current_progress = 0  # Initial progress

# Loop for generating the frames
for i in range(100, num_frames):
    # Update progress
    current_progress = (i - 100) / (num_frames - 100)  # Calculate progress between 0 and 1
    progress_bar.progress(current_progress)  # Update the progress bar

    # Select data to plot
    to_plot = dataHR[3, i, 0, :, :]

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(to_plot, cmap='viridis', aspect='auto', interpolation='bicubic')
    ax.set_title(f"Frame {i}")
    ax.axis("off")  # Remove axes for clean display

    # Save the figure in memory
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    frames_list.append(img)

    plt.close(fig)  # Close the figure to save memory

# Finalize the progress bar
progress_bar.progress(1.0)  # Mark progress as complete

# Save the GIF in memory
gif_buffer = io.BytesIO()
frames_list[0].save(
    gif_buffer, format="GIF", save_all=True, append_images=frames_list[1:], duration=100, loop=0
)
gif_buffer.seek(0)

# Display the GIF in Streamlit
st.image(gif_buffer, caption="Animation des frames", use_container_width=True)

# Download button for the GIF
st.download_button(
    label="Télécharger le GIF",
    data=gif_buffer,
    file_name="animation.gif",
    mime="image/gif"
)
