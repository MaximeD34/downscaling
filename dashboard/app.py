# Imports:
import io
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# Title
st.title("Bienvenue sur le plot animé interactif !")

# Loading the data
dataHR_dict = torch.load("../serialized_data/dataHR.pt")
dataHR = dataHR_dict["data"].numpy()

# Initialisation de l'état de session
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 100
if "playing" not in st.session_state:
    st.session_state.playing = False
if "frames_list" not in st.session_state:
    st.session_state.frames_list = []

# Génération des frames si elles n'existent pas
if not st.session_state.frames_list:
    st.write("Génération des frames, patientez...")
    progress_bar = st.progress(0)
    try:
        frames_list = []
        num_frames = dataHR.shape[1]
        for i in range(100, 400):
            to_plot = dataHR[3, i, 0, :, :]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(to_plot, cmap='viridis', aspect='auto', interpolation='bicubic')
            ax.set_title(f"Frame {i + 1}")
            ax.axis("off")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames_list.append(img)

            progress_bar.progress(round((i - 100) / (400 - 100) * 100))
            plt.close(fig)

        st.session_state.frames_list = frames_list
        st.success("Génération des frames terminée !")
    except Exception as e:
        st.error(f"Erreur lors de la génération des frames : {e}")
        st.stop()

# Placeholder pour le slider et l'image
image_placeholder = st.empty()
slider_placeholder = st.empty()

# Affichage interactif des frames
slider_value = slider_placeholder.slider(
    "Naviguez dans la vidéo :",
    101,
    100 + len(st.session_state.frames_list),
    st.session_state.current_frame + 1,
)
st.session_state.current_frame = slider_value - 1
image_placeholder.image(
    st.session_state.frames_list[st.session_state.current_frame - 100],
    caption=f"Frame {st.session_state.current_frame - 98}",
    use_container_width=True,
)

# Boutons Play/Pause/Restart
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶ Play"):
        st.session_state.playing = True
with col2:
    if st.button("⏸ Pause"):
        st.session_state.playing = False
with col3:
    if st.button("🔄 Restart"):
        st.session_state.playing = False
        st.session_state.current_frame = 100
        st.rerun()

# Animation automatique
if st.session_state.playing:
    while st.session_state.current_frame < 100 + len(st.session_state.frames_list) - 1:
        if not st.session_state.playing:
            break
        st.session_state.current_frame += 1
        image_placeholder.image(
            st.session_state.frames_list[st.session_state.current_frame - 100],
            caption=f"Frame {st.session_state.current_frame - 99}",
            use_container_width=True,
        )
        time.sleep(0.05)
    st.session_state.playing = False
