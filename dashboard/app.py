# Imports : 
import io
import streamlit as st
import torch
import matplotlib.pyplot as plt

# Title
st.title("Bienvenue sur le plot !")

# Intro text
st.write("Ceci est le plot de notre projet.")

# Interaction exemple
name = st.text_input("Quel est votre nom ?")
if name:
    st.write(f"Bonjour, {name} !")

# Loading the data
dataHR_dict = torch.load("../serialized_data/dataHR.pt")
dataHR = dataHR_dict["data"].numpy()

# "video" avec slider 
num_frames = dataHR.shape[1]
frame = st.slider("Temps", min_value=1, max_value=num_frames , value=1)
to_plot = dataHR[3, frame, 0, :, : ]

# Heatmap colored with Matplotlib
fig, ax = plt.subplots(figsize=(10, 4))  # Adjustement of the size (width, height)
# cax = ax.imshow(to_plot, cmap='plasma', aspect='auto')  # 'inferno' is a colored pallete
cax = ax.imshow(to_plot, cmap='viridis', aspect='auto', interpolation='bicubic')
fig.colorbar(cax, ax=ax)  # color bar
ax.set_title("Heatmap of the data")  # graphic title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")







# Affichage dans Streamlit
st.write("Heatmap of the data")
st.pyplot(fig)



