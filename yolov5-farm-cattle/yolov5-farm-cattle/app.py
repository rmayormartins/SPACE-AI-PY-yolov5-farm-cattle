import gradio as gr
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestyolo5.pt')

def detect(img):
    img_arr = np.array(img)
    results = model(img_arr)

    fig, ax = plt.subplots()
    ax.imshow(img_arr)

    cattle_count = 0
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, xyxy)
        label = model.names[int(cls)]
        if label == 'cattle':
            cattle_count += 1
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
        ax.text(x1, y1, f'{label} {conf:.2f}', color='white', fontsize=8, bbox={'facecolor': 'red', 'alpha': 0.5})

    plt.axis('off')

    fig.canvas.draw()
    pil_img = Image.fromarray(np.array(fig.canvas.renderer._renderer))
    plt.close(fig)

    return pil_img, cattle_count

# Gradio
iface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox(label="Number of Cattle Detected")],
    title="YOLOv5 Cattle Counter",
    description="Object detector trained to count cattle using YOLOv5.",
    examples=[["example1.jpg"]]
)

if __name__ == "__main__":
    iface.launch()
