import numpy as np

def show_segmentation_pred(pred, label_map):
    import plotly.express as px
    # example: label_map = {0: "background", 1: "cat", 2: "dog"}  # your mapping

    # Map the mask values to label names
    label_mask = np.vectorize(label_map.get)(pred)

    fig = px.imshow(pred, color_continuous_scale="Viridis")
    fig.update_traces(
        customdata=label_mask,
        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z}<br>label: %{customdata}<extra></extra>"
    )
    fig.show()