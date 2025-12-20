import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import COLOR_MAP, CLARITY_MAP

def _get_hex_color(grade):
    """Enhanced color mapping for better visual clarity."""
    color_map = {
        'D': '#E0F0FF', 'E': '#F0F0F0', 'F': '#FFFFFF',
        'G': '#FFFFE0', 'H': '#FFF9C4', 'I': '#FFF59D', 'J': '#FFF176'
    }
    return color_map.get(grade, '#FFFFFF')

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. Geometry Calculations
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table, z_girdle, z_culet = total_depth * 0.35, 0, -total_depth * 0.65

    # 2. Vertices (Increased to 16 sides for a smoother 'gem' look)
    sides = 16
    angles = np.linspace(0, 2*np.pi, sides + 1)[:-1]
    
    tx, ty = table_radius * np.cos(angles), table_radius * np.sin(angles)
    gx, gy = radius * np.cos(angles), radius * np.sin(angles)
    
    x = np.concatenate([[0], tx, gx, [0]])
    y = np.concatenate([[0], ty, gy, [0]])
    z = np.concatenate([[z_table], np.full_like(tx, z_table), np.full_like(gx, z_girdle), [z_culet]])

    fig = go.Figure()
    diamond_color = _get_hex_color(color_grade)

    # --- NEW: MULTI-LAYERED KERNELS (Subsurface effect) ---
    # We create 4 layers: 0.95x, 0.8x, 0.6x, and 0.4x scale.
    # Higher scale = more transparent; Lower scale = more opaque (color core).
    layers = [
        {'scale': 0.96, 'opacity': 0.15},
        {'scale': 0.85, 'opacity': 0.30},
        {'scale': 0.65, 'opacity': 0.50},
        {'scale': 0.40, 'opacity': 0.80}
    ]

    for layer in layers:
        s = layer['scale']
        fig.add_trace(go.Mesh3d(
            x=x * s, y=y * s, z=z * s,
            alphahull=0,
            opacity=layer['opacity'],
            color=diamond_color,
            lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.05),
            flatshading=True,
            hoverinfo='skip'
        ))

    # --- OUTER SHELL: THE HIGH SHINE ---
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        alphahull=0,
        opacity=0.2, # Very transparent surface
        color=diamond_color,
        lighting=dict(
            ambient=0.1, diffuse=0.1, 
            roughness=0.01, specular=4.0, # Massive shine
            fresnel=5.0                   # Bright edges
        ),
        flatshading=True,
        name='Diamond Surface'
    ))

    # --- THIN WIREFRAME: JUST FOR DEFINITION ---
    xl, yl, zl = [], [], []
    for i in range(sides):
        nxt = (i+1)%sides
        # Table, Girdle, and Vertical lines
        for start, end in [((tx[i], ty[i], z_table), (tx[nxt], ty[nxt], z_table)),
                           ((gx[i], gy[i], z_girdle), (gx[nxt], gy[nxt], z_girdle)),
                           ((tx[i], ty[i], z_table), (gx[i], gy[i], z_girdle)),
                           ((gx[i], gy[i], z_girdle), (0, 0, z_culet))]:
            xl.extend([start[0], end[0], None])
            yl.extend([start[1], end[1], None])
            zl.extend([start[2], end[2], None])

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1.5),
        opacity=0.3, # Faded lines for a more realistic look
        hoverinfo='skip'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode='data', bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    return fig


