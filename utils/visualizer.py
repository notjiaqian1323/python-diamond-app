import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import COLOR_MAP, CLARITY_MAP

def _get_hex_color(grade):
    """Internal helper: Maps color grade to hex."""
    color_hex = {
        'D': '#FFFFFF', 'E': '#F0F8FF', 'F': '#F0FFFF',
        'G': '#F5F5F5', 'H': '#FFFACD', 'I': '#FFF8DC', 'J': '#F0E68C'
    }
    return color_hex.get(grade, '#FFFFFF')

def _generate_inclusions(clarity_grade, radius_limit):
    """Internal helper: Generates inclusion scatter points."""
    severity = {'IF': 0, 'VVS1': 2, 'VVS2': 5, 'VS1': 10, 'VS2': 20, 'SI1': 40, 'SI2': 80, 'I1': 150}
    count = severity.get(clarity_grade, 0)
    
    if count == 0: return None, None, None
        
    rng = np.random.default_rng()
    inc_x = rng.uniform(-radius_limit*0.7, radius_limit*0.7, count)
    inc_y = rng.uniform(-radius_limit*0.7, radius_limit*0.7, count)
    inc_z = rng.uniform(0, radius_limit*0.8, count) 
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    """
    Main function to generate the Plotly Figure.
    """
    # 1. Calculate Geometry
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65

    # 2. Generate Vertices
    angles = np.linspace(0, 2*np.pi, 9)
    tx = table_radius * np.cos(angles); ty = table_radius * np.sin(angles); tz = np.full_like(tx, z_table)
    gx = radius * np.cos(angles); gy = radius * np.sin(angles); gz = np.full_like(gx, z_girdle)
    cx, cy, cz = [0], [0], [z_culet]
    cent_x, cent_y, cent_z = [0], [0], [z_table]

    # Mesh Arrays
    x = np.concatenate([cent_x, tx[:-1], gx[:-1], cx])
    y = np.concatenate([cent_y, ty[:-1], gy[:-1], cy])
    z = np.concatenate([cent_z, tz[:-1], gz[:-1], cz])

    # Wireframe Arrays
    xl, yl, zl = [], [], []
    def add_line(x1, y1, z1, x2, y2, z2):
        xl.extend([x1, x2, None]); yl.extend([y1, y2, None]); zl.extend([z1, z2, None])

    for i in range(8):
        # Add table, girdle, crown, pavilion, star lines...
        # (Simplified loop logic for brevity)
        nxt = (i+1)%8
        add_line(tx[i], ty[i], tz[i], tx[nxt], ty[nxt], tz[nxt]) # Table Ring
        add_line(gx[i], gy[i], gz[i], gx[nxt], gy[nxt], gz[nxt]) # Girdle Ring
        add_line(tx[i], ty[i], tz[i], gx[i], gy[i], gz[i])       # Crown Vertical
        add_line(gx[i], gy[i], gz[i], cx[0], cy[0], cz[0])       # Pavilion
        add_line(cent_x[0], cent_y[0], cent_z[0], tx[i], ty[i], tz[i]) # Star

    # 3. Build Figure
    fig = go.Figure()
    
    # Trace 1: Glass Body
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, alphahull=0, opacity=0.15,
        color=_get_hex_color(color_grade),
        lighting=dict(ambient=0.1, diffuse=0.1, roughness=0.01, specular=1.5, fresnel=4.0),
        flatshading=True
    ))
    
    # Trace 2: Wireframe
    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=2), opacity=0.5
    ))
    
    # Trace 3: Inclusions
    ix, iy, iz = _generate_inclusions(clarity_grade, radius)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode='markers',
            marker=dict(size=2, color='black', opacity=0.8)
        ))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0), showlegend=False
    )
    
    return fig
