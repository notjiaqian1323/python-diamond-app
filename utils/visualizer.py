import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import COLOR_MAP, CLARITY_MAP

def _get_hex_color(grade):
    """
    Maps color grade to Hex Code.
    Colors are slightly saturated to ensure they are visible in the 3D render.
    """
    color_map = {
        'D': '#F0F8FF', # Alice Blue (Icy Cold White)
        'E': '#F5F5F5', # White Smoke
        'F': '#FFFFFF', # Pure White
        'G': '#FFFFF0', # Ivory (Creamy)
        'H': '#FFFACD', # Lemon Chiffon (Pale Yellow)
        'I': '#FFF8DC', # Cornsilk (Distinct Yellow tint)
        'J': '#F0E68C'  # Khaki (Stronger Yellow)
    }
    return color_map.get(grade, '#FFFFFF')

def _generate_inclusions(clarity_grade, radius_limit):
    """Generates 3D coordinates for 'flaws' inside the diamond."""
    severity = {
        'IF': 0, 'VVS1': 1, 'VVS2': 3, 
        'VS1': 5, 'VS2': 10, 
        'SI1': 30, 'SI2': 60, 'I1': 100
    }
    count = severity.get(clarity_grade, 0)
    
    if count == 0: return None, None, None
        
    rng = np.random.default_rng()
    # Flaws are clustered in the center (body), avoiding the very edges
    inc_x = rng.uniform(-radius_limit*0.6, radius_limit*0.6, count)
    inc_y = rng.uniform(-radius_limit*0.6, radius_limit*0.6, count)
    inc_z = rng.uniform(0, radius_limit*0.5, count) 
    
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    """
    Main function to generate the Realistic 'Solid' Plotly Figure.
    """
    # 1. Calculate Geometry
    # Physics: Diameter roughly scales with cube root of carat
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    
    # Proportions
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65

    # 2. Generate Vertices (8-sided simplification for performance & look)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    
    # Rings
    tx = table_radius * np.cos(angles)
    ty = table_radius * np.sin(angles)
    tz = np.full_like(tx, z_table)
    
    gx = radius * np.cos(angles)
    gy = radius * np.sin(angles)
    gz = np.full_like(gx, z_girdle)
    
    # Tips
    cx, cy, cz = [0], [0], [z_culet]
    cent_x, cent_y, cent_z = [0], [0], [z_table]

    # Combine into Mesh Arrays
    x = np.concatenate([cent_x, tx, gx, cx])
    y = np.concatenate([cent_y, ty, gy, cy])
    z = np.concatenate([cent_z, tz, gz, cz])

    # 3. Build Figure
    fig = go.Figure()
    
    diamond_color = _get_hex_color(color_grade)
    
    # --- TRACE 1: THE INNER CORE (The "Substance") ---
    # We render a slightly smaller (0.9x), more opaque version inside.
    # This simulates light scattering internally and makes it look "solid".
    fig.add_trace(go.Mesh3d(
        x=x * 0.92, y=y * 0.92, z=z * 0.92, 
        alphahull=0, 
        opacity=0.6,          # Higher opacity for the core = Solid look
        color=diamond_color,  # Inner material has the gem color
        lighting=dict(
            ambient=0.6, 
            diffuse=0.9,      # High diffuse to catch light internally
            roughness=0.1, 
            specular=0.1
        ),
        flatshading=True,
        hoverinfo='skip'
    ))

    # --- TRACE 2: THE OUTER SHELL (The "Shine") ---
    # This is the glass surface. High shine, lower opacity.
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, 
        alphahull=0, 
        opacity=0.25,         # Transparent glass
        color=diamond_color,
        lighting=dict(
            ambient=0.1, 
            diffuse=0.1, 
            roughness=0.01,   # Super smooth glass
            specular=2.5,     # High shine reflection
            fresnel=4.5       # Strong rim lighting (glowing edges)
        ),
        flatshading=True,
        name='Diamond'
    ))
    
    # --- TRACE 3: WIREFRAME (The Cut Definition) ---
    xl, yl, zl = [], [], []
    def add_line(x1, y1, z1, x2, y2, z2):
        xl.extend([x1, x2, None]); yl.extend([y1, y2, None]); zl.extend([z1, z2, None])

    for i in range(8):
        nxt = (i+1)%8
        add_line(tx[i], ty[i], tz[i], tx[nxt], ty[nxt], tz[nxt]) # Table Ring
        add_line(gx[i], gy[i], gz[i], gx[nxt], gy[nxt], gz[nxt]) # Girdle Ring
        add_line(tx[i], ty[i], tz[i], gx[i], gy[i], gz[i])       # Crown
        add_line(gx[i], gy[i], gz[i], cx[0], cy[0], cz[0])       # Pavilion
        add_line(cent_x[0], cent_y[0], cent_z[0], tx[i], ty[i], tz[i]) # Star

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, 
        mode='lines',
        line=dict(color='white', width=2), 
        opacity=0.6,
        name='Cut Lines'
    ))
    
    # --- TRACE 4: INCLUSIONS (The Flaws) ---
    ix, iy, iz = _generate_inclusions(clarity_grade, radius)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, 
            mode='markers',
            marker=dict(size=2, color='black', opacity=0.8),
            name='Inclusions'
        ))

    # Layout: Clean and Minimal
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False), 
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    return fig

