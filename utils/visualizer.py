import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import COLOR_MAP, CLARITY_MAP

def _get_hex_color(grade):
    """Refined colors for glass-like appearance."""
    color_map = {
        'D': '#F0F8FF', 'E': '#F5FAFA', 'F': '#FFFFFF', # Cool Whites
        'G': '#FFFFF0', 'H': '#FFFACD', 'I': '#FFF8DC', 'J': '#FAFAD2'  # Warm Yellows
    }
    return color_map.get(grade, '#FFFFFF')

def _generate_inclusions(clarity_grade, radius_limit):
    """Generates internal flaws (Carbon spots)."""
    severity = {'IF': 0, 'VVS1': 1, 'VVS2': 3, 'VS1': 5, 'VS2': 10, 'SI1': 20, 'SI2': 40, 'I1': 80}
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
    
    rng = np.random.default_rng()
    inc_x = rng.uniform(-radius_limit*0.6, radius_limit*0.6, count)
    inc_y = rng.uniform(-radius_limit*0.6, radius_limit*0.6, count)
    inc_z = rng.uniform(0, radius_limit*0.5, count)
    return inc_x, inc_y, inc_z

def _generate_sparkles(x_verts, y_verts, z_verts, num_sparkles=40):
    """
    Simulates 'Fire' (Dispersion) by placing tiny colored bright spots 
    on random facet vertices.
    """
    if len(x_verts) < 1: return None, None, None, None
    
    rng = np.random.default_rng()
    indices = rng.choice(len(x_verts), num_sparkles, replace=True)
    
    sx = x_verts[indices]
    sy = y_verts[indices]
    sz = z_verts[indices]
    
    # Diamond Fire Colors (Spectral flashes: Blue, Orange, Cyan, Gold)
    # We use bright, neon versions to mimic light hitting the eye
    fire_colors = ['#A9D0F5', '#FFD700', '#FF4500', '#00FFFF', '#FF69B4', '#FFFFFF']
    scolors = rng.choice(fire_colors, num_sparkles)
    
    return sx, sy, sz, scolors

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. Physics & Geometry Math
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65
    
    table_radius = radius * (table_pct / 100)

    # 2. Construct Mesh (8-sided Brilliant Approximation)
    n_segments = 8
    angles = np.linspace(0, 2*np.pi, n_segments + 1)[:-1]
    
    # Rings
    tx = table_radius * np.cos(angles); ty = table_radius * np.sin(angles); tz = np.full_like(tx, z_table)
    gx = radius * np.cos(angles); gy = radius * np.sin(angles); gz = np.full_like(gx, z_girdle)
    
    # Tips
    cx, cy, cz = 0.0, 0.0, z_culet
    ctx, cty, ctz = 0.0, 0.0, z_table

    # Lists for Triangles
    X, Y, Z = [], [], []
    
    def add_tri(p1, p2, p3):
        X.extend([p1[0], p2[0], p3[0]])
        Y.extend([p1[1], p2[1], p3[1]])
        Z.extend([p1[2], p2[2], p3[2]])

    for i in range(n_segments):
        nxt = (i + 1) % n_segments
        # Table Star
        add_tri((ctx, cty, ctz), (tx[i], ty[i], tz[i]), (tx[nxt], ty[nxt], tz[nxt]))
        # Crown
        add_tri((tx[i], ty[i], tz[i]), (gx[i], gy[i], gz[i]), (gx[nxt], gy[nxt], gz[nxt]))
        add_tri((tx[i], ty[i], tz[i]), (gx[nxt], gy[nxt], gz[nxt]), (tx[nxt], ty[nxt], tz[nxt]))
        # Pavilion
        add_tri((gx[i], gy[i], gz[i]), (cx, cy, cz), (gx[nxt], gy[nxt], gz[nxt]))

    # Convert to numpy for easy indexing
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)

    # 3. Build The Figure
    fig = go.Figure()
    base_color = _get_hex_color(color_grade)
    
    # TRACE 1: The Core (Inner Refraction)
    # Smaller, opaque core gives the diamond "substance" so it doesn't look empty
    fig.add_trace(go.Mesh3d(
        x=X*0.9, y=Y*0.9, z=Z*0.9,
        color='white', opacity=0.1, alphahull=0,
        lighting=dict(ambient=0.9, diffuse=0.9, specular=0.0), # Glowy core
        hoverinfo='skip'
    ))

    # TRACE 2: The Surface (High Polish Glass)
    fig.add_trace(go.Mesh3d(
        x=X, y=Y, z=Z,
        color=base_color, opacity=0.2, alphahull=0,
        lighting=dict(
            ambient=0.2, diffuse=0.1, roughness=0.05, 
            specular=2.0, fresnel=5.0 # Max fresnel = Edges glow white
        ),
        flatshading=True, name='Diamond'
    ))
    
    # TRACE 3: The Wireframe (Facet Edges)
    # Explicitly drawing white lines on edges makes it look "Cut"
    wx, wy, wz = [], [], []
    for i in range(n_segments):
        nxt = (i + 1) % n_segments
        # Horizontal Rings
        wx.extend([tx[i], tx[nxt], None]); wy.extend([ty[i], ty[nxt], None]); wz.extend([tz[i], tz[nxt], None])
        wx.extend([gx[i], gx[nxt], None]); wy.extend([gy[i], gy[nxt], None]); wz.extend([gz[i], gz[nxt], None])
        # Vertical Ribs
        wx.extend([tx[i], gx[i], None]); wy.extend([ty[i], gy[i], None]); wz.extend([tz[i], gz[i], None])
        wx.extend([gx[i], cx, None]); wy.extend([gy[i], cy, None]); wz.extend([gz[i], cz, None])

    fig.add_trace(go.Scatter3d(
        x=wx, y=wy, z=wz,
        mode='lines',
        line=dict(color='white', width=3), # Thick white lines = Glinting edges
        opacity=0.3,
        hoverinfo='skip'
    ))

    # TRACE 4: The "Fire" Sparkles (The Hack!)
    # Adds bright colored dots on vertices to mimic dispersion
    sx, sy, sz, scolors = _generate_sparkles(X, Y, Z, num_sparkles=30)
    if sx is not None:
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode='markers',
            marker=dict(size=4, color=scolors, symbol='diamond', opacity=0.9),
            name='Fire (Sparkle)',
            hoverinfo='skip'
        ))

    # TRACE 5: Inclusions (If bad clarity)
    ix, iy, iz = _generate_inclusions(clarity_grade, radius)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode='markers',
            marker=dict(size=3, color='black', opacity=0.6),
            name='Inclusions'
        ))

    # Dark Environment for Contrast
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    return fig



