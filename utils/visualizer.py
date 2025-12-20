import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import COLOR_MAP, CLARITY_MAP

def _get_hex_color(grade):
    """Refined colors for glass-like appearance."""
    color_map = {
        'D': '#F0F8FF', # Alice Blue (Icy)
        'E': '#F5FAFA',
        'F': '#FFFFFF', # Pure White
        'G': '#FFFFF0', # Ivory
        'H': '#FFFACD', # Lemon Chiffon
        'I': '#FFF8DC', # Cornsilk
        'J': '#FAFAD2'  # Light Goldenrod
    }
    return color_map.get(grade, '#FFFFFF')

def _generate_inclusions(clarity_grade, radius_limit):
    """Generates internal flaws."""
    severity = {'IF': 0, 'VVS1': 1, 'VVS2': 3, 'VS1': 5, 'VS2': 10, 'SI1': 20, 'SI2': 40, 'I1': 80}
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
    
    rng = np.random.default_rng()
    # Flaws are usually black carbon spots or white feathers. We use black for visibility.
    inc_x = rng.uniform(-radius_limit*0.6, radius_limit*0.6, count)
    inc_y = rng.uniform(-radius_limit*0.6, radius_limit*0.6, count)
    inc_z = rng.uniform(0, radius_limit*0.5, count) # Mostly in the body
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. Physics Calculations
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    
    # Standard Brilliant Proportions (approximate for visual)
    crown_height = total_depth * 0.15
    pavilion_depth = total_depth * 0.43
    girdle_thickness = total_depth * 0.02
    
    table_radius = radius * (table_pct / 100)
    
    # Z-Coordinates (Top to Bottom)
    z_table = crown_height + girdle_thickness/2
    z_girdle_top = girdle_thickness/2
    z_girdle_bot = -girdle_thickness/2
    z_culet = -pavilion_depth - girdle_thickness/2

    # 2. Vertex Generation (8-sided symmetry for cleaner mesh than 16 or 32)
    # We use 8 segments to keep the mesh sharp and fast.
    n_segments = 8
    angles = np.linspace(0, 2*np.pi, n_segments + 1)[:-1] # Remove last duplicate
    
    # Helper to rotate points for "Star Facet" effect
    offset_angles = angles + (np.pi / n_segments)

    # Ring 1: Table (Top Flat)
    tx = table_radius * np.cos(angles)
    ty = table_radius * np.sin(angles)
    tz = np.full_like(tx, z_table)
    
    # Ring 2: Girdle Top
    gux = radius * np.cos(angles)
    guy = radius * np.sin(angles)
    guz = np.full_like(gux, z_girdle_top)

    # Ring 3: Girdle Bottom
    glx = radius * np.cos(angles)
    gly = radius * np.sin(angles)
    glz = np.full_like(glx, z_girdle_bot)
    
    # Point: Culet
    cx, cy, cz = 0.0, 0.0, z_culet
    
    # Point: Table Center
    ctx, cty, ctz = 0.0, 0.0, z_table

    # 3. Constructing Triangles (Manually for perfect facets)
    # Lists to store triangle vertices
    X, Y, Z = [], [], []
    
    def add_tri(p1, p2, p3):
        X.extend([p1[0], p2[0], p3[0]])
        Y.extend([p1[1], p2[1], p3[1]])
        Z.extend([p1[2], p2[2], p3[2]])

    for i in range(n_segments):
        nxt = (i + 1) % n_segments
        
        # A. Table Facet (Star)
        # Center -> Table[i] -> Table[next]
        add_tri((ctx, cty, ctz), (tx[i], ty[i], tz[i]), (tx[nxt], ty[nxt], tz[nxt]))
        
        # B. Crown Main Facets (Quadrilaterals split into 2 tris)
        # Table[i] -> GirdleTop[i] -> GirdleTop[next]
        add_tri((tx[i], ty[i], tz[i]), (gux[i], guy[i], guz[i]), (gux[nxt], guy[nxt], guz[nxt]))
        # Table[i] -> GirdleTop[next] -> Table[next]
        add_tri((tx[i], ty[i], tz[i]), (gux[nxt], guy[nxt], guz[nxt]), (tx[nxt], ty[nxt], tz[nxt]))
        
        # C. Girdle Facets (Rectangles split into 2 tris)
        add_tri((gux[i], guy[i], guz[i]), (glx[i], gly[i], glz[i]), (glx[nxt], gly[nxt], glz[nxt]))
        add_tri((gux[i], guy[i], guz[i]), (glx[nxt], gly[nxt], glz[nxt]), (gux[nxt], guy[nxt], guz[nxt]))
        
        # D. Pavilion Main Facets (Triangles to Culet)
        add_tri((glx[i], gly[i], glz[i]), (cx, cy, cz), (glx[nxt], gly[nxt], glz[nxt]))

    # 4. Creating the Figure
    fig = go.Figure()
    
    base_color = _get_hex_color(color_grade)
    
    # TRACE 1: The "Inner Light" (Simulation of refraction)
    # We plot the same mesh but smaller and opaque to give it "body"
    fig.add_trace(go.Mesh3d(
        x=[val * 0.9 for val in X], 
        y=[val * 0.9 for val in Y], 
        z=[val * 0.9 for val in Z],
        color='white',
        opacity=0.1,
        alphahull=0,
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
        hoverinfo='skip'
    ))

    # TRACE 2: The "Surface Glass" (High reflection)
    fig.add_trace(go.Mesh3d(
        x=X, y=Y, z=Z,
        color=base_color,
        opacity=0.25, # Transparent glass
        alphahull=0,
        lighting=dict(
            ambient=0.1,
            diffuse=0.1,
            roughness=0.01, # Super smooth
            specular=2.0,   # Maximum shine
            fresnel=5.0     # Rim lighting (critical for gem look)
        ),
        flatshading=True,   # Shows individual facets explicitly
        name='Diamond'
    ))
    
    # TRACE 3: Wireframe (Essential for "Cut" visibility)
    # Extract unique edges for cleaner lines
    # (Simplified: just connecting main rings)
    wx, wy, wz = [], [], []
    for i in range(n_segments):
        nxt = (i + 1) % n_segments
        # Table Ring
        wx.extend([tx[i], tx[nxt], None])
        wy.extend([ty[i], ty[nxt], None])
        wz.extend([tz[i], tz[nxt], None])
        # Girdle Top Ring
        wx.extend([gux[i], gux[nxt], None])
        wy.extend([guy[i], guy[nxt], None])
        wz.extend([guz[i], guz[nxt], None])
        # Crown Ribs
        wx.extend([tx[i], gux[i], None])
        wy.extend([ty[i], guy[i], None])
        wz.extend([tz[i], guz[i], None])
        # Pavilion Ribs
        wx.extend([glx[i], cx, None])
        wy.extend([gly[i], cy, None])
        wz.extend([glz[i], cz, None])

    fig.add_trace(go.Scatter3d(
        x=wx, y=wy, z=wz,
        mode='lines',
        line=dict(color='white', width=1.5),
        opacity=0.4,
        hoverinfo='skip'
    ))

    # TRACE 4: Inclusions
    ix, iy, iz = _generate_inclusions(clarity_grade, radius)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz,
            mode='markers',
            marker=dict(size=2, color='black', opacity=0.7),
            name='Inclusions'
        ))

    # Layout: Black background for contrast
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

