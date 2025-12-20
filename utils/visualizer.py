import numpy as np
import plotly.graph_objects as go

def _get_hex_color_layers(grade):
    """
    Returns a palette for the diamond based on grade.
    Each grade returns (Surface Color, Core Color, Heart Color).
    """
    palettes = {
        'D': ('#F0F8FF', '#E0F0FF', '#B0D0FF'), # Icy Blue-White
        'E': ('#F8F8FF', '#F0F0F8', '#D0D0E0'), # Ghost White
        'F': ('#FFFFFF', '#F5F5F5', '#E0E0E0'), # Pure White
        'G': ('#FFFFF0', '#FFFEE0', '#E8E8C0'), # Ivory
        'H': ('#FFFACD', '#FFF5B0', '#E5D090'), # Lemon
        'I': '#FFF8DC', # Cornsilk
        'J': '#F0E68C'  # Khaki
    }
    
    # If it's a single string, derive shades
    color = palettes.get(grade, ('#FFFFFF', '#F5F5F5', '#E0E0E0'))
    if isinstance(color, str):
        # Fallback logic to create shades for single-hex colors
        return (color, color, color) # Simplified for custom logic below
    return color

def _generate_inclusions(clarity_grade, radius_limit):
    """Generates 3D coordinates for 'flaws' inside the heart of the diamond."""
    severity = {
        'IF': 0, 'VVS1': 1, 'VVS2': 3, 
        'VS1': 6, 'VS2': 12, 
        'SI1': 35, 'SI2': 70, 'I1': 120
    }
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
        
    rng = np.random.default_rng()
    # Flaws are strictly in the deep heart of the diamond
    inc_x = rng.uniform(-radius_limit*0.5, radius_limit*0.5, count)
    inc_y = rng.uniform(-radius_limit*0.5, radius_limit*0.5, count)
    inc_z = rng.uniform(-radius_limit*0.2, radius_limit*0.4, count) 
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. Geometry Math
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65

    # Points Generation
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    tx = table_radius * np.cos(angles); ty = table_radius * np.sin(angles); tz = np.full_like(tx, z_table)
    gx = radius * np.cos(angles); gy = radius * np.sin(angles); gz = np.full_like(gx, z_girdle)
    cx, cy, cz = [0], [0], [z_culet]
    cent_x, cent_y, cent_z = [0], [0], [z_table]

    # Combined Mesh Arrays
    x_base = np.concatenate([cent_x, tx, gx, cx])
    y_base = np.concatenate([cent_y, ty, gy, cy])
    z_base = np.concatenate([cent_z, tz, gz, cz])

    # 2. Setup Plot
    fig = go.Figure()
    surf_c, core_c, heart_c = _get_hex_color_layers(color_grade)

    # --- LAYER 1: THE HEART (Deep Interior) ---
    # Most opaque, smallest, darkest color.
    fig.add_trace(go.Mesh3d(
        x=x_base * 0.6, y=y_base * 0.6, z=z_base * 0.6,
        alphahull=0, opacity=0.8, color=heart_c,
        lighting=dict(ambient=0.8, diffuse=1.0, roughness=0.5, specular=0.0),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 2: THE CORE (Medium Interior) ---
    # Mid-opacity, creates the "body" of the gemstone.
    fig.add_trace(go.Mesh3d(
        x=x_base * 0.85, y=y_base * 0.85, z=z_base * 0.85,
        alphahull=0, opacity=0.4, color=core_c,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.2, specular=0.2),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 3: THE SURFACE (High Shine Glass) ---
    # Lowest opacity, highest shine, handles reflections.
    fig.add_trace(go.Mesh3d(
        x=x_base, y=y_base, z=z_base,
        alphahull=0, opacity=0.15, color=surf_c,
        lighting=dict(ambient=0.1, diffuse=0.1, roughness=0.01, specular=3.0, fresnel=5.0),
        flatshading=True, name='Diamond Surface'
    ))

    # --- LAYER 4: SELECTIVE WIREFRAME (Glints) ---
    # We reduce the wireframe to just the table and girdle to keep it "solid" looking
    xl, yl, zl = [], [], []
    def add_l(x1, y1, z1, x2, y2, z2):
        xl.extend([x1, x2, None]); yl.extend([y1, y2, None]); zl.extend([z1, z2, None])
    
    for i in range(8):
        nxt = (i+1)%8
        add_l(tx[i], ty[i], tz[i], tx[nxt], ty[nxt], tz[nxt]) # Table Ring
        add_l(gx[i], gy[i], gz[i], gx[nxt], gy[nxt], gz[nxt]) # Girdle Ring
        # Only draw four vertical ribs instead of eight to reduce "cage" effect
        if i % 2 == 0:
            add_l(tx[i], ty[i], tz[i], gx[i], gy[i], gz[i])
            add_l(gx[i], gy[i], gz[i], cx[0], cy[0], cz[0])

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1.5), opacity=0.3,
        hoverinfo='skip'
    ))

    # --- LAYER 5: INCLUSIONS ---
    ix, iy, iz = _generate_inclusions(clarity_grade, radius)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode='markers',
            marker=dict(size=2.5, color='black', opacity=0.7),
            name='Internal Flaws'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False), 
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    return fig

