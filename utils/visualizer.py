import numpy as np
import plotly.graph_objects as go

def _get_diamond_palette(grade):
    """
    Returns a specialized color palette for different layers of the diamond.
    D (Colorless) is icy/blue, while J (Light Yellow) has a deep warm heart.
    Returns: (Surface_Color, Core_Color, Heart_Color)
    """
    palettes = {
        'D': ('#F0F8FF', '#E0F0FF', '#B0D0FF'), # Icy Blue-White
        'E': ('#F8F8FF', '#F0F0F8', '#D0D0E0'), # Crisp White
        'F': ('#FFFFFF', '#F5F5F5', '#E0E0E0'), # Pure White
        'G': ('#FFFFF0', '#FFFEE0', '#E8E8C0'), # Ivory
        'H': ('#FFFACD', '#FFF5B0', '#E5D090'), # Lemon Tint
        'I': ('#FFF8DC', '#F5EBC0', '#D2B48C'), # Cornsilk/Tan
        'J': ('#F0E68C', '#EEDD82', '#DAA520')  # Golden Heart
    }
    return palettes.get(grade, ('#FFFFFF', '#F5F5F5', '#E0E0E0'))

def _generate_inclusions(clarity_grade, radius_limit):
    """Generates internal 'flaws' strictly within the deep heart of the diamond."""
    severity = {
        'IF': 0, 'VVS1': 1, 'VVS2': 3, 
        'VS1': 6, 'VS2': 12, 
        'SI1': 35, 'SI2': 75, 'I1': 130
    }
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
        
    rng = np.random.default_rng()
    # Flaws are concentrated in the center to look like internal carbon/feathers
    inc_x = rng.uniform(-radius_limit*0.5, radius_limit*0.5, count)
    inc_y = rng.uniform(-radius_limit*0.5, radius_limit*0.5, count)
    inc_z = rng.uniform(-radius_limit*0.3, radius_limit*0.4, count) 
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    """
    Generates a realistic 3D diamond visualization using nested kernels.
    Fixed specular values to remain within Plotly's valid range [0, 2].
    """
    # 1. Base Geometry Calculations
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65

    # Points Generation (Standard Round Brilliant Facets)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    tx = table_radius * np.cos(angles); ty = table_radius * np.sin(angles); tz = np.full_like(tx, z_table)
    gx = radius * np.cos(angles); gy = radius * np.sin(angles); gz = np.full_like(gx, z_girdle)
    cx, cy, cz = [0.0], [0.0], [z_culet]
    cent_x, cent_y, cent_z = [0.0], [0.0], [z_table]

    # Base arrays for the 100% scale mesh
    x_base = np.concatenate([cent_x, tx, gx, cx])
    y_base = np.concatenate([cent_y, ty, gy, cy])
    z_base = np.concatenate([cent_z, tz, gz, cz])

    # 2. Figure Initialization
    fig = go.Figure()
    surf_color, core_color, heart_color = _get_diamond_palette(color_grade)

    # --- LAYER 1: THE HEART (Deep Interior, 60% Size) ---
    # High ambient and diffuse to make the internal color glow.
    fig.add_trace(go.Mesh3d(
        x=x_base * 0.6, y=y_base * 0.6, z=z_base * 0.6,
        alphahull=0, opacity=0.8, color=heart_color,
        lighting=dict(ambient=0.8, diffuse=1.0, roughness=0.5, specular=0.0),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 2: THE CORE (Body Substance, 88% Size) ---
    fig.add_trace(go.Mesh3d(
        x=x_base * 0.88, y=y_base * 0.88, z=z_base * 0.88,
        alphahull=0, opacity=0.4, color=core_color,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.2, specular=0.5),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 3: THE SURFACE (Outer Glass, 100% Size) ---
    # specular set to 2.0 (MAX allowed by Plotly) for the shine effect.
    fig.add_trace(go.Mesh3d(
        x=x_base, y=y_base, z=z_base,
        alphahull=0, opacity=0.15, color=surf_color,
        lighting=dict(
            ambient=0.2, 
            diffuse=0.2, 
            roughness=0.01, 
            specular=2.0,   # Fix: Plotly allows [0, 2]
            fresnel=5.0     # Fresnel allows up to 5.0
        ),
        flatshading=True, name='Diamond Surface'
    ))

    # --- LAYER 4: SELECTIVE WIREFRAME ---
    xl, yl, zl = [], [], []
    def add_l(x1, y1, z1, x2, y2, z2):
        xl.extend([x1, x2, None]); yl.extend([y1, y2, None]); zl.extend([z1, z2, None])
    
    for i in range(8):
        nxt = (i+1)%8
        add_l(tx[i], ty[i], tz[i], tx[nxt], ty[nxt], tz[nxt])
        add_l(gx[i], gy[i], gz[i], gx[nxt], gy[nxt], gz[nxt])
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

    # 3. Layout Configuration
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
    
    return fig fig





