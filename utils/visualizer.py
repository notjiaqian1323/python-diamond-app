import numpy as np
import plotly.graph_objects as go

def _get_diamond_palette(grade):
    """
    Returns a specialized color palette for different layers.
    Refined to be more subtle so the light glints stand out more.
    """
    palettes = {
        'D': ('#F0F8FF', '#E0F0FF', '#B0D0FF'), # Icy Blue-White
        'E': ('#F8F8FF', '#F0F0F8', '#D0D0E0'), 
        'F': ('#FFFFFF', '#F5F5F5', '#E0E0E0'), 
        'G': ('#FFFFF0', '#FFFEE0', '#E8E8C0'), 
        'H': ('#FFFACD', '#FFF5B0', '#E5D090'), 
        'I': ('#FFF8DC', '#F5EBC0', '#D2B48C'), 
        'J': ('#F0E68C', '#EEDD82', '#DAA520')  
    }
    return palettes.get(grade, ('#FFFFFF', '#F5F5F5', '#E0E0E0'))

def _generate_inclusions(clarity_grade, radius_limit):
    """Generates internal flaws (only for lower clarity)."""
    severity = {'IF': 0, 'VVS1': 1, 'VVS2': 3, 'VS1': 6, 'VS2': 12, 'SI1': 35, 'SI2': 75, 'I1': 130}
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
    rng = np.random.default_rng()
    inc_x = rng.uniform(-radius_limit*0.5, radius_limit*0.5, count)
    inc_y = rng.uniform(-radius_limit*0.5, radius_limit*0.5, count)
    inc_z = rng.uniform(-radius_limit*0.3, radius_limit*0.4, count) 
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. High-Detail Geometry (32 segments for maximum light-catching surfaces)
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_girdle = 0 
    z_culet = -total_depth * 0.65

    # 32 Segments makes the 'bright spots' much smaller and more numerous
    n = 32 
    angles = np.linspace(0, 2*np.pi, n+1)[:-1]
    tx, ty, tz = table_radius * np.cos(angles), table_radius * np.sin(angles), np.full(n, z_table)
    gx, gy, gz = radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)
    
    # Mesh vertices
    x_base = np.concatenate([[0], tx, gx, [0]])
    y_base = np.concatenate([[0], ty, gy, [0]])
    z_base = np.concatenate([[z_table], tz, gz, [z_culet]])

    fig = go.Figure()
    surf_c, core_c, heart_c = _get_diamond_palette(color_grade)

    # --- LAYER 1: THE HEART (Deep Internal Glow) ---
    fig.add_trace(go.Mesh3d(
        x=x_base*0.6, y=y_base*0.6, z=z_base*0.6, alphahull=0, opacity=0.8, color=heart_c,
        lighting=dict(ambient=0.8, diffuse=1.0, roughness=0.5, specular=0.0),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 2: THE CORE (Substance) ---
    fig.add_trace(go.Mesh3d(
        x=x_base*0.88, y=y_base*0.88, z=z_base*0.88, alphahull=0, opacity=0.4, color=core_c,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.2, specular=0.5),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 3: THE SHIMMER SHELL (Primary Reflection) ---
    # High specular and low roughness creates that 'bright thing' you like.
    fig.add_trace(go.Mesh3d(
        x=x_base, y=y_base, z=z_base, alphahull=0, opacity=0.15, color=surf_c,
        lighting=dict(ambient=0.2, diffuse=0.2, roughness=0.01, specular=2.0, fresnel=5.0),
        flatshading=True, name='Diamond Surface'
    ))

    # --- LAYER 4: THE SECONDARY REFLECTOR (Double Shine) ---
    # By rotating the base mesh by just 1 degree, we create a second set of glints
    # that appear in different positions, making the diamond look 'alive'.
    rot = np.radians(1.0)
    x_rot = x_base * np.cos(rot) - y_base * np.sin(rot)
    y_rot = x_base * np.sin(rot) + y_base * np.cos(rot)
    
    fig.add_trace(go.Mesh3d(
        x=x_rot, y=y_rot, z=z_base, alphahull=0, opacity=0.1, color='#FFFFFF',
        lighting=dict(ambient=0.1, diffuse=0.1, roughness=0.01, specular=2.0, fresnel=5.0),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 5: SELECTIVE WIREFRAME (Facet Definition) ---
    # Subtle lines to define the 'Cut' without looking like a cage.
    xl, yl, zl = [], [], []
    for i in range(n):
        nxt = (i+1)%n
        if i % 4 == 0: # Only draw every 4th rib to keep it elegant
            xl += [tx[i], tx[nxt], None, gx[i], gx[nxt], None, tx[i], gx[i], None, gx[i], 0, None]
            yl += [ty[i], ty[nxt], None, gy[i], gy[nxt], None, ty[i], gy[i], None, gy[i], 0, None]
            zl += [tz[i], tz[nxt], None, gz[i], gz[nxt], None, tz[i], gz[i], None, gz[i], z_culet, None]

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1.0), opacity=0.2, hoverinfo='skip'
    ))

    # --- LAYER 6: INCLUSIONS (The Flaws) ---
    ix, iy, iz = _generate_inclusions(clarity_grade, radius)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode='markers',
            marker=dict(size=2.5, color='black', opacity=0.6),
            name='Internal Flaws'
        ))

    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0), showlegend=False
    )
    
    return fig








