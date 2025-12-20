import numpy as np
import plotly.graph_objects as go

def _get_diamond_palette(grade):
    """
    Returns a specialized color palette for different layers.
    D (Colorless) is icy/blue, while J (Light Yellow) has a deep warm heart.
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

def _generate_sparkles(x, y, z, count=25):
    """Generates 'Fire' (dispersion) points to mimic rainbow sparkles."""
    rng = np.random.default_rng()
    indices = rng.choice(len(x), min(count, len(x)), replace=False)
    # Spectral colors: Blue, Gold, Pink, White
    fire_colors = ['#00B4D8', '#FFD700', '#FF00FF', '#FFFFFF']
    return x[indices], y[indices], z[indices], rng.choice(fire_colors, count)

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. High-Detail Geometry (16 sides instead of 8 for 'Brilliant' look)
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table, z_girdle, z_culet = total_depth * 0.35, 0, -total_depth * 0.65

    # 16-Segment Brilliant Cut approximation
    n = 16 
    angles = np.linspace(0, 2*np.pi, n+1)[:-1]
    tx, ty, tz = table_radius * np.cos(angles), table_radius * np.sin(angles), np.full(n, z_table)
    gx, gy, gz = radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)
    
    # Vertices for the mesh
    x_base = np.concatenate([[0], tx, gx, [0]])
    y_base = np.concatenate([[0], ty, gy, [0]])
    z_base = np.concatenate([[z_table], tz, gz, [z_culet]])

    fig = go.Figure()
    surf_c, core_c, heart_c = _get_diamond_palette(color_grade)

    # --- LAYER 1 & 2: THE KERNELS (Substance) ---
    for scale, op, col in [(0.6, 0.8, heart_c), (0.88, 0.4, core_c)]:
        fig.add_trace(go.Mesh3d(
            x=x_base*scale, y=y_base*scale, z=z_base*scale, alphahull=0, opacity=op, color=col,
            lighting=dict(ambient=0.7, diffuse=0.9, roughness=0.3, specular=0.1),
            flatshading=True, hoverinfo='skip'
        ))

    # --- LAYER 3: THE OUTER SHELL (Reflection) ---
    fig.add_trace(go.Mesh3d(
        x=x_base, y=y_base, z=z_base, alphahull=0, opacity=0.15, color=surf_c,
        lighting=dict(ambient=0.2, diffuse=0.1, roughness=0.01, specular=2.0, fresnel=5.0),
        flatshading=True, name='Diamond'
    ))

    # --- LAYER 4: SPARKLES (The Fire) ---
    sx, sy, sz, sc = _generate_sparkles(x_base, y_base, z_base)
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz, mode='markers',
        marker=dict(size=4, color=sc, symbol='diamond', opacity=0.9),
        name='Fire (Dispersion)'
    ))

    # --- LAYER 5: WIREFRAME (The Cut) ---
    xl, yl, zl = [], [], []
    for i in range(n):
        nxt = (i+1)%n
        xl += [tx[i], tx[nxt], None, gx[i], gx[nxt], None, tx[i], gx[i], None, gx[i], 0, None]
        yl += [ty[i], ty[nxt], None, gy[i], gy[nxt], None, ty[i], gy[i], None, gy[i], 0, None]
        zl += [tz[i], tz[nxt], None, gz[i], gz[nxt], None, tz[i], gz[i], None, gz[i], z_culet, None]

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1.5), opacity=0.3, hoverinfo='skip'
    ))

    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0), showlegend=False
    )
    return fig







