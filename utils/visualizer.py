import numpy as np
import plotly.graph_objects as go

def _get_diamond_palette(grade):
    """
    Maps color grade to a 3-layer palette. 
    Interior layers are more saturated to make color differences visible.
    """
    palettes = {
        'D': ('#F0F8FF', '#E0F0FF', '#B0D0FF'), # Icy Blue-White
        'E': ('#F8F8FF', '#F0F0F8', '#D0D0E0'), 
        'F': ('#FFFFFF', '#F5F5F5', '#E0E0E0'), 
        'G': ('#FFFFF0', '#FFFEE0', '#E8E8C0'), # Ivory
        'H': ('#FFFACD', '#FFF5B0', '#E5D090'), # Lemon
        'I': ('#FFF8DC', '#F5EBC0', '#D2B48C'), # Cornsilk
        'J': ('#F0E68C', '#EEDD82', '#DAA520')  # Golden Heart
    }
    res = palettes.get(grade, ('#FFFFFF', '#F5F5F5', '#E0E0E0'))
    return res if isinstance(res, tuple) else (res, res, res)

def _get_cut_lighting(cut_grade):
    """
    Simulation of Light Performance based on Cut Quality.
    Returns: (Specular, Fresnel, Roughness, Interior_Glow)
    Note: Specular is capped at 2.0 per Plotly constraints.
    """
    configs = {
        'Ideal':     (2.0, 5.0, 0.01, 1.0), # Max brilliance
        'Premium':   (1.7, 4.5, 0.03, 0.8),
        'Very Good': (1.3, 3.5, 0.06, 0.6),
        'Good':      (0.9, 2.5, 0.12, 0.4),
        'Fair':      (0.4, 1.2, 0.25, 0.2)  # Dull, leaky appearance
    }
    return configs.get(cut_grade, (1.3, 3.5, 0.06, 0.6))

def _generate_inclusions(clarity_grade, radius, table_radius, z_table, z_culet):
    """
    Geometry-Aware Inclusion Engine.
    Mathematically clips flaws to stay inside the diamond facets.
    """
    severity = {
        'IF': 0, 'VVS1': 2, 'VVS2': 6, 
        'VS1': 16, 'VS2': 40, 
        'SI1': 90, 'SI2': 180, 'I1': 380
    }
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
        
    rng = np.random.default_rng()
    ix, iy, iz = [], [], []
    
    for _ in range(count):
        # Pick a height
        z = rng.uniform(z_culet, z_table)
        # Calculate max allowed radius at this height (clipping)
        if z >= 0:
            # Crown slope
            max_r = radius + (table_radius - radius) * (z / z_table)
        else:
            # Pavilion slope
            max_r = radius * (1 - (z / z_culet))
            
        r = np.sqrt(rng.uniform(0, 1)) * (max_r * 0.96) # 4% safety margin
        theta = rng.uniform(0, 2 * np.pi)
        
        ix.append(r * np.cos(theta))
        iy.append(r * np.sin(theta))
        iz.append(z)
        
    return ix, iy, iz

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade, cut_grade):
    """
    Main entry point for generating the 3D Diamond Model.
    """
    # 1. Physics & Proportions
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_crown_mid = z_table * 0.5 
    z_girdle = 0
    z_culet = -total_depth * 0.65

    # 2. Vertex Definition (8-fold symmetry for True Brilliant)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    half_angles = angles + (np.pi / 8)
    
    v_table_center = [0, 0, z_table]
    v_table = [[table_radius * np.cos(a), table_radius * np.sin(a), z_table] for a in angles]
    v_crown_mid = [[radius * 0.88 * np.cos(a), radius * 0.88 * np.sin(a), z_crown_mid] for a in half_angles]
    v_girdle = [[radius * np.cos(a), radius * np.sin(a), z_girdle] for a in angles]
    v_culet_pt = [0, 0, z_culet]

    vertices = [v_table_center] + v_table + v_crown_mid + v_girdle + [v_culet_pt]
    verts_np = np.array(vertices)
    vx, vy, vz = verts_np[:,0], verts_np[:,1], verts_np[:,2]

    # 3. Manual Triangle Indices (i, j, k)
    i, j, k = [], [], []
    for idx in range(8):
        nxt = (idx + 1) % 8
        # Table (Index 0 is center)
        i.append(0); j.append(idx + 1); k.append(nxt + 1)
        # Crown Star Facets
        i.append(idx + 1); j.append(idx + 9); k.append(nxt + 1)
        # Bezel Facets
        i.append(idx + 9); j.append(idx + 17); k.append(nxt + 17)
        i.append(idx + 9); j.append(nxt + 17); k.append(nxt + 1)
        # Pavilion Facets (Index 25 is culet)
        i.append(idx + 17); j.append(25); k.append(nxt + 17)

    # 4. Apply Lighting Engine
    spec, fres, rough, glow = _get_cut_lighting(cut_grade)
    surf_c, core_c, heart_c = _get_diamond_palette(color_grade)

    fig = go.Figure()

    # --- LAYER 1: THE HEART (Deep Substance) ---
    fig.add_trace(go.Mesh3d(
        x=vx*0.6, y=vy*0.6, z=vz*0.6, i=i, j=j, k=k,
        opacity=0.8 * glow, color=heart_c,
        lighting=dict(ambient=0.7 * glow, diffuse=0.9 * glow, roughness=0.5, specular=0.0),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 2: THE CORE (Body) ---
    fig.add_trace(go.Mesh3d(
        x=vx*0.88, y=vy*0.88, z=vz*0.88, i=i, j=j, k=k,
        opacity=0.4 * glow, color=core_c,
        lighting=dict(ambient=0.4 * glow, diffuse=0.7 * glow, roughness=rough, specular=0.4),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 3: THE BRILLIANT SHELL (Surface) ---
    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz, i=i, j=j, k=k,
        opacity=0.2, color=surf_c,
        lighting=dict(ambient=0.2, diffuse=0.1, roughness=rough, specular=spec, fresnel=fres),
        flatshading=True, name='Facet Surface'
    ))

    # --- LAYER 4: INCLUSIONS (The Flaws) ---
    ix, iy, iz = _generate_inclusions(clarity_grade, radius, table_radius, z_table, z_culet)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode='markers',
            marker=dict(size=3, color='rgba(50, 50, 50, 0.7)', symbol='circle'),
            name='Internal Inclusions'
        ))

    # --- LAYER 5: FACET OUTLINES ---
    xl, yl, zl = [], [], []
    for tidx in range(len(i)):
        pts = [i[tidx], j[tidx], k[tidx], i[tidx]]
        for p in range(3):
            p1, p2 = vertices[pts[p]], vertices[pts[p+1]]
            xl += [p1[0], p2[0], None]; yl += [p1[1], p2[1], None]; zl += [p1[2], p2[2], None]

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1), opacity=0.1, hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0), showlegend=False
    )
    
    return fig










