import numpy as np
import plotly.graph_objects as go

def _get_diamond_palette(grade):
    """Maps color grade to a 3-layer palette for depth."""
    palettes = {
        'D': ('#F0F8FF', '#E0F0FF', '#B0D0FF'), 
        'E': ('#F8F8FF', '#F0F0F8', '#D0D0E0'), 
        'F': ('#FFFFFF', '#F5F5F5', '#E0E0E0'), 
        'G': ('#FFFFF0', '#FFFEE0', '#E8E8C0'), 
        'H': ('#FFFACD', '#FFF5B0', '#E5D090'), 
        'I': ('#FFF8DC', '#F5EBC0', '#D2B48C'), 
        'J': '#F0E68C'
    }
    res = palettes.get(grade, ('#FFFFFF', '#F5F5F5', '#E0E0E0'))
    return res if isinstance(res, tuple) else (res, res, res)

def _generate_inclusions(clarity_grade, radius, table_radius, z_table, z_culet):
    """
    Geometry-Aware Inclusion Engine.
    Generates flaws that are mathematically guaranteed to be INSIDE the diamond.
    """
    severity = {
        'IF': 0, 'VVS1': 2, 'VVS2': 6, 
        'VS1': 15, 'VS2': 35, 
        'SI1': 80, 'SI2': 160, 'I1': 350
    }
    count = severity.get(clarity_grade, 0)
    if count == 0: return None, None, None
        
    rng = np.random.default_rng()
    inc_x, inc_y, inc_z = [], [], []
    
    for _ in range(count):
        # 1. Pick a random height z between culet and table
        z = rng.uniform(z_culet, z_table)
        
        # 2. Calculate the maximum allowed radius at this specific height z
        # This ensures the dots don't 'poke out' of the sloping facets
        if z >= 0:
            # Crown Area: Linear interpolation between girdle radius and table radius
            # Height ranges from 0 to z_table
            t = z / z_table
            max_r = radius + (table_radius - radius) * t
        else:
            # Pavilion Area: Linear interpolation between girdle radius and culet (0)
            # Height ranges from 0 to z_culet
            t = z / z_culet
            max_r = radius * (1 - t)
            
        # 3. Generate random point within that circular cross-section (with a small safety buffer)
        r = np.sqrt(rng.uniform(0, 1)) * (max_r * 0.95)
        theta = rng.uniform(0, 2 * np.pi)
        
        inc_x.append(r * np.cos(theta))
        inc_y.append(r * np.sin(theta))
        inc_z.append(z)
        
    return inc_x, inc_y, inc_z

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. Dimensions & Proportions
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    z_table = total_depth * 0.35
    z_crown_mid = z_table * 0.5 
    z_girdle = 0
    z_culet = -total_depth * 0.65

    # 2. Vertex Definition (8-fold Brilliant Symmetry)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    half_angles = angles + (np.pi / 8)
    
    v_table_center = [0, 0, z_table]
    v_table = [[table_radius * np.cos(a), table_radius * np.sin(a), z_table] for a in angles]
    v_crown_mid = [[radius * 0.85 * np.cos(a), radius * 0.85 * np.sin(a), z_crown_mid] for a in half_angles]
    v_girdle = [[radius * np.cos(a), radius * np.sin(a), z_girdle] for a in angles]
    v_culet_pt = [0, 0, z_culet]

    vertices = [v_table_center] + v_table + v_crown_mid + v_girdle + [v_culet_pt]
    verts_np = np.array(vertices)
    x, y, z = verts_np[:,0], verts_np[:,1], verts_np[:,2]

    # 3. Manual Triangle Mesh (i, j, k)
    i_list, j_list, k_list = [], [], []
    for idx in range(8):
        nxt = (idx + 1) % 8
        # Table
        i_list.append(0); j_list.append(idx + 1); k_list.append(nxt + 1)
        # Crown
        i_list.append(idx + 1); j_list.append(idx + 9); k_list.append(nxt + 1)
        # Bezel/Girdle
        i_list.append(idx + 9); j_list.append(idx + 17); k_list.append(nxt + 17)
        i_list.append(idx + 9); j_list.append(nxt + 17); k_list.append(nxt + 1)
        # Pavilion
        i_list.append(idx + 17); j_list.append(25); k_list.append(nxt + 17)

    # 4. Generate Figure
    fig = go.Figure()
    surf_c, core_c, heart_c = _get_diamond_palette(color_grade)

    # --- LAYER 1: INTERNAL SUBSTANCE ---
    fig.add_trace(go.Mesh3d(
        x=x*0.9, y=y*0.9, z=z*0.9, i=i_list, j=j_list, k=k_list,
        opacity=0.75, color=heart_c,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.1, specular=0.1),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 2: BRILLIANT EXTERIOR ---
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=i_list, j=j_list, k=k_list,
        opacity=0.2, color=surf_c,
        lighting=dict(ambient=0.1, diffuse=0.1, roughness=0.01, specular=2.0, fresnel=5.0),
        flatshading=True, name='Surface'
    ))

    # --- LAYER 3: INCLUSIONS (The Flaws) ---
    ix, iy, iz = _generate_inclusions(clarity_grade, radius, table_radius, z_table, z_culet)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz,
            mode='markers',
            marker=dict(
                size=3, 
                color='rgba(40, 40, 40, 0.8)', # Dark carbon spots
                symbol='circle'
            ),
            name='Inclusions/Flaws'
        ))

    # --- LAYER 4: FACET OUTLINES ---
    xl, yl, zl = [], [], []
    for t_idx in range(len(i_list)):
        pts = [i_list[t_idx], j_list[t_idx], k_list[t_idx], i_list[t_idx]]
        for p in range(3):
            p1, p2 = vertices[pts[p]], vertices[pts[p+1]]
            xl += [p1[0], p2[0], None]; yl += [p1[1], p2[1], None]; zl += [p1[2], p2[2], None]

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1), opacity=0.15, hoverinfo='skip'
    ))

    # 5. Layout
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0), showlegend=False
    )
    
    return fig









