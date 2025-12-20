import numpy as np
import plotly.graph_objects as go

def _get_diamond_palette(grade):
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

def create_diamond_fig(carat, table_pct, depth_pct, color_grade, clarity_grade):
    # 1. Dimensions
    diameter = 6.5 * (carat ** (1/3)) 
    radius = diameter / 2
    total_depth = diameter * (depth_pct / 100)
    table_radius = radius * (table_pct / 100)
    
    # Proportions based on real diamond physics
    z_table = total_depth * 0.35
    z_crown_mid = z_table * 0.5 # For star facets
    z_girdle = 0
    z_culet = -total_depth * 0.65

    # 2. Vertex Definition (Manual Faceting)
    # We use 8-fold symmetry (The standard for Round Brilliant)
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    half_angles = angles + (np.pi / 8)
    
    # Vertex Groups
    v_table_center = [0, 0, z_table]
    v_table = [[table_radius * np.cos(a), table_radius * np.sin(a), z_table] for a in angles]
    v_crown_mid = [[radius * 0.8 * np.cos(a), radius * 0.8 * np.sin(a), z_crown_mid] for a in half_angles]
    v_girdle = [[radius * np.cos(a), radius * np.sin(a), z_girdle] for a in angles]
    v_culet = [0, 0, z_culet]

    # Combine all vertices into one list
    # Indices: 0(center), 1-8(table), 9-16(crown_mid), 17-24(girdle), 25(culet)
    vertices = [v_table_center] + v_table + v_crown_mid + v_girdle + [v_culet]
    verts_np = np.array(vertices)
    x, y, z = verts_np[:,0], verts_np[:,1], verts_np[:,2]

    # 3. Manual Triangle Definition (i, j, k)
    # This is what creates the "Triangular Shapes" you mentioned
    i_list, j_list, k_list = [], [], []

    for idx in range(8):
        nxt = (idx + 1) % 8
        # TABLE FACETS (Triangles)
        i_list.append(0); j_list.append(idx + 1); k_list.append(nxt + 1)
        
        # CROWN STAR FACETS (Triangles)
        # Table edge to crown mid
        i_list.append(idx + 1); j_list.append(idx + 9); k_list.append(nxt + 1)
        
        # BEZEL FACETS (Kites/Triangles)
        i_list.append(idx + 9); j_list.append(idx + 17); k_list.append(nxt + 17)
        i_list.append(idx + 9); j_list.append(nxt + 17); k_list.append(nxt + 1)
        
        # PAVILION FACETS (Long Triangles to Culet)
        i_list.append(idx + 17); j_list.append(25); k_list.append(nxt + 17)

    # 4. Generate the Figure
    fig = go.Figure()
    surf_c, core_c, heart_c = _get_diamond_palette(color_grade)

    # --- LAYER 1: INTERNAL CORE (SUBSTANCE) ---
    # We render the same triangular mesh but slightly smaller and more opaque
    fig.add_trace(go.Mesh3d(
        x=x*0.9, y=y*0.9, z=z*0.9,
        i=i_list, j=j_list, k=k_list,
        opacity=0.7, color=heart_c,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.1, specular=0.1),
        flatshading=True, hoverinfo='skip'
    ))

    # --- LAYER 2: THE EXTERNAL BRILLIANT SHELL (SHINE) ---
    # This layer has the sharp triangular facets that catch the glints
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i_list, j=j_list, k=k_list,
        opacity=0.2, color=surf_c,
        lighting=dict(
            ambient=0.1, diffuse=0.1, roughness=0.01, 
            specular=2.0, # Max Specular for the "bright things"
            fresnel=5.0   # Glowing edges
        ),
        flatshading=True, # Critical: makes every triangle a flat mirror
        name='Brilliant Cut Surface'
    ))

    # --- LAYER 3: THE CUT LINES ---
    # We draw lines only where the triangles meet
    xl, yl, zl = [], [], []
    for triangle in range(len(i_list)):
        # Get indices of the 3 points in the triangle
        pts = [i_list[triangle], j_list[triangle], k_list[triangle], i_list[triangle]]
        for p_idx in range(3):
            p1, p2 = vertices[pts[p_idx]], vertices[pts[p_idx+1]]
            xl += [p1[0], p2[0], None]; yl += [p1[1], p2[1], None]; zl += [p1[2], p2[2], None]

    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines',
        line=dict(color='white', width=1), opacity=0.15,
        hoverinfo='skip'
    ))

    # 5. Layout
    fig.update_layout(
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode='data', bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    return fig








