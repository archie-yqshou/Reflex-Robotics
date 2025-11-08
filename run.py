import matplotlib.pyplot as plt
from geometry import (
    parse_stl_ascii, 
    get_model_bounds, 
    slice_at_z, 
    deduplicate_segments_keep_one,
    build_contours_nearest_neighbor
)
from visualization import plot_layer, print_layer_info

# NOTE: Use this to choose the STL file you want to slice
STL_FILE = "STLs/Through cube.stl"


def main():    
    triangles_dict = parse_stl_ascii(STL_FILE)
    
    print(f"\nTriangle categorization:")
    print(f"  Regular triangles (walls/angled): {len(triangles_dict['regular'])}")
    print(f"  Bottom triangles (normal z=-1): {len(triangles_dict['bottom'])}")
    print(f"  Top triangles (normal z=1): {len(triangles_dict['top'])}")
    total_triangles = sum(len(triangles_dict[k]) for k in triangles_dict)
    print(f"  Total: {total_triangles}")
    
    # Get model bounds
    z_min, z_max = get_model_bounds(triangles_dict)
    print(f"\nModel bounds: Z from {z_min:.2f} to {z_max:.2f} mm")
    print(f"Model height: {z_max - z_min:.2f} mm")
    
    # NOTE: Use this to define the layers you want to slice at
    test_layers = [
        (z_min, "Bottom Layer"),
        (25, "25mm"),
        ((z_max - z_min) / 2 + z_min, "Middle Layer"),
        (z_max, "Top Layer")
    ]


    all_figures = []
    
    for z, label in test_layers:
        # Slice at this Z height
        segments, has_horizontal_faces = slice_at_z(triangles_dict, z, z_min, z_max)
        
        segments_unique = deduplicate_segments_keep_one(segments)
        
        contours = build_contours_nearest_neighbor(segments_unique, epsilon=0.05)
        
        print_layer_info(z, label, segments_unique, contours, z_min, z_max)
        
        # Create visualization
        if contours or segments_unique:
            fig = plot_layer(segments_unique, contours, z, title=label, has_horizontal_faces=has_horizontal_faces)
            all_figures.append(fig)
    
    print("\n" + "=" * 60)
    print("Displaying plots...")
    print("Close plot windows to exit.")
    print("=" * 60)
    
    # Display all plots
    plt.show()


if __name__ == "__main__":
    main()

