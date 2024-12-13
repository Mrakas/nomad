import sys
sys.path.append('/home/marcus/workplace/nav_models/visualnav-transformer/deployment')
from src.api_explore import VisualNavigationAPI


#image_size = 96*96

def main():
    # Paths to your context images
    image_paths = [
        '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test1.jpg',
        '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test2.jpg',
        '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test3.jpg',
        '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test4.jpg'
    ]

    try:
        # Initialize the Visual Navigation API
        nav_api = VisualNavigationAPI(model_name="nomad")

        # Call the navigation method
        waypoint = nav_api.navigate(
            context_images=image_paths,
            num_samples=8,    # Optional: number of action samples
            waypoint_index=2  # Optional: which waypoint to use
        )

        print(f"Recommended Navigation Waypoint: {waypoint}")

    except Exception as e:
        print(f"Navigation Error: {e}")

if __name__ == "__main__":
    main()