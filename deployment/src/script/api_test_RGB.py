import sys
sys.path.append('/home/marcus/workplace/nav_models/visualnav-transformer/deployment')
from src.api_explore_RGB import VisualNavigationAPI
import PIL.Image as PILImage

#image_size = 96*96

def main():
    # Paths to your context images
    #给img的path拿到rgb
    single_img_path = '/home/marcus/workplace/nav_models/visualnav-transformer/imgs/test1.jpg'
    single_img = PILImage.open(single_img_path).convert("RGB")
    
    nav_api = VisualNavigationAPI(model_name="nomad")
    # Call the navigation method
    waypoint = nav_api.navigate(
        single_img=single_img,
        num_samples=8,    # Optional: number of action samples
        waypoint_index=2  # Optional: which waypoint to use
    )

    print(f"Recommended Navigation Waypoint: {waypoint}")

if __name__ == "__main__":
    main()