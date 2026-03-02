import numpy as np
import cv2
import matplotlib.pyplot as plt

def convert_images(path_photo):
    width = 750
    height = 750
    dim = (width, height)

    # Load the images
    img_bgr = cv2.imread(path_photo)

    # Convert to RGB for correct colors in Matplotlib and analysis
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Force resize
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Check if images loaded correctly
    if img is None:
        print("Error: Could not load image. Check the file path!")
    else:
        print(f"Success: '{path_photo}' loaded. Resized shape: {img.shape}")

        # Display using Matplotlib
        # Simple Background Image
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title("Loaded Image Verification")
        plt.axis('off')
        plt.show()

        return img

def repetitive_kmeans(img, k, reps=100):
    h, w, c = img.shape
    pixels = img.reshape((-1, 3)).astype(np.float32)
    
    # M stores cluster assignments: size [total_pixels, repetitions]
    M = np.zeros((h * w, reps), dtype=np.uint8)

    print(f"Starting {reps} repetitions for k={k}...")

    for i in range(reps):
        # Randomly choose k pixels
        random_indices = np.random.choice(pixels.shape[0], k, replace=False)
        initial_centroids = pixels[random_indices]

        # Perform k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Pass initial_centroids to 'centers'. 
        # Set flags to cv2.KMEANS_USE_INITIAL_LABELS
        _labels = np.zeros((pixels.shape[0], 1), dtype=np.int32)
        
        _, labels, centers = cv2.kmeans(pixels, k, _labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS, initial_centroids)

        # Align cluster numbers (Sort by Red value)
        sorted_indices = np.argsort(centers[:, 0]) 
        
        # Create a mapping: old_label -> new_sorted_label
        lookup = np.zeros(k, dtype=np.uint8)
        for new_idx, old_idx in enumerate(sorted_indices):
            lookup[old_idx] = new_idx
        
        aligned_labels = lookup[labels.flatten()]

        # Store results
        M[:, i] = aligned_labels

    # Calculate Probability Maps
    prob_maps = []
    for cluster_id in range(k):
        counts = np.sum(M == cluster_id, axis=1)
        prob_map = (counts / reps).reshape((h, w))
        prob_maps.append(prob_map)

    return prob_maps

def main():
    # Define the paths to the photos
    photo_1 = 'Hand_Simple_Background.jpg'
    photo_2 = 'Hand_Complex_Background.jpg'

    img1 = convert_images(photo_1)
    img2 = convert_images(photo_2)

    # List of k-values to test
    k_values = [2, 3, 5]

    # Analyze the first image
    for k in k_values:
        probs = repetitive_kmeans(img1, k, reps=100)
        
        # Display the results
        fig, axes = plt.subplots(1, k + 1, figsize=(15, 5))
        if k == 1: axes = [axes]
        
        for idx, p_map in enumerate(probs):
            axes[idx].imshow(p_map, cmap='hot')
            axes[idx].set_title(f"P(Cluster {idx})")
            axes[idx].axis('off')
        
        # Display the detected skin region
        skin_mask = probs[-1] > 0.5
        masked_img = img1.copy()
        masked_img[~skin_mask] = masked_img[~skin_mask] // 3
        axes[k].imshow(masked_img)
        axes[k].set_title("Detected Skin Region")
        axes[k].axis('off')

        plt.suptitle(f"Probability Maps for Simple Background k={k}")

        # Saving logicx
        save_path = f"output_simple_k{k}.jpg"
        plt.savefig(save_path, format='jpg', bbox_inches='tight')

        plt.show()
        plt.close(fig)

    for k in k_values:
        probs = repetitive_kmeans(img2, k, reps=100)
        
        # Display the results
        fig, axes = plt.subplots(1, k + 1, figsize=(15, 5))
        if k == 1: axes = [axes]
        
        for idx, p_map in enumerate(probs):
            axes[idx].imshow(p_map, cmap='hot')
            axes[idx].set_title(f"P(Cluster {idx})")
            axes[idx].axis('off')
        
        # Display the detected skin region
        skin_mask = probs[-1] > 0.5
        masked_img = img2.copy()
        masked_img[~skin_mask] = masked_img[~skin_mask] // 3
        axes[k].imshow(masked_img)
        axes[k].set_title("Detected Skin Region")
        axes[k].axis('off')

        plt.suptitle(f"Probability Maps for Complex Background k={k}")

        save_path = f"output_complex_k{k}.jpg"
        plt.savefig(save_path, format='jpg', bbox_inches='tight')

        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    main()