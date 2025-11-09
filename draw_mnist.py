import pygame
import numpy as np
import torch
from cpt import CollaborativeStack
from scipy import ndimage

# ---- load model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("mnist_collabnet.pth", map_location=device, weights_only=False)
model.eval()

# ---- pygame setup ----
pygame.init()
W, H = 840, 840 
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Draw a digit (Enter=classify, R=reset, Q=quit)")
clock = pygame.time.Clock()

WHITE, BLACK = (255,255,255), (0,0,0)
screen.fill(BLACK)
radius = 24  # Increased brush size for better drawing
font = pygame.font.SysFont(None, 40)

def predict_digit(surface):
    # Get pixel data
    arr = pygame.surfarray.array3d(surface)
    
    # Convert to grayscale (0-255 range)
    gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114])
    
    # Transpose to correct orientation
    gray = np.transpose(gray)
    
    # Normalize to 0-1
    gray = gray / 255.0
    
    # Resize to 28x28 using a better method
    from scipy.ndimage import zoom
    h, w = gray.shape
    scale_h = 28.0 / h
    scale_w = 28.0 / w
    small_gray = zoom(gray, (scale_h, scale_w), order=1)
    
    # Ensure exact size
    small_gray = small_gray[:28, :28]
    
    # Center the digit (like MNIST does)
    # Find bounding box of drawn content
    threshold = 0.1
    rows = np.any(small_gray > threshold, axis=1)
    cols = np.any(small_gray > threshold, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the digit
        digit_img = small_gray[rmin:rmax+1, cmin:cmax+1]
        
        # Calculate scaling to fit in 20x20 (MNIST style)
        h, w = digit_img.shape
        scale = min(20.0/h, 20.0/w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        if new_h > 0 and new_w > 0:
            digit_resized = zoom(digit_img, (new_h/h, new_w/w), order=1)
            
            # Create centered 28x28 image
            centered = np.zeros((28, 28))
            y_offset = (28 - new_h) // 2
            x_offset = (28 - new_w) // 2
            centered[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
            
            small_gray = centered
    
    # Optional: show what the model sees
    import matplotlib.pyplot as plt
    plt.imshow(small_gray, cmap='gray')
    plt.title('What the model sees')
    plt.show()

    # Flatten and send to model
    x = small_gray.reshape(1, -1).astype(np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        outs = model.forward_until(x_tensor, upto_layer=len(model.branches) - 1)
        logits = outs[len(model.branches) - 1][0]
        
        # Print probabilities for debugging
        probs = torch.softmax(logits, dim=1)
        print("Probabilities:", probs.cpu().numpy()[0])
        
        digit = int(logits.argmax(dim=1).item())
    
    return digit

# ---- main loop ----
running = True
pred_text = ""
drawing = False
last_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_r:
                screen.fill(BLACK)
                pred_text = ""
            elif event.key == pygame.K_RETURN:
                digit = predict_digit(screen)
                pred_text = f"Predicted: {digit}"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                drawing = True
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                last_pos = None
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                # Draw smooth lines between points
                if last_pos is not None:
                    pygame.draw.line(screen, WHITE, last_pos, event.pos, radius*2)
                pygame.draw.circle(screen, WHITE, event.pos, radius)
                last_pos = event.pos

    # Clear previous text area and redraw
    if pred_text:
        pygame.draw.rect(screen, BLACK, (0, 0, W, 50))
        textsurf = font.render(pred_text, True, (255,0,0))
        screen.blit(textsurf, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()