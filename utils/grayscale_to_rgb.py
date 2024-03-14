from torchvision.transforms.functional import to_pil_image


def grayscale_to_rgb(tensor):
    try:
        return to_pil_image(tensor, mode="RGB")
    except:
        return to_pil_image(tensor, mode="L")
