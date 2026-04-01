def calculate_severity(bbox, image_width, image_height):
    x, y, w, h = bbox
    area = (w * h) / (image_width * image_height)
    
    if area < 0.05:
        return "Low"
    elif area <= 0.2:
        return "Medium"
    else:
        return "High"
