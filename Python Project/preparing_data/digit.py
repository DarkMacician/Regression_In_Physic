from google.cloud import vision
import io

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()


def detect_text_from_image(image_path):
    # Read the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Prepare the image to send to the API
    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Output the detected text
    if texts:
        print("Detected text:")
        for text in texts:
            print(f'"{text.description}"')
        return texts[0].description  # Return the first detected text (full text)

    if response.error.message:
        raise Exception(f'Error from Vision API: {response.error.message}')
    else:
        print("No text detected.")
        return None


# Path to your handwritten digit image
image_path = 'path_to_your_digit_image.png'
detected_text = detect_text_from_image(image_path)
print("Predicted Text:", detected_text)
