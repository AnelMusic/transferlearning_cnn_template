import matplotlib.pyplot as plt
import torch
from PIL import Image


def predict(model, transformation, device, idx_to_class, test_directory):
    """Predict Class for Image.

    Args:
        model (AlexNet): Model
        transformation (transforms.Compose(): Transformation
        device (str): Hardware Device
        idx_to_class (dict): Label index dictionary
        test_directory (os.Path): Path to test data
    """
    test_image = Image.open(test_directory + "/apu.jpg")
    plt.imshow(test_image)
    plt.show()

    test_image_tensor = transformation(test_image)
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).to(device)

    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)

        for i in range(3):
            print(
                "Predcition",
                i + 1,
                ":",
                idx_to_class[topclass.cpu().numpy()[0][i]],
                ", Score: ",
                topk.cpu().numpy()[0][i],
            )
