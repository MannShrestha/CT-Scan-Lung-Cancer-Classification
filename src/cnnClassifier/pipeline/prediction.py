import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

        # Resolve the project root directory
        self.root_dir = os.getcwd()

        # Path to the trained model
        self.model_path = os.path.join(
            self.root_dir,
            "artifacts",
            "training",
            "model.h5"
        )

    def predict(self):

        # Check whether the trained model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # Load the trained CNN model
        model = load_model(self.model_path)

        # Check whether the uploaded image exists
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f"Image file not found: {self.filename}"
            )

        # Load and preprocess the image
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)

        # Normalise pixel values if the model was trained
        # using pixel values scaled to the range [0, 1]
        test_image = test_image / 255.0

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        # Generate prediction
        predictions = model.predict(test_image)

        result = np.argmax(predictions, axis=1)

        print("Prediction:", result)

        # Interpret prediction
        if result[0] == 1:
            prediction = "Normal"
        else:
            prediction = "Adenocarcinoma Cancer"

        return [{"image": prediction}]



# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os



# class PredictionPipeline:
#     def __init__(self,filename):
#         self.filename =filename


    
#     def predict(self):
#         ## load model
        
#         # model = load_model(os.path.join("artifacts","training", "model.h5"))
#         model = load_model(os.path.join("model", "model.h5"))

#         imagename = self.filename
#         test_image = image.load_img(imagename, target_size = (224,224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = np.argmax(model.predict(test_image), axis=1)
#         print(result)

#         if result[0] == 1:
#             prediction = 'Normal'
#             return [{ "image" : prediction}]
#         else:
#             prediction = 'Adenocarcinoma Cancer'
#             return [{ "image" : prediction}]


