def get_resnet50():
    print('import resnet to extract 2048 features from images')
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input

    base_model = ResNet50(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    head = GlobalAveragePooling2D()(x)
    imgmodel = Model(inputs=base_model.input, outputs=head)  # this is the model we will train

    return imgmodel, image, preprocess_input
