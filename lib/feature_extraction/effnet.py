def get_effnetb2():
    print('import efficientnet to extract 1408 features from images')
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.efficientnet import preprocess_input

    base_model = EfficientNetB2(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    head = GlobalAveragePooling2D()(x)
    imgmodel = Model(inputs=base_model.input, outputs=head)  # this is the model we will train

    return imgmodel, image, preprocess_input
