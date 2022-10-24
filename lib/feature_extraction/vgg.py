def get_vgg16():
    print('import vgg to extract 512 features from images')
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input

    base_model = VGG16(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    head = GlobalAveragePooling2D()(x)
    imgmodel = Model(inputs=base_model.input, outputs=head)  # this is the model we will train
    return imgmodel, image, preprocess_input
